"""Base fine-tuning on the synthetic corpus (GPU, Kaggle).

Trains a LoRA adapter over the synthetic (prompt, completion) corpus so the model
learns the ARC I/O format and a broad transformation library BEFORE per-task TTT.
The resulting adapter is staged as a Kaggle Dataset and loaded by `HFModel(...,
adapter_path=...)` at inference; TTT then adapts further on top of it.

torch/transformers/peft import lazily — this module is import-safe off-GPU but
`finetune()` only runs where they're installed (Kaggle). Mirrors the masking /
loop style of `solvers/llm/ttt.py` so the two training paths stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..solvers.llm.ttt_data import TrainExample


@dataclass(frozen=True)
class TrainConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 1e-4
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 2048
    seed: int = 0


def _tokenize(tokenizer, examples: list[TrainExample], max_seq_len: int):
    """Tokenise examples with the prompt tokens masked out of the loss (-100)."""
    eos = tokenizer.eos_token or ""
    rows = []
    for ex in examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(ex.completion + eos, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + comp_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:max_seq_len]
        rows.append((ids, labels))
    return rows


def _collate(batch, pad_id, torch):
    max_len = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
    )


def finetune(
    base_model_path: str,
    examples: list[TrainExample],
    output_dir: str,
    config: TrainConfig | None = None,
    device: str = "cuda",
) -> str:
    """Fine-tune a LoRA adapter on `examples`; save to `output_dir`; return it."""
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_safetensors=True,  # refuse pickle .bin checkpoints (RCE surface)
    )
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    rows = _tokenize(tokenizer, examples, cfg.max_seq_len)
    pad_id = tokenizer.pad_token_id
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(cfg.epochs):
        order = torch.randperm(len(rows), generator=gen).tolist()
        optim.zero_grad()
        for step, start in enumerate(range(0, len(order), cfg.batch_size)):
            idx = order[start : start + cfg.batch_size]
            input_ids, labels, attn = _collate([rows[i] for i in idx], pad_id, torch)
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
            )
            (out.loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad()
        print(f"epoch {epoch + 1}/{cfg.epochs} done")

    model.save_pretrained(output_dir, safe_serialization=True)  # write .safetensors
    tokenizer.save_pretrained(output_dir)
    return output_dir
