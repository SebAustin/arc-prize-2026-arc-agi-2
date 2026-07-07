"""Base fine-tuning on the synthetic corpus (GPU, Kaggle).

Trains a LoRA adapter over the synthetic (prompt, completion) corpus so the model
learns the ARC I/O format and a broad transformation library BEFORE per-task TTT.
The resulting adapter is staged as a Kaggle Dataset and loaded by `HFModel(...,
adapter_path=...)` at inference; TTT then adapts further on top of it.

torch/transformers/peft import lazily — this module is import-safe off-GPU but
`finetune()` only runs where they're installed (Kaggle). Mirrors the masking /
loop style of `solvers/llm/ttt.py` so the two training paths stay consistent.

Kaggle's 12h kernel cap means a run can be interrupted (quota, restart, crash)
partway through 50k examples; `resume=True` (default) makes re-running the same
call idempotent-ish: it picks up at the last completed epoch and fast-forwards
past already-seen batches within the current epoch, rather than re-training from
scratch. Progress is logged (not printed) every `_LOG_EVERY_STEPS` steps so the
Kaggle log gives a live steps/s + ETA heartbeat — the only observability window
into a run that takes hours.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..solvers.llm.ttt_data import TrainExample

_log = logging.getLogger(__name__)

_LOG_EVERY_STEPS = 50
_STATE_FILENAME = "state.json"
_CHECKPOINT_DIRNAME = "checkpoint"


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
    dtype: str | None = None  # None = hardware-adaptive (see _resolve_dtype)
    checkpoint_every_steps: int = 200
    resume: bool = True


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


def _pick_dtype(cfg: TrainConfig, torch_module) -> str:
    """Resolve the training compute dtype the same way inference does.

    Extracted as a pure(ish) wrapper around `_resolve_dtype` so the plumbing
    (config override reaches the shared hardware-adaptive picker) is testable
    with a stub `torch_module` and no real torch installed.
    """
    from ..solvers.llm.model import _resolve_dtype  # noqa: PLC0415

    return _resolve_dtype(torch_module, cfg.dtype)


def select_examples(
    examples: list[TrainExample], max_examples: int | None, seed: int
) -> list[TrainExample]:
    """Deterministically pick a subset of `examples` for a smaller/faster run.

    A generator-ordered head-slice (`examples[:max_examples]`) is biased — the
    synthetic corpus is built round-robin over generators, so an unshuffled
    prefix under-represents whichever generators appear later. Shuffling with a
    seeded PRNG first, then slicing, keeps the subset representative while
    staying fully deterministic (same `seed` -> same subset, every time).
    """
    if max_examples is None or max_examples >= len(examples):
        return list(examples)
    import random  # noqa: PLC0415 — stdlib only; keeps this helper torch-free

    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    keep = sorted(order[:max_examples])  # preserve original relative order
    return [examples[i] for i in keep]


def _save_state(output_dir: str | Path, epoch: int, step: int, examples_seen: int) -> Path:
    """Persist resume bookkeeping next to the checkpoint. `epoch` is the index of
    the epoch IN PROGRESS (0-based); `step` is the optimizer step reached within
    it (0-based count of completed batches, pre-grad-accum-collapse)."""
    path = Path(output_dir) / _STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "step": step, "examples_seen": examples_seen}
    path.write_text(json.dumps(payload))
    return path


def _load_state(output_dir: str | Path) -> dict | None:
    """Load resume bookkeeping, or None if this is a fresh run (no prior state)."""
    path = Path(output_dir) / _STATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _batches_to_skip(order_len: int, batch_size: int, resume_step: int) -> int:
    """How many whole batches to fast-forward past when resuming mid-epoch.

    `resume_step` is the count of batches already completed in that epoch (from
    the saved state); since the epoch's shuffle order is deterministic
    (seed-per-epoch `torch.randperm`), skipping the first `resume_step` batches
    of the SAME order reproduces exactly where training left off. Clamped to
    the number of batches the epoch actually has, so a corrupt/stale state file
    can't skip past the end and silently no-op the whole epoch.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    total_batches = (order_len + batch_size - 1) // batch_size
    return max(0, min(resume_step, total_batches))


def finetune(
    base_model_path: str,
    examples: list[TrainExample],
    output_dir: str,
    config: TrainConfig | None = None,
    device: str = "cuda",
    max_examples: int | None = None,
) -> str:
    """Fine-tune a LoRA adapter on `examples`; save to `output_dir`; return it.

    `max_examples`, if set, trains on a deterministic representative subset
    (see `select_examples`) — useful for a fast canary before committing to the
    full corpus. `config.resume` (default True) makes re-invoking this function
    against the same `output_dir` continue from the last checkpoint instead of
    restarting, which matters on Kaggle's 12h kernel cap.
    """
    import torch  # noqa: PLC0415
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    examples = select_examples(examples, max_examples, cfg.seed)
    _log.info("training on %d examples (max_examples=%s)", len(examples), max_examples)

    dtype = _pick_dtype(cfg, torch)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=getattr(torch, dtype),
        device_map=device,
        use_safetensors=True,  # refuse pickle .bin checkpoints (RCE surface)
    )

    checkpoint_dir = Path(output_dir) / _CHECKPOINT_DIRNAME
    state = _load_state(output_dir) if cfg.resume else None
    if state is not None and checkpoint_dir.exists():
        _log.info("resuming from checkpoint: %s", state)
        # `is_trainable=True` re-attaches the adapter as a trainable PEFT model
        # (the default load is inference-only/frozen) so training can continue
        # on the exact weights the previous run left off at, instead of
        # re-initializing a fresh LoRA from scratch.
        model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=True)
    else:
        state = {"epoch": 0, "step": 0, "examples_seen": 0}
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

    start_epoch = state["epoch"]
    examples_seen = state["examples_seen"]
    t_start = time.monotonic()
    total_steps_run = 0

    for epoch in range(start_epoch, cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(rows), generator=gen).tolist()
        skip_batches = (
            _batches_to_skip(len(order), cfg.batch_size, state["step"])
            if epoch == start_epoch
            else 0
        )
        optim.zero_grad()
        for step, start in enumerate(range(0, len(order), cfg.batch_size)):
            if step < skip_batches:
                continue
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

            examples_seen += len(idx)
            total_steps_run += 1
            if total_steps_run % _LOG_EVERY_STEPS == 0:
                elapsed = time.monotonic() - t_start
                rate = total_steps_run / elapsed if elapsed > 0 else 0.0
                total_batches_left = (
                    (cfg.epochs - epoch - 1) * ((len(order) + cfg.batch_size - 1) // cfg.batch_size)
                    + max(0, ((len(order) + cfg.batch_size - 1) // cfg.batch_size) - step - 1)
                )
                eta_s = total_batches_left / rate if rate > 0 else float("inf")
                _log.info(
                    "epoch=%d step=%d loss=%.4f examples_seen=%d steps/s=%.2f eta_s=%.0f",
                    epoch + 1,
                    step + 1,
                    float(out.loss),
                    examples_seen,
                    rate,
                    eta_s,
                )
            if (step + 1) % cfg.checkpoint_every_steps == 0:
                model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
                _save_state(output_dir, epoch, step + 1, examples_seen)
                _log.info("checkpoint saved at epoch=%d step=%d", epoch, step + 1)

        _log.info("epoch %d/%d done", epoch + 1, cfg.epochs)
        model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
        _save_state(output_dir, epoch + 1, 0, examples_seen)

    model.save_pretrained(output_dir, safe_serialization=True)  # write .safetensors
    tokenizer.save_pretrained(output_dir)
    return output_dir
