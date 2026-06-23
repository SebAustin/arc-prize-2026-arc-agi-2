"""Test-time training: per-task LoRA adaptation, then transduction.

`TTTRunner` is the seam between the (CPU-testable) orchestration and the
(GPU-only) gradient step:

  * `MockTTTRunner` performs no training and returns the base model — lets the
    whole adapt -> infer -> vote flow run and be unit-tested on CPU.
  * `LoraTTTRunner` fine-tunes a fresh LoRA adapter on the task's corpus, then
    serves inference through the adapted model. torch/peft import lazily; this
    path is validated on Kaggle.

`TTTSolver` ties it together: build the corpus (CPU), adapt (runner), then
delegate to the existing `LLMSolver` for augmentation-based inference + voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...io.loader import Task
from ..base import Candidates, Solver
from .model import LanguageModel
from .solver import LLMSolver
from .ttt_data import TrainExample, build_ttt_examples


@dataclass(frozen=True)
class TTTConfig:
    """LoRA + training hyperparameters for per-task adaptation."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
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
    max_steps: int = 64
    batch_size: int = 2
    max_seq_len: int = 2048
    seed: int = 0


class TTTRunner(Protocol):
    """Adapts a base model to a task's corpus and serves inference."""

    def adapt(self, examples: list[TrainExample]) -> LanguageModel:
        """Train on `examples` and return a model ready for inference."""
        ...

    def reset(self) -> None:
        """Restore the base model for the next task."""
        ...


class MockTTTRunner:
    """No-op runner for CPU tests: records the corpus, returns the base model."""

    def __init__(self, base_model: LanguageModel):
        self.base_model = base_model
        self.last_examples: list[TrainExample] = []
        self.adapt_calls = 0
        self.reset_calls = 0

    def adapt(self, examples: list[TrainExample]) -> LanguageModel:
        self.last_examples = examples
        self.adapt_calls += 1
        return self.base_model

    def reset(self) -> None:
        self.reset_calls += 1


class TTTSolver(Solver):
    """Per-task test-time training + transduction."""

    name = "llm_ttt"

    def __init__(
        self,
        runner: TTTRunner,
        llm_kwargs: dict | None = None,
        ttt_data_kwargs: dict | None = None,
    ):
        self.runner = runner
        self.llm_kwargs = llm_kwargs or {}
        self.ttt_data_kwargs = ttt_data_kwargs or {}

    def solve(self, task: Task, budget_s: float) -> Candidates:
        examples = build_ttt_examples(task, **self.ttt_data_kwargs)
        adapted = self.runner.adapt(examples)
        try:
            return LLMSolver(adapted, **self.llm_kwargs).solve(task, budget_s)
        finally:
            self.runner.reset()


class LoraTTTRunner:
    """GPU LoRA fine-tuner (torch/peft, lazy import). Validated on Kaggle.

    Trains a fresh adapter on the task corpus in place, serving inference through
    the same `HFModel`, then unloads the adapter on `reset()` to restore the base
    weights for the next task.
    """

    def __init__(self, hf_model, config: TTTConfig | None = None):
        self.hf_model = hf_model
        self.config = config or TTTConfig()
        self._base = hf_model.model  # original (un-adapted) module

    def adapt(self, examples: list[TrainExample]) -> LanguageModel:
        import torch  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415

        cfg = self.config
        torch.manual_seed(cfg.seed)
        tokenizer = self.hf_model.tokenizer

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(self._base, lora)
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        batches = self._tokenize(tokenizer, examples, cfg.max_seq_len)
        optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
        )
        device = self.hf_model.device
        rng = torch.Generator().manual_seed(cfg.seed)
        for _step in range(cfg.max_steps):
            batch = self._sample_batch(batches, cfg.batch_size, rng)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            optim.step()
            optim.zero_grad()

        model.eval()
        self.hf_model.model = model
        return self.hf_model

    def reset(self) -> None:
        model = self.hf_model.model
        if hasattr(model, "unload"):
            self.hf_model.model = model.unload()  # drop adapter, restore base
        else:  # pragma: no cover
            self.hf_model.model = self._base

    def _tokenize(self, tokenizer, examples, max_seq_len):
        eos = tokenizer.eos_token or ""
        rows = []
        for ex in examples:
            prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            comp_ids = tokenizer(ex.completion + eos, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + comp_ids)[:max_seq_len]
            labels = ([-100] * len(prompt_ids) + comp_ids)[:max_seq_len]
            rows.append((ids, labels))
        return rows

    def _sample_batch(self, rows, batch_size, rng):
        import torch  # noqa: PLC0415

        n = len(rows)
        idx = torch.randint(0, n, (min(batch_size, n),), generator=rng).tolist()
        chosen = [rows[i] for i in idx]
        max_len = max(len(ids) for ids, _ in chosen)
        pad_id = 0
        input_ids, labels, attn = [], [], []
        for ids, lab in chosen:
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
