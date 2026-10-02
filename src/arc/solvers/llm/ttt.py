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

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from ...io.loader import Task
from ..base import Candidates, Solver
from .model import LanguageModel
from .solver import LLMSolver
from .ttt_data import TrainExample, build_ttt_examples

_log = logging.getLogger(__name__)


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
    # Once past the adapt deadline, at least this many steps still run (bounded
    # overrun) so a briefly-late clock doesn't yield an untrained adapter.
    min_steps: int = 8
    batch_size: int = 2
    max_seq_len: int = 2048
    seed: int = 0


class TTTRunner(Protocol):
    """Adapts a base model to a task's corpus and serves inference."""

    def adapt(
        self, examples: list[TrainExample], deadline_s: float | None = None
    ) -> LanguageModel:
        """Train on `examples` and return a model ready for inference.

        `deadline_s` is an absolute `time.monotonic()` deadline: training should
        stop early once it passes (after `min_steps`), so adaptation cannot eat
        the whole per-task budget and starve the decode phase.
        """
        ...

    def reset(self) -> None:
        """Restore the base model for the next task."""
        ...


class MockTTTRunner:
    """No-op runner for CPU tests: records the corpus, returns the base model."""

    def __init__(self, base_model: LanguageModel):
        self.base_model = base_model
        self.last_examples: list[TrainExample] = []
        self.last_deadline_s: float | None = None
        self.adapt_calls = 0
        self.reset_calls = 0

    def adapt(
        self, examples: list[TrainExample], deadline_s: float | None = None
    ) -> LanguageModel:
        self.last_examples = examples
        self.last_deadline_s = deadline_s
        self.adapt_calls += 1
        return self.base_model

    def reset(self) -> None:
        self.reset_calls += 1


class TTTSolver(Solver):
    """Per-task test-time training + transduction.

    The per-task budget is split: adaptation gets at most `ttt_fraction` of it
    (enforced via the runner's deadline), so decode + selection always retain
    the rest. Without this split, 64 unconditioned LoRA steps can consume most
    of a 150 s budget and leave only 1-3 of the augmented decodes any time.
    """

    name = "llm_ttt"

    def __init__(
        self,
        runner: TTTRunner,
        llm_kwargs: dict | None = None,
        ttt_data_kwargs: dict | None = None,
        ttt_fraction: float = 0.4,
    ):
        self.runner = runner
        self.llm_kwargs = llm_kwargs or {}
        self.ttt_data_kwargs = ttt_data_kwargs or {}
        self.ttt_fraction = ttt_fraction
        self.last_telemetry: dict = {}

    def solve(self, task: Task, budget_s: float) -> Candidates:
        t0 = time.monotonic()
        examples = build_ttt_examples(task, **self.ttt_data_kwargs)
        corpus_s = time.monotonic() - t0
        try:
            adapt_deadline = t0 + self.ttt_fraction * budget_s
            t1 = time.monotonic()
            adapted = self.runner.adapt(examples, deadline_s=adapt_deadline)
            adapt_s = time.monotonic() - t1
            # Charge corpus-build + adaptation time against this solver's budget
            # so the inner transduction gets the time that is *actually* left,
            # not a fresh full budget (which silently overran the per-task cap).
            remaining = max(0.0, budget_s - (time.monotonic() - t0))
            inner = LLMSolver(adapted, **self.llm_kwargs)
            result = inner.solve(task, remaining)
            self.last_telemetry = {
                "task_id": task.task_id,
                "corpus_s": round(corpus_s, 2),
                "corpus_n": len(examples),
                "adapt_s": round(adapt_s, 2),
                "decode_budget_s": round(remaining, 2),
                **getattr(inner, "last_telemetry", {}),
            }
            _log.info("ttt telemetry %s", self.last_telemetry)
            return result
        finally:
            # `adapt()` is inside the try so reset() runs even if adaptation
            # raises (e.g. CUDA OOM), restoring the shared base model for the
            # next task instead of leaving it wrapped in a partial adapter.
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

    def adapt(
        self, examples: list[TrainExample], deadline_s: float | None = None
    ) -> LanguageModel:
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
        # `get_peft_model` injects LoRA layers into `self._base`'s module tree by
        # reference, so a failure partway through training must unload the partial
        # adapter — otherwise the shared model stays corrupted for later tasks.
        model = get_peft_model(self._base, lora)
        try:
            model.train()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            batches = self._tokenize(tokenizer, examples, cfg.max_seq_len)
            optim = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=cfg.learning_rate,
            )
            device = self.hf_model.device
            rng = torch.Generator().manual_seed(cfg.seed)
            steps_run = 0
            for _step in range(cfg.max_steps):
                # Deadline check BEFORE the step: past-deadline entry -> a clean
                # 0-step skip (a fresh LoRA has B=0, so the model is functionally
                # the base); once running, min_steps bounds the overrun.
                past_floor = _step == 0 or _step >= cfg.min_steps
                if (
                    deadline_s is not None
                    and past_floor
                    and time.monotonic() >= deadline_s
                ):
                    break
                batch = self._sample_batch(batches, cfg.batch_size, rng)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                out.loss.backward()
                optim.step()
                optim.zero_grad()
                steps_run += 1

            if steps_run < cfg.max_steps:
                _log.info(
                    "ttt adapt stopped at %d/%d steps (deadline)",
                    steps_run,
                    cfg.max_steps,
                )
            model.eval()
            self.hf_model.model = model
            return self.hf_model
        except Exception:
            self.hf_model.model = (
                model.unload() if hasattr(model, "unload") else self._base
            )
            raise

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
