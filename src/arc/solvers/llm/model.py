"""Language-model abstraction for transduction.

Defines a minimal `LanguageModel` protocol (generate completions; score a
completion's log-likelihood) plus two implementations:

  * `MockModel` — pure-Python, CPU, no torch. Deterministic; applies a
    configurable grid `transform` to the prompt's final test input so the entire
    augment -> infer -> invert -> vote loop is unit-testable without a GPU.
  * `HFModel` — the real model (Qwen2.5 by default), loaded from a local path
    (the Kaggle dataset mount). torch/transformers are imported lazily so the
    CPU dev environment never needs them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...io.grid import Grid
from ...serialize.prompt import extract_last_input
from ...serialize.tokenizer import grid_to_str

_log = logging.getLogger(__name__)


@runtime_checkable
class LanguageModel(Protocol):
    """What the LLM solver needs from any model backend."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        num_samples: int = 1,
        temperature: float = 0.0,
        max_time_s: float | None = None,
    ) -> list[str]:
        """Return `num_samples` completion strings for `prompt`. `max_time_s`, if
        given, caps decode wall-clock so a stalled generation cannot blow the
        per-task budget."""
        ...

    def score(self, prompt: str, completion: str) -> float:
        """Return the mean per-token log-probability of `completion` given
        `prompt` (higher = more likely). Used for candidate selection."""
        ...

    def score_sum(self, prompt: str, completion: str) -> float:
        """Return the SUMMED log-probability of `completion` given `prompt`.

        Product-of-experts selection multiplies probabilities across prompts,
        i.e. sums log-probs — the mean-normalized `score` cannot be summed
        across differently-sized completions without bias."""
        ...


class MockModel:
    """Deterministic CPU stand-in. Predicts `transform(test_input)` for the
    prompt's final Input block (identity by default)."""

    def __init__(self, transform: Callable[[Grid], Grid] | None = None):
        self.transform = transform or (lambda g: g)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        num_samples: int = 1,
        temperature: float = 0.0,
        max_time_s: float | None = None,
    ) -> list[str]:
        grid = extract_last_input(prompt)
        if grid is None:
            return [""] * num_samples
        try:
            predicted = self.transform(grid)
        except Exception:
            predicted = grid
        return [grid_to_str(predicted)] * num_samples

    def score(self, prompt: str, completion: str) -> float:
        # Deterministic pseudo-score: prefer completions that parse to the
        # transform of the prompt's input (exact match -> 0.0, else -1.0).
        grid = extract_last_input(prompt)
        if grid is None:
            return -1.0
        try:
            target = grid_to_str(self.transform(grid))
        except Exception:
            return -1.0
        return 0.0 if completion.strip() == target.strip() else -1.0

    def score_sum(self, prompt: str, completion: str) -> float:
        # Sum-scaled variant of the pseudo-score (match -> 0.0, else -10.0), so
        # PoE tests can add scores across augmented prompts meaningfully.
        return 0.0 if self.score(prompt, completion) == 0.0 else -10.0

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        max_time_s: float | None = None,
    ) -> list[str]:
        """One greedy completion per prompt (loop — parity with HFModel's API)."""
        return [
            self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature)[0]
            for p in prompts
        ]


def _resolve_dtype(torch_module, override: str | None) -> str:
    """Pick the compute dtype for the current hardware.

    Priority: explicit `override` arg > `ARC_MODEL_DTYPE` env var > bfloat16 where
    the GPU supports it (Ampere/Ada: A100, L4, ...) > float16 (Turing/Pascal:
    T4, P100 have no bf16 — hardcoding bfloat16 there breaks or crawls).
    """
    import os  # noqa: PLC0415

    choice = override or os.environ.get("ARC_MODEL_DTYPE")
    if choice:
        return choice
    try:
        if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:  # pragma: no cover — exotic torch builds
        pass
    return "float16"


class HFModel:
    """HuggingFace causal-LM backend (Kaggle/GPU). Lazy torch import.

    `device_map="auto"` shards the model across all visible GPUs (2xT4, 4xL4, ...)
    so a 7B fits environments where a single card would OOM; on one GPU it
    behaves as before. Dtype auto-selects bf16 only where supported.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str | None = None,
        adapter_path: str | None = None,
        device_map: str = "auto",
    ):
        import torch  # noqa: PLC0415 — lazy: only present in the GPU env
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        dtype = _resolve_dtype(torch, dtype)
        _log.info("loading %s dtype=%s device_map=%s", model_path, dtype, device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            use_safetensors=True,  # refuse pickle .bin checkpoints (RCE surface)
        )
        if adapter_path is not None:
            from peft import PeftModel  # noqa: PLC0415

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        # Input tensors go to the embedding layer's device (cuda:0 under
        # device_map="auto"); accelerate hooks route activations across shards.
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        num_samples: int = 1,
        temperature: float = 0.0,
        max_time_s: float | None = None,
    ) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.0
        gen_kwargs = {}
        if max_time_s is not None and max_time_s > 0:
            # transformers stops generating once this wall-clock elapses, so a
            # stalled decode cannot overrun the per-task time budget.
            gen_kwargs["max_time"] = max_time_s
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_samples if do_sample else 1,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]
        # Greedy yields one sequence; pad up to num_samples for a uniform API.
        if not do_sample and num_samples > 1:
            completions = completions * num_samples
        return completions

    def _completion_logprobs(self, prompt: str, completion: str):
        """Log-probs of exactly the completion tokens (shift-by-one aligned)."""
        torch = self._torch
        full = prompt + completion
        enc = self.tokenizer(full, return_tensors="pt").to(self.device)
        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = self.model(**enc).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target_ids = enc["input_ids"][0, 1:]
        token_lp = log_probs[range(target_ids.shape[0]), target_ids]
        return token_lp[prompt_len - 1 :]

    def score(self, prompt: str, completion: str) -> float:
        completion_lp = self._completion_logprobs(prompt, completion)
        if completion_lp.numel() == 0:
            return float("-inf")
        return float(completion_lp.mean())

    def score_sum(self, prompt: str, completion: str) -> float:
        completion_lp = self._completion_logprobs(prompt, completion)
        if completion_lp.numel() == 0:
            return float("-inf")
        return float(completion_lp.sum())

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        max_time_s: float | None = None,
    ) -> list[str]:
        """One completion per prompt in a single left-padded generate call.

        Batching the per-augmentation decodes is a 3-5x throughput win over the
        sequential loop; greedy-only (the production default). Left padding is
        required for decoder-only generation so completions start aligned.
        """
        torch = self._torch
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                self.device
            )
        finally:
            self.tokenizer.padding_side = prev_side
        gen_kwargs = {}
        if max_time_s is not None and max_time_s > 0:
            gen_kwargs["max_time"] = max_time_s
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        padded_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[padded_len:], skip_special_tokens=True)
            for seq in out
        ]
