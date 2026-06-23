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

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...io.grid import Grid
from ...serialize.prompt import extract_last_input
from ...serialize.tokenizer import grid_to_str


@runtime_checkable
class LanguageModel(Protocol):
    """What the LLM solver needs from any model backend."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        num_samples: int = 1,
        temperature: float = 0.0,
    ) -> list[str]:
        """Return `num_samples` completion strings for `prompt`."""
        ...

    def score(self, prompt: str, completion: str) -> float:
        """Return the mean per-token log-probability of `completion` given
        `prompt` (higher = more likely). Used for candidate selection."""
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


class HFModel:
    """HuggingFace causal-LM backend (Kaggle/GPU). Lazy torch import."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
    ):
        import torch  # noqa: PLC0415 — lazy: only present in the GPU env
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        if adapter_path is not None:
            from peft import PeftModel  # noqa: PLC0415

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        num_samples: int = 1,
        temperature: float = 0.0,
    ) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_samples if do_sample else 1,
                pad_token_id=self.tokenizer.pad_token_id,
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

    def score(self, prompt: str, completion: str) -> float:
        torch = self._torch
        full = prompt + completion
        enc = self.tokenizer(full, return_tensors="pt").to(self.device)
        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = self.model(**enc).logits
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        target_ids = enc["input_ids"][0, 1:]
        token_lp = log_probs[range(target_ids.shape[0]), target_ids]
        completion_lp = token_lp[prompt_len - 1 :]
        if completion_lp.numel() == 0:
            return float("-inf")
        return float(completion_lp.mean())
