"""DFS/threshold decoding (Rung 6, ARChitects-style).

Instead of ONE greedy completion per prompt, explore the completion token tree
depth-first and keep every completion whose CUMULATIVE log-probability stays
above log(eps). Each surviving completion carries its cumulative log-prob —
real likelihood evidence that replaces the flat 1.0 vote weights in
`LLMSolver` and compounds with PoE selection.

Four parts:
  * `StepModel` — a minimal stateful decode-session protocol (start/step/
    backtrack), deliberately NOT routed through `LanguageModel.generate`.
  * `dfs_decode` — the search itself: explicit stack, threshold pruning,
    EOS/max-token/deadline termination, work accounting.
  * `TrieStepModel` — pure-Python mock ({prefix: {token: logprob}} table) so
    every DFS behavior is CPU-testable; its internal stack-discipline guard is
    the CPU analogue of KV-cache corruption.
  * `HFStepModel` + `greedy_selftest` — the GPU backend (KV cache cropped on
    backtrack) and an advisory on-GPU check that DFS's first leaf equals
    `model.generate` greedy (catches cache/position corruption).

No torch at module import: `HFStepModel` reuses the already-imported torch of
the `HFModel` it wraps, so this module imports clean in the CPU dev env.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Per-node branching cap. Threshold decoding needs at most ceil(1/eps) children
# per node (a child with conditional prob < eps already has cum < log(eps) when
# its parent's cum <= 0), so 16 covers every eps >= 1/16 exactly and merely
# bounds work for smaller eps.
MAX_TOP_K = 16


def top_k_for_eps(eps: float) -> int:
    """Provably-complete per-node branching for threshold `eps`."""
    return min(MAX_TOP_K, math.ceil(1.0 / eps))


@runtime_checkable
class StepModel(Protocol):
    """A stateful decode session over one prompt.

    The session is positioned at a node of the completion token tree; `start`
    prefills the prompt (depth 0), `step` descends one token, `backtrack`
    returns to a shallower ancestor. Children are (token_id, logprob) pairs
    sorted best-first, at most the implementation's top_k of them.
    """

    eos_token_id: int

    def start(self, prompt: str) -> list[tuple[int, float]]:
        """Prefill `prompt`; position at depth 0; return the root's children."""
        ...

    def step(self, token_id: int) -> list[tuple[int, float]]:
        """Descend one level via `token_id`; return the new node's children."""
        ...

    def backtrack(self, depth: int) -> None:
        """Return to the ancestor at `depth` completion tokens (0 = just after
        the prompt). Only ever called with depth < the current depth."""
        ...

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Detokenize a completion path to text."""
        ...


@dataclass(frozen=True)
class DfsCandidate:
    """One surviving completion: its text, evidence, and how it terminated."""

    text: str
    cum_logprob: float
    ended_with_eos: bool
    n_tokens: int


@dataclass
class DfsStats:
    """Work accounting for telemetry and tests."""

    expansions: int = 0  # start() + step() calls
    leaves: int = 0  # candidates emitted
    pruned: int = 0  # frames cut by the threshold
    aborted_deadline: bool = False
    aborted_budget: bool = False
    extra: dict = field(default_factory=dict)


def dfs_decode(
    model: StepModel,
    prompt: str,
    *,
    eps: float,
    max_new_tokens: int = 1024,
    max_expansions: int = 3072,
    max_candidates: int = 16,
    deadline_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[DfsCandidate], DfsStats]:
    """Depth-first threshold decode of `prompt`; see the module docstring.

    Guarantees (each covered by a CPU test):
      * The FIRST emitted leaf is the greedy completion, and it is EXEMPT from
        the eps threshold — cumulative probability decays with length, so a
        long-but-confident greedy spine would otherwise prune itself and DFS
        could return LESS than greedy. eps governs only the EXTRA candidates.
      * Every other completion with cumulative prob >= eps is found (children
        are ranked best-first; a frame is cut at the first below-threshold
        child, which bounds the tree at ~1/eps leaves).
      * On deadline/budget abort before any leaf, the current (greedy-prefix)
        path is emitted as a truncated candidate — never strictly less than a
        time-capped greedy decode.
      * `backtrack` is called once per upward jump (crops are O(#branch
        points), not O(#tokens)) and always with a strictly shallower depth.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    log_eps = math.log(eps)
    stats = DfsStats()
    candidates: list[DfsCandidate] = []
    path: list[int] = []  # token ids of the current node's completion prefix

    children = model.start(prompt)
    stats.expansions += 1
    # A frame is (remaining_children_reversed, cum_logprob_at_this_node); the
    # children list is reversed so pop() yields the best remaining child.
    stack: list[tuple[list[tuple[int, float]], float]] = [(list(reversed(children)), 0.0)]
    # After frame pops the model session sits deeper than `path`; the crop is
    # deferred until the next descend so consecutive pops coalesce into one.
    pending_backtrack = False

    while stack:
        if deadline_s is not None and clock() > deadline_s:
            stats.aborted_deadline = True
            break
        if stats.expansions >= max_expansions or stats.leaves >= max_candidates:
            stats.aborted_budget = True
            break

        frame_children, cum = stack[-1]
        if not frame_children:
            stack.pop()
            if path:
                path.pop()
            pending_backtrack = True
            continue

        tok, lp = frame_children.pop()
        child_cum = cum + lp
        # Threshold prune — but never before the first (greedy) leaf, see the
        # docstring. Children are sorted best-first, so once one child is below
        # the bar the whole rest of the frame is too.
        if stats.leaves > 0 and child_cum < log_eps:
            frame_children.clear()
            stats.pruned += 1
            continue
        if tok == model.eos_token_id:
            candidates.append(
                DfsCandidate(model.decode_tokens(path), child_cum, True, len(path))
            )
            stats.leaves += 1
            continue
        if len(path) + 1 >= max_new_tokens:
            candidates.append(
                DfsCandidate(
                    model.decode_tokens([*path, tok]), child_cum, False, len(path) + 1
                )
            )
            stats.leaves += 1
            continue
        if pending_backtrack:
            model.backtrack(len(path))
            pending_backtrack = False
        path.append(tok)
        children = model.step(tok)
        stats.expansions += 1
        stack.append((list(reversed(children)), child_cum))

    if not candidates and path:
        # Aborted mid-descent with nothing emitted: the current path IS the
        # greedy prefix (pre-first-leaf DFS only ever descends best children).
        cum = stack[-1][1] if stack else 0.0
        candidates.append(DfsCandidate(model.decode_tokens(path), cum, False, len(path)))
        stats.leaves += 1

    candidates.sort(key=lambda c: -c.cum_logprob)
    return candidates, stats


class TrieStepModel:
    """Pure-Python `StepModel` over a {prefix: {token: logprob}} table.

    Records every protocol call in `self.calls` (work accounting) and raises
    AssertionError on stack-discipline violations (stepping to a non-child,
    backtracking to a non-shallower depth) — the CPU analogue of the KV-cache
    corruption `greedy_selftest` guards against on GPU.
    """

    def __init__(
        self,
        tree: dict[tuple[int, ...], dict[int, float]],
        *,
        eos_token_id: int = 0,
        detok: dict[int, str] | None = None,
        top_k: int = MAX_TOP_K,
        clock=None,  # object with .advance(seconds); pairs with dfs_decode's clock
        step_cost_s: float = 0.0,
    ):
        self.tree = tree
        self.eos_token_id = eos_token_id
        self.detok = detok or {}
        self.top_k = top_k
        self.calls: list[tuple] = []
        self._prefix: list[int] = []
        self._clock = clock
        self._step_cost_s = step_cost_s

    def _tick(self) -> None:
        if self._clock is not None and self._step_cost_s:
            self._clock.advance(self._step_cost_s)

    def _children(self) -> list[tuple[int, float]]:
        node = self.tree.get(tuple(self._prefix), {})
        ranked = sorted(node.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: self.top_k]

    def start(self, prompt: str) -> list[tuple[int, float]]:
        self.calls.append(("start",))
        self._prefix = []
        self._tick()
        return self._children()

    def step(self, token_id: int) -> list[tuple[int, float]]:
        self.calls.append(("step", token_id))
        node = self.tree.get(tuple(self._prefix), {})
        assert token_id in node, f"step({token_id}) is not a child of {self._prefix}"
        self._prefix.append(token_id)
        self._tick()
        return self._children()

    def backtrack(self, depth: int) -> None:
        self.calls.append(("backtrack", depth))
        assert 0 <= depth < len(self._prefix), (
            f"backtrack({depth}) from depth {len(self._prefix)} is not shallower"
        )
        del self._prefix[depth:]

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return "".join(self.detok.get(t, f"<{t}>") for t in token_ids)


def _crop_cache(cache, seq_len: int):
    """Crop a KV cache to `seq_len` positions.

    Prefers `DynamicCache.crop` (transformers >= 4.38). Falls back to legacy
    tuple slicing (O(1) tensor views) if Kaggle's build lacks it. Torch-free by
    construction — only calls methods/slices on what it is given, so the
    branch logic is unit-testable with duck-typed stubs.
    """
    if hasattr(cache, "crop"):
        cache.crop(seq_len)
        return cache
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    sliced = tuple(
        (k[:, :, :seq_len, :], v[:, :, :seq_len, :]) for k, v in legacy
    )
    if hasattr(cache, "from_legacy_cache"):
        return type(cache).from_legacy_cache(sliced)
    return sliced


class HFStepModel:
    """GPU `StepModel` over an `HFModel`: ONE working KV cache (batch=1, never
    snapshotted — T4 memory), cropped in place on backtrack. Positions derive
    from the cache length, so RoPE stays correct after crops."""

    def __init__(self, hf_model, top_k: int = MAX_TOP_K):
        self._hf = hf_model
        self._torch = hf_model._torch  # reuse the lazily-imported torch
        self.top_k = top_k
        self.eos_token_id = int(hf_model.tokenizer.eos_token_id)
        self._cache = None
        self._prompt_len = 0
        self._depth = 0

    def _topk_children(self, logits) -> list[tuple[int, float]]:
        torch = self._torch
        log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        top = torch.topk(log_probs, k=min(self.top_k, log_probs.shape[-1]))
        return list(zip(top.indices.tolist(), top.values.tolist(), strict=True))

    def start(self, prompt: str) -> list[tuple[int, float]]:
        torch = self._torch
        enc = self._hf.tokenizer(prompt, return_tensors="pt").to(self._hf.device)
        with torch.no_grad():
            out = self._hf.model(**enc, use_cache=True)
        self._cache = out.past_key_values
        self._prompt_len = enc["input_ids"].shape[1]
        self._depth = 0
        return self._topk_children(out.logits)

    def step(self, token_id: int) -> list[tuple[int, float]]:
        torch = self._torch
        ids = torch.tensor([[token_id]], device=self._hf.device)
        with torch.no_grad():
            out = self._hf.model(
                input_ids=ids, past_key_values=self._cache, use_cache=True
            )
        self._cache = out.past_key_values
        self._depth += 1
        return self._topk_children(out.logits)

    def backtrack(self, depth: int) -> None:
        self._cache = _crop_cache(self._cache, self._prompt_len + depth)
        self._depth = depth

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self._hf.tokenizer.decode(list(token_ids), skip_special_tokens=True)


def greedy_selftest(
    hf_model, prompt: str, max_new_tokens: int = 256
) -> tuple[bool, str]:
    """ADVISORY on-GPU check: DFS's first leaf must equal `generate` greedy.

    The first DFS leaf is the greedy path by construction (best-first children
    + greedy-first exemption), and `max_candidates=1` stops the search right
    after it — cost is one greedy spine regardless of eps. A mismatch means
    KV-cache/position corruption. Compares PARSED GRIDS, not raw text (fp16
    argmax near-ties can differ harmlessly at whitespace). Callers print a
    loud warning on mismatch; never raise.
    """
    from ...serialize.prompt import parse_completion  # noqa: PLC0415 — avoid cycle

    greedy_text = hf_model.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=0.0
    )[0]
    cands, _stats = dfs_decode(
        hf_model.as_step_model(),
        prompt,
        eps=0.12,  # irrelevant: the first leaf is threshold-exempt
        max_new_tokens=max_new_tokens,
        max_candidates=1,
    )
    if not cands:
        return False, "dfs returned no candidates"
    greedy_grid = parse_completion(greedy_text)
    dfs_grid = parse_completion(cands[0].text)
    if greedy_grid == dfs_grid:
        return True, "dfs first leaf == generate greedy"
    return False, f"grids differ: greedy={greedy_grid!r} dfs={dfs_grid!r}"
