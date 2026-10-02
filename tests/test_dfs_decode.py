"""DFS/threshold decoding core (src/arc/solvers/llm/dfs_decode.py), pure CPU.

Every behavior is exercised against `TrieStepModel` (a {prefix: {token:
logprob}} table) whose internal stack-discipline guard is the CPU analogue of
KV-cache corruption. Tokens: 0 is EOS throughout; detok maps 1->"a", 2->"b", ...
"""

from __future__ import annotations

import math

import pytest

from arc.solvers.llm.dfs_decode import (
    MAX_TOP_K,
    TrieStepModel,
    _crop_cache,
    dfs_decode,
    top_k_for_eps,
)

_DETOK = {1: "a", 2: "b", 3: "c", 4: "d"}


def _lp(p: float) -> float:
    return math.log(p)


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ---- threshold search ---------------------------------------------------------


def _three_leaf_trie() -> dict:
    """Root splits {0.5, 0.3, 0.15, 0.049}; every branch ends in EOS."""
    return {
        (): {1: _lp(0.5), 2: _lp(0.3), 3: _lp(0.15), 4: _lp(0.049)},
        (1,): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
        (3,): {0: _lp(1.0)},
        (4,): {0: _lp(1.0)},
    }


def test_finds_all_leaves_above_threshold():
    model = TrieStepModel(_three_leaf_trie(), detok=_DETOK)
    cands, stats = dfs_decode(model, "p", eps=0.1)

    assert [c.text for c in cands] == ["a", "b", "c"]  # sorted by cum desc
    assert [c.cum_logprob for c in cands] == pytest.approx(
        [_lp(0.5), _lp(0.3), _lp(0.15)]
    )
    assert all(c.ended_with_eos and c.n_tokens == 1 for c in cands)
    assert stats.leaves == 3


def test_prunes_below_threshold_and_skips_work():
    model = TrieStepModel(_three_leaf_trie(), detok=_DETOK)
    cands, stats = dfs_decode(model, "p", eps=0.1)

    assert "d" not in [c.text for c in cands]
    assert stats.pruned >= 1
    # Work accounting: the pruned subtree was never entered.
    assert ("step", 4) not in model.calls


def test_greedy_path_emitted_first():
    # Greedy is locally-best per node but globally NOT the highest-cum leaf
    # (0.6*0.55=0.33 vs 0.4). With max_candidates=1 the single returned leaf
    # must still be the greedy path — the invariant greedy_selftest relies on.
    tree = {
        (): {1: _lp(0.6), 2: _lp(0.4)},
        (1,): {3: _lp(0.55), 0: _lp(0.45)},
        (1, 3): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
    }
    model = TrieStepModel(tree, detok=_DETOK)
    cands, _ = dfs_decode(model, "p", eps=0.01, max_candidates=1)

    assert len(cands) == 1
    assert cands[0].text == "ac"  # argmax at every node
    assert cands[0].ended_with_eos


def test_greedy_leaf_exempt_from_threshold():
    # Cumulative prob decays with length: the greedy spine here has cum 0.125,
    # far below eps=0.3. It must STILL be returned in full — DFS can never do
    # worse than greedy; eps governs only the extra candidates.
    tree = {
        (): {1: _lp(0.5)},
        (1,): {2: _lp(0.5)},
        (1, 2): {0: _lp(0.5)},
    }
    model = TrieStepModel(tree, detok=_DETOK)
    cands, stats = dfs_decode(model, "p", eps=0.3)

    assert [c.text for c in cands] == ["ab"]
    assert cands[0].ended_with_eos
    assert cands[0].cum_logprob == pytest.approx(_lp(0.125))
    assert stats.pruned == 0


def test_eos_terminates_branch_and_counts_logprob():
    tree = {
        (): {1: _lp(0.9), 0: _lp(0.1)},  # EOS directly at the root too
        (1,): {0: _lp(1.0)},
    }
    model = TrieStepModel(tree, detok=_DETOK)
    cands, _ = dfs_decode(model, "p", eps=0.05)

    by_text = {c.text: c for c in cands}
    assert by_text["a"].cum_logprob == pytest.approx(_lp(0.9))  # includes EOS lp
    assert by_text[""].n_tokens == 0  # empty completion via root EOS
    assert ("step", 0) not in model.calls  # never descends INTO EOS


def test_max_new_tokens_truncates():
    chain = {
        (): {1: _lp(1.0)},
        (1,): {2: _lp(1.0)},
        (1, 2): {3: _lp(1.0)},
        (1, 2, 3): {4: _lp(1.0)},  # never reached
    }
    model = TrieStepModel(chain, detok=_DETOK)
    cands, _ = dfs_decode(model, "p", eps=0.5, max_new_tokens=3)

    assert len(cands) == 1
    assert cands[0].text == "abc"
    assert cands[0].n_tokens == 3
    assert not cands[0].ended_with_eos


def test_deadline_abort_returns_greedy_partial():
    chain = {(tuple(range(1, k + 1))): {k + 1: _lp(1.0)} for k in range(0, 9)}
    clock = _FakeClock()
    model = TrieStepModel(
        chain, detok={i: chr(96 + i) for i in range(1, 11)},
        clock=clock, step_cost_s=1.0,
    )
    cands, stats = dfs_decode(
        model, "p", eps=0.5, deadline_s=2.5, clock=clock.now
    )

    assert stats.aborted_deadline
    assert len(cands) == 1  # exactly the truncated greedy prefix
    assert cands[0].text == "ab"  # two steps fit inside the deadline
    assert not cands[0].ended_with_eos


def test_max_expansions_abort():
    model = TrieStepModel(_three_leaf_trie(), detok=_DETOK)
    cands, stats = dfs_decode(model, "p", eps=0.1, max_expansions=2)

    assert stats.aborted_budget
    assert stats.expansions <= 2
    assert len(cands) == 1  # greedy-prefix fallback


def test_max_candidates_cap():
    model = TrieStepModel(_three_leaf_trie(), detok=_DETOK)
    cands, stats = dfs_decode(model, "p", eps=0.01, max_candidates=2)

    assert len(cands) == 2
    assert {c.text for c in cands} == {"a", "b"}  # best-first exploration
    assert stats.aborted_budget


def test_backtrack_stack_discipline():
    tree = {
        (): {1: _lp(0.5), 2: _lp(0.4)},
        (1,): {3: _lp(0.6), 0: _lp(0.4)},
        (1, 3): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
    }
    model = TrieStepModel(tree, detok=_DETOK)
    cands, _ = dfs_decode(model, "p", eps=0.01)

    # The trie's internal guard did not fire, and pops COALESCED: one
    # backtrack(0) covers the two-frame jump from (1,3) back to the root.
    assert model.calls == [
        ("start",),
        ("step", 1),
        ("step", 3),
        ("backtrack", 0),
        ("step", 2),
    ]
    assert [c.text for c in cands] == ["b", "ac", "a"]  # sorted by cum


def test_eps_validation():
    model = TrieStepModel(_three_leaf_trie(), detok=_DETOK)
    with pytest.raises(ValueError, match="eps"):
        dfs_decode(model, "p", eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        dfs_decode(model, "p", eps=1.5)


def test_top_k_for_eps():
    assert top_k_for_eps(0.5) == 2
    assert top_k_for_eps(0.12) == 9
    assert top_k_for_eps(0.001) == MAX_TOP_K  # capped


# ---- _crop_cache (duck-typed stubs; no torch) ---------------------------------


class _CropCache:
    def __init__(self):
        self.cropped_to = None

    def crop(self, seq_len):
        self.cropped_to = seq_len


class _FakeTensor:
    def __init__(self):
        self.key = None

    def __getitem__(self, key):
        self.key = key
        return ("sliced", key)


def test_crop_cache_prefers_crop():
    cache = _CropCache()
    out = _crop_cache(cache, 7)
    assert out is cache
    assert cache.cropped_to == 7


def test_crop_cache_legacy_fallback():
    k, v = _FakeTensor(), _FakeTensor()
    out = _crop_cache(((k, v),), 7)  # plain legacy tuple: no crop() available

    want = (slice(None), slice(None), slice(None, 7), slice(None))
    assert k.key == want and v.key == want
    assert out == ((("sliced", want), ("sliced", want)),)
