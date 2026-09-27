"""Competition metric tests."""

from __future__ import annotations

from arc.eval.metrics import (
    best_cell_accuracy,
    score_output,
    score_predictions,
    shape_correct,
)
from arc.io.submission import Attempt

G1 = ((1, 2), (3, 4))
G2 = ((0, 0), (0, 0))          # same 2x2 shape as G1, 0 cells match
G3 = ((1, 2), (3, 9))          # same shape, 3/4 cells match G1
WRONG_SHAPE = ((1,),)          # 1x1, shape differs from G1


def test_score_output_match_in_attempt_1():
    assert score_output(Attempt(G1, G2), truth=G1) == 1


def test_score_output_match_in_attempt_2():
    assert score_output(Attempt(G2, G1), truth=G1) == 1


def test_score_output_no_match():
    assert score_output(Attempt(G2, G2), truth=G1) == 0


def test_score_predictions_averages_over_outputs():
    preds = {
        "a": [Attempt(G1, G2)],          # correct
        "b": [Attempt(G2, G2)],          # wrong
    }
    sols = {"a": [G1], "b": [G1]}
    summary = score_predictions(preds, sols)
    assert summary["correct"] == 1
    assert summary["total"] == 2
    assert summary["score"] == 0.5


def test_score_predictions_handles_multi_output_task():
    # First output: attempt_1 matches G1 -> correct.
    # Second output: neither attempt equals the truth G2 -> wrong.
    preds = {"m": [Attempt(G1, G2), Attempt(G1, G1)]}
    sols = {"m": [G1, G2]}
    summary = score_predictions(preds, sols)
    assert summary["correct"] == 1 and summary["total"] == 2


def test_best_cell_accuracy_partial_and_max():
    assert best_cell_accuracy(Attempt(G3, G2), truth=G1) == 0.75
    assert best_cell_accuracy(Attempt(G2, G1), truth=G1) == 1.0  # max over attempts


def test_cell_accuracy_zero_on_shape_mismatch():
    assert best_cell_accuracy(Attempt(WRONG_SHAPE, WRONG_SHAPE), truth=G1) == 0.0


def test_shape_correct_either_attempt():
    assert shape_correct(Attempt(G2, WRONG_SHAPE), truth=G1) == 1
    assert shape_correct(Attempt(WRONG_SHAPE, WRONG_SHAPE), truth=G1) == 0


def test_partial_credit_gives_gradient_below_exact_match():
    # Exact-match is 0, but the partial-credit signals are non-zero -> gradient.
    summary = score_predictions({"a": [Attempt(G3, G3)]}, {"a": [G1]})
    assert summary["score"] == 0.0
    assert summary["shape_correct_rate"] == 1.0
    assert summary["cell_accuracy"] == 0.75
