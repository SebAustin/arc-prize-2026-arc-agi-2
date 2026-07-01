"""Competition metric tests."""

from __future__ import annotations

from arc.eval.metrics import score_output, score_predictions
from arc.io.submission import Attempt

G1 = ((1, 2), (3, 4))
G2 = ((0, 0), (0, 0))


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
