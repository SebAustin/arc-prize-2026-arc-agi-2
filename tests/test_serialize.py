"""Grid<->text and prompt assembly tests."""

from __future__ import annotations

from arc.io.loader import Pair
from arc.serialize.prompt import build_prompt, parse_completion
from arc.serialize.tokenizer import grid_to_str, str_to_grid


def test_grid_str_roundtrip(grid_factory):
    grid = grid_factory(3, 4, ncolors=10, seed=9)
    assert str_to_grid(grid_to_str(grid)) == grid


def test_str_to_grid_is_lenient():
    text = "garbage\n01\n23\ntrailing prose here"
    assert str_to_grid(text) == ((0, 1), (2, 3))


def test_str_to_grid_returns_none_on_empty():
    assert str_to_grid("no digits at all") is None


def test_build_prompt_structure():
    pair = Pair(input=((1, 2),), output=((2, 1),))
    prompt = build_prompt([pair], test_input=((3, 4),))
    assert prompt.count("Input:") == 2
    assert prompt.count("Output:") == 2
    assert prompt.rstrip().endswith("Output:")  # open for completion


def test_parse_completion_extracts_grid():
    completion = "\n12\n34\n\nInput:\n99"
    assert parse_completion(completion) == ((1, 2), (3, 4))
