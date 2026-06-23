"""Build transduction prompts from a task's demonstration pairs + test input.

Format (few-shot, completion style):

    Input:
    <grid>
    Output:
    <grid>

    Input:
    <grid>
    Output:
    <grid>

    Input:
    <grid>
    Output:

The model completes the final grid. `parse_completion` extracts it. This module
is model-agnostic text assembly; tokenisation/decoding live in solvers/llm.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..io.grid import Grid
from ..io.loader import Pair
from .tokenizer import grid_to_str, str_to_grid

INPUT_TAG = "Input:"
OUTPUT_TAG = "Output:"
PAIR_SEP = "\n\n"


def _format_pair(pair: Pair, include_output: bool) -> str:
    block = f"{INPUT_TAG}\n{grid_to_str(pair.input)}\n{OUTPUT_TAG}"
    if include_output and pair.output is not None:
        block += f"\n{grid_to_str(pair.output)}"
    return block


def build_prompt(train: Sequence[Pair], test_input: Grid) -> str:
    """Assemble a few-shot prompt ending with an open Output: for the test input."""
    blocks = [_format_pair(p, include_output=True) for p in train]
    blocks.append(_format_pair(Pair(input=test_input), include_output=False))
    return PAIR_SEP.join(blocks)


def parse_completion(completion: str) -> Grid | None:
    """Extract the predicted grid from the model's completion text.

    The completion is everything generated after the trailing `Output:`; if an
    `Input:` marker appears (model ran on), we cut at it first.
    """
    text = completion
    cut = text.find(INPUT_TAG)
    if cut != -1:
        text = text[:cut]
    return str_to_grid(text)


def extract_last_input(prompt: str) -> Grid | None:
    """Return the grid of the final `Input:` block in a prompt (used by mocks /
    sanity checks). Text between the last INPUT_TAG and the following OUTPUT_TAG.
    """
    idx = prompt.rfind(INPUT_TAG)
    if idx == -1:
        return None
    tail = prompt[idx + len(INPUT_TAG) :]
    end = tail.find(OUTPUT_TAG)
    if end != -1:
        tail = tail[:end]
    return str_to_grid(tail)
