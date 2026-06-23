"""Bounded-depth composition search over DSL primitives.

Strategy: build a small base vocabulary (parameter-free geometric ops plus any
parameters inferred from the train pairs — scale, tile, colour map), enumerate
single ops and depth-2 compositions, keep only programs that reproduce ALL train
outputs, and rank survivors by simplicity. Tiny by design so it runs in
milliseconds and never blows the per-task budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ...io.grid import Grid
from ...io.loader import Task
from ..base import Program, verify_program
from . import primitives as P

LabeledProgram = tuple[int, str, Program]  # (complexity, label, program)


def _compose(f: Program, g: Program) -> Program:
    """Return the program x -> g(f(x))."""
    return lambda x: g(f(x))


def _base_vocabulary(task: Task) -> list[tuple[str, Program]]:
    """Parameter-free ops plus task-inferred parameter ops."""
    vocab: list[tuple[str, Program]] = list(P.PARAM_FREE.items())

    ratio = P._shape_ratio(task)
    if ratio is not None:
        fy, fx = ratio
        vocab.append((f"scale{fy}x{fx}", P.scale_program(fy, fx)))
        vocab.append((f"tile{fy}x{fx}", P.tile_program(fy, fx)))

    cmap = P.learn_colormap(task)
    if cmap is not None:
        vocab.append(("colormap", P.colormap_program(cmap)))

    return vocab


def generate_programs(task: Task) -> list[tuple[str, Program]]:
    """Single ops and depth-2 compositions over the base vocabulary."""
    vocab = _base_vocabulary(task)
    programs: list[tuple[str, Program]] = list(vocab)  # depth 1
    for n1, f1 in vocab:
        if n1 == "identity":
            continue
        for n2, f2 in vocab:
            if n2 == "identity":
                continue
            programs.append((f"{n1}|{n2}", _compose(f1, f2)))  # depth 2
    return programs


def search(task: Task, budget_s: float = 5.0) -> list[tuple[str, Program]]:
    """Return verified programs, simplest (fewest ops) first, deduped by label."""
    deadline = time.monotonic() + budget_s
    verified: list[tuple[int, str, Program]] = []
    seen_labels: set[str] = set()
    for label, program in generate_programs(task):
        if time.monotonic() > deadline:
            break
        if label in seen_labels:
            continue
        if verify_program(program, task):
            complexity = label.count("|") + 1
            verified.append((complexity, label, program))
            seen_labels.add(label)
    verified.sort(key=lambda t: (t[0], t[1]))
    return [(label, program) for _, label, program in verified]
