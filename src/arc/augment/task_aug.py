"""Task-level augmentation: compose D4 symmetry + colour permutation, and
reformulate a single task into many training views (leave-one-out, shuffle).

A `TaskAug` applies one consistent transform to every grid in a task so the
task's underlying rule is preserved. At inference we augment the test input,
predict in the augmented frame, then `invert_grid` to recover the canonical
answer for voting. Symmetry and colour commute, so invert order is immaterial;
we keep a fixed convention for clarity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..io.grid import Grid
from ..io.loader import Pair, Task
from . import color, symmetry
from .color import Perm


@dataclass(frozen=True)
class TaskAug:
    """A composed, invertible task augmentation."""

    sym_name: str
    perm: Perm

    def apply_grid(self, grid: Grid) -> Grid:
        return color.apply(self.perm, symmetry.apply(self.sym_name, grid))

    def invert_grid(self, grid: Grid) -> Grid:
        return symmetry.invert(self.sym_name, color.invert(self.perm, grid))

    def apply_pair(self, pair: Pair) -> Pair:
        return Pair(
            input=self.apply_grid(pair.input),
            output=self.apply_grid(pair.output) if pair.output is not None else None,
        )

    def apply_task(self, task: Task) -> Task:
        return Task(
            task_id=task.task_id,
            train=tuple(self.apply_pair(p) for p in task.train),
            test=tuple(self.apply_pair(p) for p in task.test),
        )


IDENTITY_AUG = TaskAug(sym_name="identity", perm=color.IDENTITY_PERM)


def random_aug(seed: int, keep_zero: bool = False) -> TaskAug:
    """A random composed augmentation, deterministic in `seed`."""
    rng = random.Random(seed)
    sym_name = rng.choice(symmetry.D4_NAMES)
    perm = color.random_perm(seed=rng.randrange(1 << 30), keep_zero=keep_zero)
    return TaskAug(sym_name=sym_name, perm=perm)


def distinct_augs(n: int, seed: int = 0, keep_zero: bool = False) -> list[TaskAug]:
    """Up to `n` distinct augmentations (always includes the identity first)."""
    seen: set[tuple[str, Perm]] = set()
    out: list[TaskAug] = []
    ident_key = (IDENTITY_AUG.sym_name, IDENTITY_AUG.perm)
    seen.add(ident_key)
    out.append(IDENTITY_AUG)
    i = 0
    while len(out) < n and i < n * 50:
        aug = random_aug(seed=seed * 1_000_003 + i, keep_zero=keep_zero)
        key = (aug.sym_name, aug.perm)
        if key not in seen:
            seen.add(key)
            out.append(aug)
        i += 1
    return out


def leave_one_out(task: Task) -> list[tuple[tuple[Pair, ...], Pair]]:
    """Reformulate a task into (support_pairs, query_pair) views.

    Each demonstration pair becomes the query once, with the remaining pairs as
    support. This turns a single task's supervision into N self-supervised
    prediction problems — the core data source for test-time training.
    """
    views = []
    train = task.train
    if len(train) < 2:
        return views
    for i, query in enumerate(train):
        support = train[:i] + train[i + 1 :]
        views.append((support, query))
    return views


def shuffle_train(task: Task, seed: int) -> Task:
    """Return a copy of `task` with demonstration pairs reordered."""
    rng = random.Random(seed)
    order = list(task.train)
    rng.shuffle(order)
    return Task(task_id=task.task_id, train=tuple(order), test=task.test)
