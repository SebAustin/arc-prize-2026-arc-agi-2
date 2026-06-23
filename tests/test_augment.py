"""Augmentation invertibility — the M0 property gate.

If any apply/invert pair fails to round-trip to identity, candidate voting is
silently corrupted, so these tests are load-bearing.
"""

from __future__ import annotations

import pytest

from arc.augment import color, symmetry
from arc.augment.symmetry import D4_NAMES
from arc.augment.task_aug import (
    IDENTITY_AUG,
    distinct_augs,
    leave_one_out,
    random_aug,
)


@pytest.mark.parametrize("name", D4_NAMES)
@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (5, 2), (1, 4), (4, 1)])
def test_symmetry_roundtrips_to_identity(name, shape, grid_factory):
    grid = grid_factory(*shape, seed=hash((name, shape)) % 1000)
    assert symmetry.invert(name, symmetry.apply(name, grid)) == grid


def test_symmetry_actually_changes_grid(grid_factory):
    grid = grid_factory(2, 3, ncolors=9, seed=1)
    assert symmetry.apply("rot180", grid) != grid


@pytest.mark.parametrize("keep_zero", [True, False])
def test_color_perm_roundtrips(keep_zero, grid_factory):
    grid = grid_factory(4, 4, ncolors=10, seed=7)
    perm = color.random_perm(seed=42, keep_zero=keep_zero)
    assert color.invert(perm, color.apply(perm, grid)) == grid


def test_color_keep_zero_fixes_background():
    perm = color.random_perm(seed=5, keep_zero=True)
    assert perm[0] == 0


def test_task_aug_invert_roundtrips(grid_factory):
    grid = grid_factory(3, 5, ncolors=8, seed=2)
    aug = random_aug(seed=11)
    assert aug.invert_grid(aug.apply_grid(grid)) == grid


def test_distinct_augs_unique_and_includes_identity():
    augs = distinct_augs(8, seed=1)
    assert augs[0] == IDENTITY_AUG
    keys = {(a.sym_name, a.perm) for a in augs}
    assert len(keys) == len(augs)  # all distinct


def test_leave_one_out_counts(task_factory):
    g = ((1, 2), (3, 4))
    task = task_factory([(g, g), (g, g), (g, g)], [g])
    views = leave_one_out(task)
    assert len(views) == 3
    for support, _query in views:
        assert len(support) == 2
