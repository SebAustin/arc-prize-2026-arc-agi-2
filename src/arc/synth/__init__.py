"""Synthetic ARC-like task generation for base fine-tuning (M3)."""

from .build_dataset import (
    build_synthetic_tasks,
    generate_examples,
    is_dsl_solvable,
    load_examples_jsonl,
    save_examples_jsonl,
    task_to_example,
    tasks_to_examples,
)
from .generators import GENERATORS, HARD_GENERATORS, GeneratedTask, build_task, is_well_formed
from .nvarc import load_nvarc_pairs, nvarc_examples, nvarc_file_to_examples

__all__ = [
    "GENERATORS",
    "HARD_GENERATORS",
    "GeneratedTask",
    "build_task",
    "is_well_formed",
    "build_synthetic_tasks",
    "is_dsl_solvable",
    "tasks_to_examples",
    "task_to_example",
    "generate_examples",
    "save_examples_jsonl",
    "load_examples_jsonl",
    "load_nvarc_pairs",
    "nvarc_examples",
    "nvarc_file_to_examples",
]
