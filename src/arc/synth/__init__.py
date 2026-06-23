"""Synthetic ARC-like task generation for base fine-tuning (M3)."""

from .build_dataset import (
    build_synthetic_tasks,
    generate_examples,
    load_examples_jsonl,
    save_examples_jsonl,
    task_to_example,
    tasks_to_examples,
)
from .generators import GENERATORS, GeneratedTask, build_task, is_well_formed

__all__ = [
    "GENERATORS",
    "GeneratedTask",
    "build_task",
    "is_well_formed",
    "build_synthetic_tasks",
    "tasks_to_examples",
    "task_to_example",
    "generate_examples",
    "save_examples_jsonl",
    "load_examples_jsonl",
]
