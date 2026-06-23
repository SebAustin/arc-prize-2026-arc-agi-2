"""LLM transduction + test-time-training solvers (M1+)."""

from .model import HFModel, LanguageModel, MockModel
from .solver import LLMSolver
from .ttt import (
    LoraTTTRunner,
    MockTTTRunner,
    TTTConfig,
    TTTRunner,
    TTTSolver,
)
from .ttt_data import TrainExample, build_ttt_examples

__all__ = [
    "HFModel",
    "LanguageModel",
    "MockModel",
    "LLMSolver",
    "TTTSolver",
    "TTTRunner",
    "MockTTTRunner",
    "LoraTTTRunner",
    "TTTConfig",
    "TrainExample",
    "build_ttt_examples",
]
