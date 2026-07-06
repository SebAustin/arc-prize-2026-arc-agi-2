"""Assembles the GPU solver ensemble (DSL + heuristics + LLM/TTT) from a loaded
model. Shared by the sequential (`scripts/kaggle_submit.py`) and parallel
(`arc.parallel`) entrypoints so worker processes and the main process build an
identical ensemble from the same knobs.

No torch import at module scope: `model` is already a constructed
`LanguageModel` by the time this is called, and the `arc.solvers.llm` imports
below are function-local (lazy) exactly as in the code this was moved from.
"""

from __future__ import annotations

# Corpus size for per-task test-time training (leave-one-out x augmentation).
# Exposed here (not just in scripts/kaggle_submit.py) so worker processes and
# the sequential entrypoint share one definition.
DEFAULT_TTT_DATA_KWARGS = {"num_augs": 16, "max_examples": 250}


def build_solvers(
    model,
    use_ttt: bool,
    llm_kwargs: dict,
    ttt_config: dict | None = None,
):
    """Assemble the GPU ensemble: DSL + heuristics + (TTT or plain LLM).

    `model` is any already-constructed `LanguageModel` (real `HFModel` on
    Kaggle, or `MockModel` in tests). `ttt_config` overrides `TTTConfig`
    fields (e.g. `{"max_steps": 16}` for a faster canary/worker run).
    """
    from arc.solvers.dsl.solver import DSLSolver  # noqa: PLC0415
    from arc.solvers.identity import CHEAP_SOLVERS  # noqa: PLC0415

    if use_ttt:
        from arc.solvers.llm import LoraTTTRunner, TTTConfig, TTTSolver  # noqa: PLC0415

        ttt = TTTSolver(
            LoraTTTRunner(model, TTTConfig(**(ttt_config or {}))),
            llm_kwargs=llm_kwargs,
            ttt_data_kwargs=DEFAULT_TTT_DATA_KWARGS,
        )
        print("ensemble: DSL + heuristics + TTT(LoRA)")
        return [DSLSolver(), *CHEAP_SOLVERS, ttt]

    from arc.solvers.llm import LLMSolver  # noqa: PLC0415

    print("ensemble: DSL + heuristics + LLM(HFModel)")
    return [DSLSolver(), *CHEAP_SOLVERS, LLMSolver(model, **llm_kwargs)]
