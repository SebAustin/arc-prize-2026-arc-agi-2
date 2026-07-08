"""Generate SELF-CONTAINED Kaggle notebooks (submission + training).

Embeds the entire `arc` package (and the relevant entrypoint script) as base64
inside each notebook, so the kernel needs NO code dataset and NO Kaggle API
token — just attach the competition data / corpus / model, set Internet=Off,
Run All.

Usage:
    python scripts/build_kaggle_notebook.py
    # -> notebooks/submission_selfcontained.ipynb  (inference + TTT)
    # -> notebooks/train_adapter.ipynb             (Rung 4 base fine-tune)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION_OUT = REPO / "notebooks" / "submission_selfcontained.ipynb"
TRAIN_OUT = REPO / "notebooks" / "train_adapter.ipynb"


def _collect_files(entry_scripts: tuple[str, ...]) -> dict[str, str]:
    """Map repo-relative path -> base64 source: the `arc` package plus whichever
    top-level scripts the target notebook needs (kept per-notebook so the
    submission kernel doesn't carry training code and vice versa)."""
    files: dict[str, str] = {}
    for path in sorted((REPO / "src").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        files[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
    for name in entry_scripts:
        entry = REPO / "scripts" / name
        files[f"scripts/{name}"] = base64.b64encode(entry.read_bytes()).decode("ascii")
    return files


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _bootstrap_source(files: dict[str, str]) -> str:
    payload = json.dumps(files, indent=0)
    return (
        "# Self-contained bootstrap: write the embedded `arc` package to disk.\n"
        "import base64, json, os, sys\n"
        f"FILES = json.loads(r'''{payload}''')\n"
        "ROOT = '/kaggle/working/arc_code'\n"
        "for rel, b64 in FILES.items():\n"
        "    dst = os.path.join(ROOT, rel)\n"
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "    with open(dst, 'wb') as f:\n"
        "        f.write(base64.b64decode(b64))\n"
        "sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
        "sys.path.insert(0, os.path.join(ROOT, 'scripts'))\n"
        "print('bootstrapped', len(FILES), 'files ->', ROOT)\n"
    )


ENV_PREP_SRC = (
    "# Kaggle env prep (offline-safe). Some Kaggle images ship a `torchao` too old\n"
    "# for the installed `peft`, which makes LoRA (TTT) adapter creation raise\n"
    "# `ImportError: incompatible version of torchao` on EVERY task -> TTT silently\n"
    "# degrades to the DSL-only ensemble. We don't use torchao, so drop the\n"
    "# incompatible version and let peft fall back to the standard LoRA path.\n"
    "import sys, subprocess\n"
    "# GPU visibility check FIRST: if this prints nothing/fails, the accelerator is\n"
    "# OFF -> fix Settings > Accelerator before wasting a run (the model can't load).\n"
    "subprocess.run(['nvidia-smi', '-L'], check=False)\n"
    "try:\n"
    "    import torchao\n"
    "    from packaging.version import parse as _p\n"
    "    if _p(getattr(torchao, '__version__', '0')) < _p('0.16.0'):\n"
    "        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],\n"
    "                       check=False)\n"
    "        for _m in [k for k in list(sys.modules) if k.startswith('torchao')]:\n"
    "            del sys.modules[_m]\n"
    "        print('removed incompatible torchao (<0.16) so PEFT/LoRA can run')\n"
    "    else:\n"
    "        print('torchao', torchao.__version__, 'is compatible')\n"
    "except ImportError:\n"
    "    print('torchao not installed -> nothing to do')\n"
)

# PRODUCTION defaults. A first submission scored 0.00 because a UI re-save
# reverted the config cell to a then-default MODEL_DS=None (DSL-only, no model).
# The safe default is therefore the REAL run config; opt DOWN to Phase-A
# debugging by setting MODEL_DS = None explicitly.
CONFIG_SRC = (
    "# Production run config (attach the model under Add Input -> Models).\n"
    "# Set MODEL_DS = None ONLY for a DSL-only plumbing check (scores ~0).\n"
    "MODEL_DS = '/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b/1'\n"
    "ADAPTER_DS = None    # e.g. '/kaggle/input/arc-base-ft-adapter'  (optional)\n"
    "MAX_TASKS = None     # e.g. 8 = quick canary; None = full run. Commit runs\n"
    "                     # auto-canary via the placeholder-hash fast path anyway.\n"
)

RUN_SRC = (
    "import logging\n"
    "logging.basicConfig(level=logging.INFO, force=True)  # surface telemetry\n"
    "import torch\n"
    "n_gpu = torch.cuda.device_count()\n"
    "if n_gpu >= 4:\n"
    "    # L4x4: one full model replica per GPU, 4-way task parallelism.\n"
    "    workers, budget, ttt_cfg = 4, 550.0, {'max_steps': 96}\n"
    "else:\n"
    "    # T4x2 or fewer: sequential + memory-safe TTT (14.5GB cards).\n"
    "    ttt_cfg = {'max_steps': 16, 'batch_size': 1, 'max_seq_len': 1536}\n"
    "    workers, budget = 0, 150.0\n"
    "print(f'{n_gpu} GPU(s) -> num_workers={workers}, budget={budget}s, ttt={ttt_cfg}')\n"
    "from kaggle_submit import main\n"
    "result = main(model_path=MODEL_DS, adapter_path=ADAPTER_DS,\n"
    "              per_task_budget_s=budget, use_ttt=True, max_tasks=MAX_TASKS,\n"
    "              ttt_config=ttt_cfg, num_workers=workers)\n"
    "print(result)\n"
    "assert result['problems'] == [], result['problems']\n"
)

INTRO = (
    "# ARC-AGI-2 — self-contained submission notebook\n\n"
    "No code dataset or API token needed: the entire solver is embedded below.\n\n"
    "**Steps:** attach the competition data (auto on a competition notebook) + a model "
    "dataset, set `MODEL_DS`, **Internet = Off**, then **Run All**. Phase A: leave "
    "`MODEL_DS = None` to validate the plumbing with the DSL ensemble (no model)."
)

# ---- Training notebook (Rung 4: base fine-tune adapter) --------------------

TRAIN_CONFIG_SRC = (
    "# Point these at your attached model / corpus datasets.\n"
    "MODEL_DS = '/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b/1'\n"
    "CORPUS_DS = '/kaggle/input/arc-synth-corpus/synth_50k.jsonl'\n"
    "MAX_EXAMPLES = None  # e.g. 2000 for a fast canary; None = full 50k corpus\n"
    "EPOCHS = 1\n"
)

TRAIN_RUN_SRC = (
    "from kaggle_train import main\n"
    "adapter_dir = main(\n"
    "    base_model_path=MODEL_DS,\n"
    "    corpus_path=CORPUS_DS,\n"
    "    max_examples=MAX_EXAMPLES,\n"
    "    epochs=EPOCHS,\n"
    "    resume=True,\n"
    ")\n"
    "print('adapter_dir:', adapter_dir)\n"
)

TRAIN_INTRO = (
    "# ARC-AGI-2 — base fine-tune adapter training notebook (Rung 4)\n\n"
    "Trains a LoRA adapter over the synthetic (prompt, completion) corpus so the "
    "model learns the ARC I/O format and a broad transformation library BEFORE "
    "per-task TTT. No code dataset or API token needed: the solver + training "
    "entrypoint are embedded below; the corpus itself is NOT embedded (it's a "
    "staged Kaggle Dataset — see `CORPUS_DS`).\n\n"
    "**Expected duration:** roughly 3.5-6h for the full 50k-example corpus on one "
    "L4 (one epoch); scales down proportionally with `MAX_EXAMPLES` for a canary, "
    "and roughly with GPU count if `device_map` shards across more than one.\n\n"
    "**Resume-on-rerun:** the trainer checkpoints every "
    "`checkpoint_every_steps` optimizer steps (default 200) plus once per epoch, "
    "under `output_dir/checkpoint` with a `state.json` sidecar. If the kernel "
    "gets interrupted (12h cap, restart, crash), just **Run All again** with the "
    "same `output_dir` (default `/kaggle/working/adapter`) — training resumes "
    "from the last checkpoint instead of starting over.\n\n"
    "**Steps:** attach the base model dataset + the synthetic corpus dataset, "
    "set `MODEL_DS`/`CORPUS_DS` below, **Internet = Off**, then **Run All**. "
    "When done: create a Kaggle Dataset from the output adapter directory and "
    "point `ADAPTER_DS` at it in the submission notebook."
)


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def build_submission() -> Path:
    files = _collect_files(("kaggle_submit.py", "kaggle_eval.py"))
    nb = _notebook(
        [
            _markdown_cell(INTRO),
            _code_cell(ENV_PREP_SRC),
            _code_cell(_bootstrap_source(files)),
            _code_cell(CONFIG_SRC),
            _code_cell(RUN_SRC),
        ]
    )
    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION_OUT.write_text(json.dumps(nb, indent=1))
    return SUBMISSION_OUT


def build_train_adapter() -> Path:
    files = _collect_files(("kaggle_train.py",))
    nb = _notebook(
        [
            _markdown_cell(TRAIN_INTRO),
            _code_cell(ENV_PREP_SRC),
            _code_cell(_bootstrap_source(files)),
            _code_cell(TRAIN_CONFIG_SRC),
            _code_cell(TRAIN_RUN_SRC),
        ]
    )
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.write_text(json.dumps(nb, indent=1))
    return TRAIN_OUT


# Backward-compat alias: the original single-notebook entrypoint name.
build = build_submission


def build_all() -> list[Path]:
    return [build_submission(), build_train_adapter()]


if __name__ == "__main__":
    for out_path in build_all():
        size_kb = out_path.stat().st_size / 1024
        print(f"wrote {out_path}  ({size_kb:.0f} KB)")
