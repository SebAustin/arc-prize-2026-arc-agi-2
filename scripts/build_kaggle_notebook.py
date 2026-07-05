"""Generate a SELF-CONTAINED Kaggle notebook.

Embeds the entire `arc` package (and the kaggle_submit entrypoint) as base64
inside the notebook, so the kernel needs NO code dataset and NO Kaggle API token
— just attach the competition data and a model, set Internet=Off, Run All.

Usage:
    python scripts/build_kaggle_notebook.py
    # -> notebooks/submission_selfcontained.ipynb
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "submission_selfcontained.ipynb"


def _collect_files() -> dict[str, str]:
    """Map repo-relative path -> base64 source, for everything the kernel needs."""
    files: dict[str, str] = {}
    for path in sorted((REPO / "src").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        files[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
    entry = REPO / "scripts" / "kaggle_submit.py"
    files["scripts/kaggle_submit.py"] = base64.b64encode(entry.read_bytes()).decode("ascii")
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

CONFIG_SRC = (
    "# Point these at your attached datasets/models.\n"
    "MODEL_DS = None      # e.g. '/kaggle/input/qwen2.5-3b-instruct'  (None = DSL-only Phase A)\n"
    "ADAPTER_DS = None    # e.g. '/kaggle/input/arc-base-ft-adapter'  (optional)\n"
    "MAX_TASKS = None     # e.g. 8 = CANARY (validate GPU+model+TTT in ~20 min);\n"
    "                     # None = full scored run. Canary output is still schema-complete.\n"
)

RUN_SRC = (
    "from kaggle_submit import main\n"
    "result = main(model_path=MODEL_DS, adapter_path=ADAPTER_DS,\n"
    "              per_task_budget_s=150.0, use_ttt=True, max_tasks=MAX_TASKS)\n"
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


def build() -> Path:
    files = _collect_files()
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": _lines(INTRO)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": _lines(ENV_PREP_SRC)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": _lines(_bootstrap_source(files))},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": _lines(CONFIG_SRC)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": _lines(RUN_SRC)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    return OUT


if __name__ == "__main__":
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path}  ({size_kb:.0f} KB)")
