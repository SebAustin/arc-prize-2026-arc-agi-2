"""Kaggle offline base fine-tuning entrypoint (Rung 4).

Runs on the L4x4/T4x2 GPU with NO internet: loads the synthetic corpus (staged
as a Kaggle Dataset), LoRA fine-tunes the base model over it, and writes the
resulting adapter to `output_dir`. That adapter is then staged as its OWN
Kaggle Dataset (`ADAPTER_DS`) and loaded by `kaggle_submit.main(adapter_path=...)`
at inference time, with per-task TTT adapting further on top of it.

Kept as a plain script — importable and lint-clean off-GPU; torch/peft/
transformers only load inside `finetune()`, so this module never needs them
just to be imported (e.g. by the notebook builder, or by a test that checks
"does this script blow up without a GPU env").

Usage (inside the Kaggle notebook, after the bootstrap cell):
    from kaggle_train import main
    adapter_dir = main(
        base_model_path=MODEL_DS,
        corpus_path=CORPUS_DS,
        max_examples=MAX_EXAMPLES,
        epochs=EPOCHS,
    )
"""

from __future__ import annotations

import argparse
import glob
import os

_DEFAULT_LOCAL_OUTPUT = "artifacts/adapter"
_DEFAULT_KAGGLE_OUTPUT = "/kaggle/working/adapter"


def _locate_corpus(corpus_path: str) -> str:
    """Resolve the corpus file even when the Dataset mounts under an
    unexpected folder name (dataset slugs and kernel attachments don't always
    line up — the exact failure that cost the first training canary). If the
    given path is missing, search /kaggle/input for the same basename; on
    failure, raise with a listing of what IS mounted so the fix is obvious."""
    if os.path.exists(corpus_path):
        return corpus_path
    basename = os.path.basename(corpus_path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{basename}", recursive=True))
    if matches:
        print(f"corpus not at {corpus_path}; found {matches[0]}")
        return matches[0]
    inp = "/kaggle/input"
    mounted = sorted(os.listdir(inp)) if os.path.isdir(inp) else []
    listing = "\n".join(f"    - {inp}/{m}" for m in mounted) or "    (nothing mounted)"
    raise FileNotFoundError(
        f"Corpus not found: {corpus_path} (and no {basename} anywhere under "
        f"/kaggle/input).\nAttach the corpus Dataset to this notebook "
        f"(Add Input -> Datasets -> arc-synth-corpus).\nCurrently mounted:\n{listing}"
    )


def _default_output_dir() -> str:
    """Kaggle working dir if we're on Kaggle, else the local artifacts dir —
    mirrors `arc.config._detect_output_dir`'s Kaggle-vs-local split without
    importing `arc.config` (this script must stay import-safe standalone)."""
    if os.path.isdir("/kaggle/working"):
        return _DEFAULT_KAGGLE_OUTPUT
    return _DEFAULT_LOCAL_OUTPUT


def _corpus_stats(examples) -> dict:
    """Cheap corpus sanity numbers printed before burning GPU hours on it."""
    n = len(examples)
    if n == 0:
        return {"count": 0, "mean_prompt_chars": 0.0, "mean_completion_chars": 0.0}
    total_prompt = sum(len(ex.prompt) for ex in examples)
    total_completion = sum(len(ex.completion) for ex in examples)
    return {
        "count": n,
        "mean_prompt_chars": round(total_prompt / n, 1),
        "mean_completion_chars": round(total_completion / n, 1),
    }


def _load_corpus_examples(corpus_path: str, max_files: int | None = None) -> list:
    """Load training examples from either a JSONL corpus (our synthetic corpus)
    or a DIRECTORY of NVARC puzzle files (the 2025 winners' corpus, converted
    IN-KERNEL — one leave-one-out example per file for maximal template
    diversity). `max_files` bounds the directory walk so a huge corpus doesn't
    blow the in-kernel load budget (finetune subsets further via max_examples)."""
    from arc.synth import load_examples_jsonl, nvarc_file_to_examples  # noqa: PLC0415

    if os.path.isdir(corpus_path):
        files = sorted(glob.glob(os.path.join(corpus_path, "**", "*.json"), recursive=True))
        if max_files is not None:
            files = files[:max_files]
        examples: list = []
        for i, path in enumerate(files):
            examples.extend(nvarc_file_to_examples(path, max_support=4, max_queries=1, seed=i))
        print(f"loaded {len(examples)} examples from {len(files)} NVARC files under {corpus_path}")
        return examples
    return load_examples_jsonl(corpus_path)


def main(
    base_model_path: str | None = None,
    corpus_path: str | None = None,
    output_dir: str | None = None,
    max_examples: int | None = None,
    epochs: int = 1,
    resume: bool = True,
    config_overrides: dict | None = None,
    max_train_seconds: float | None = 39600.0,  # 11 h: stop+save before Kaggle's 12 h kill
) -> str:
    """Fine-tune a LoRA adapter on the synthetic corpus; return the output dir.

    Resolves `base_model_path`/`corpus_path` from `ARC_MODEL_PATH`/
    `ARC_CORPUS_PATH` when not passed explicitly, and `output_dir` from
    `/kaggle/working/adapter` on Kaggle else `artifacts/adapter` locally.
    `config_overrides` sets arbitrary `TrainConfig` fields (e.g.
    `{"checkpoint_every_steps": 100}`) without editing this script.
    """
    from arc.train.finetune import TrainConfig, finetune  # noqa: PLC0415

    base_model_path = base_model_path or os.environ.get("ARC_MODEL_PATH")
    corpus_path = corpus_path or os.environ.get("ARC_CORPUS_PATH")
    output_dir = output_dir or _default_output_dir()
    if not base_model_path:
        raise ValueError(
            "base_model_path is required (pass explicitly or set ARC_MODEL_PATH)"
        )
    if not corpus_path:
        raise ValueError(
            "corpus_path is required (pass explicitly or set ARC_CORPUS_PATH)"
        )

    corpus_path = _locate_corpus(corpus_path)
    print(f"base_model_path={base_model_path}")
    print(f"corpus_path={corpus_path}")
    print(f"output_dir={output_dir}  max_examples={max_examples}  epochs={epochs}  resume={resume}")

    examples = _load_corpus_examples(corpus_path, max_files=max_examples)
    stats = _corpus_stats(examples)
    print(f"corpus stats: {stats}")

    overrides = dict(config_overrides or {})
    overrides["epochs"] = epochs
    overrides["resume"] = resume
    cfg = TrainConfig(**overrides)

    adapter_dir = finetune(
        base_model_path,
        examples,
        output_dir,
        config=cfg,
        max_examples=max_examples,
        max_train_seconds=max_train_seconds,
    )
    print(f"adapter saved -> {adapter_dir}")
    print(
        "next step: create a Kaggle Dataset from this output directory, then "
        "point ADAPTER_DS at it in the submission notebook (kaggle_submit.main"
        "(adapter_path=ADAPTER_DS)) so inference loads the base-fine-tuned "
        "weights before per-task TTT adapts further on top."
    )
    return adapter_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--corpus-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint/state.json and train from scratch.",
    )
    args = parser.parse_args()
    main(
        base_model_path=args.base_model_path,
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        epochs=args.epochs,
        resume=not args.no_resume,
    )
