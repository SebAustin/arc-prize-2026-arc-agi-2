"""Walk a downloaded NVARC synthetic-puzzles tree and emit our JSONL corpus.

Point `--src` at the extracted `nvarc_full/` directory (download/extract
sorokin/nvarc-synthetic-puzzles first) and this writes a `{prompt, completion}`
JSONL identical in shape to `synth_50k.jsonl`. Stage that JSONL as a Kaggle
Dataset and set `CORPUS_DS` in the training notebook to retrain the adapter on
the winners' curated corpus (research-ranked lever #2).

    # one-time, on a networked machine:
    kaggle datasets download sorokin/nvarc-synthetic-puzzles -p artifacts/nvarc --unzip
    python scripts/convert_nvarc_corpus.py --src artifacts/nvarc --out artifacts/nvarc_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc.synth.nvarc import nvarc_file_to_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="dir containing NVARC *.json puzzle files")
    parser.add_argument("--out", default="artifacts/nvarc_corpus.jsonl")
    parser.add_argument("--max-support", type=int, default=4)
    parser.add_argument("--max-per-file", type=int, default=6)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    files = sorted(Path(args.src).rglob("*.json"))
    if args.limit_files is not None:
        files = files[: args.limit_files]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_examples = 0
    n_ok_files = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, path in enumerate(files):
            try:
                examples = nvarc_file_to_examples(
                    path, max_support=args.max_support, max_queries=args.max_per_file, seed=i
                )
            except (json.JSONDecodeError, OSError):
                continue
            if examples:
                n_ok_files += 1
            for ex in examples:
                fh.write(json.dumps({"prompt": ex.prompt, "completion": ex.completion}) + "\n")
                n_examples += 1

    print(f"{len(files)} files scanned -> {n_ok_files} usable -> {n_examples} examples -> {out}")


if __name__ == "__main__":
    main()
