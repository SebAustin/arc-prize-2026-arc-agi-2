"""Verify every Kaggle mount-path constant resolves against the LIVE Kaggle API.

`autopilot_config.check_mount_conventions()` guards the STRUCTURE of each mount
offline (owner-stripping, lowercase framework); this script confirms the
referenced dataset/model actually EXISTS and is accessible — the definitive check
before relying on a new mount. Needs network + Kaggle auth (~/.kaggle/kaggle.json)
so it is NOT part of the offline test suite; run it by hand after changing any
`/kaggle/input/...` constant:

    python scripts/verify_kaggle_mounts.py     # exit 0 = all resolve
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from autopilot_config import _mount_specs


def _kaggle_bin() -> str:
    """Prefer the kaggle CLI next to the running interpreter (the project venv)."""
    candidate = Path(sys.executable).parent / "kaggle"
    return str(candidate) if candidate.exists() else "kaggle"


def _resolves(ref: str, kind: str) -> tuple[bool, str]:
    """Does the model/dataset `ref` resolve on Kaggle? Returns (ok, detail)."""
    if kind == "model":
        cmd = [_kaggle_bin(), "models", "instances", "versions", "files", ref]
    else:
        cmd = [_kaggle_bin(), "datasets", "files", ref]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stderr or proc.stdout).strip()
    ok = proc.returncode == 0 and "404" not in combined and "not found" not in combined.lower()
    detail = "" if ok else (combined.splitlines()[-1] if combined else "no output")
    return ok, detail


def main() -> int:
    specs = _mount_specs()
    failures = 0
    for name, _mount, ref, kind in specs:
        ok, detail = _resolves(ref, kind)
        print(f"[{'OK ' if ok else 'FAIL'}] {name:18s} {kind:7s} {ref}")
        if not ok:
            failures += 1
            print(f"         -> {detail}")
    print(f"\n{len(specs) - failures}/{len(specs)} mounts resolve on Kaggle.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
