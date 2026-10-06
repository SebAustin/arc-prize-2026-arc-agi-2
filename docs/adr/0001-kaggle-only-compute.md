# Kaggle-only compute

The development machine is a Mac with no CUDA, so all GPU work — training, TTT,
and evaluation — runs on Kaggle kernels, and the repo is split into two
environments (`SMOKE` local for fast CPU correctness checks, `KAGGLE` for real
GPU runs) with `src/arc/config.py` as the only environment-aware module. This is
why the codebase tolerates Kaggle's quirks (mount conventions, session caps,
weekly quota, the 12h cap) as first-class concerns rather than treating them as
incidental: there is no local fallback, so a Kaggle landmine is a hard stop.
