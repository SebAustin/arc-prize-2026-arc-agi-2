# Staging offline assets for Kaggle (no-internet runtime)

The competition kernel has **no internet**, so the base model, any Python wheels
not pre-installed on Kaggle, and our code must all arrive as **Kaggle Datasets**
attached to the notebook. This guide covers the M1 baseline.

## 1. Base model → a private Kaggle Dataset

Default: **Qwen2.5-3B-Instruct** (fits one L4 with room for TTT; permissive
license). Download once on a networked machine, then upload.

```bash
# On any machine WITH internet (not the Kaggle kernel):
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen2.5-3b-instruct

# Create the dataset (Kaggle CLI):
pip install kaggle
cd qwen2.5-3b-instruct
kaggle datasets init -p .
#  -> edit dataset-metadata.json: set "title"/"id", keep it private
kaggle datasets create -p . --dir-mode zip
```

It mounts at `/kaggle/input/<your-model-dataset>/` in the notebook.

## 2. Code → a Kaggle Dataset (or notebook utility script)

Upload the repo's `src/` (and `scripts/`) as a dataset so `import arc` works
offline:

```bash
kaggle datasets create -p .   # from the repo root, packaging src/ + scripts/
```

In the notebook:

```python
import sys
sys.path.append('/kaggle/input/<your-code-dataset>/src')
sys.path.append('/kaggle/input/<your-code-dataset>/scripts')
```

## 3. Extra wheels (only if needed)

Kaggle images already include `torch`, `transformers`, `accelerate`. If a
specific version of `peft`/`unsloth`/`vllm` is required, stage wheels as a
dataset and `pip install --no-index --find-links /kaggle/input/<wheels>` offline.

## 4. Notebook body (M1 baseline)

```python
import sys
sys.path.append('/kaggle/input/<code-dataset>/src')
sys.path.append('/kaggle/input/<code-dataset>/scripts')
from kaggle_submit import main
main(model_path='/kaggle/input/<model-dataset>')   # writes /kaggle/working/submission.json
```

## 5. No-internet dry-run gate (REQUIRED before submitting)

In the notebook settings turn **Internet = Off**, run end-to-end on the public
test placeholder, and confirm: (a) the model loads from the dataset mount, (b)
`submission.json` is written with `schema_problems=0`, (c) total time is
comfortably under 12 h for 240 tasks. Only then "Submit to Competition".

## Reminders

- 1 submission/day — never submit a notebook that hasn't passed the dry run.
- L4x4 = 4×24 GB. A 3B model in bf16 needs ~6 GB; plenty of headroom for TTT
  (M2) and larger models later.
- Everything staged must be open-source-compatible (Apache/MIT/CC-BY) for prize
  eligibility — Qwen2.5 qualifies.
