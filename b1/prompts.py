"""Prompt construction.

The prompt is deliberately minimal: task description (from the batch's
task.md), input schema, class list, a compact view of the training set
(per-class mean spectrum in 64 bins), output contract, and constraints.
No modelling advice is given; method choice is part of what B1 measures.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TASK = """Multi-class classification of spectra.
Each sample is a measured spectrum; each class is a physical or chemical category."""


def _task_text(batch: Path) -> str:
    path = batch / "task.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return DEFAULT_TASK


def _minmax(values):
    lo, hi = float(min(values)), float(max(values))
    return [(v - lo) / (hi - lo + 1e-12) for v in values]


def _profiles_json(ref, waves, bins):
    X = np.array([[r["spectrum"][w] for w in waves] for r in ref], float)
    y = np.array([r["label"] for r in ref])
    idx = np.linspace(0, len(waves) - 1, bins).astype(int)
    lines = []
    for lab in sorted(set(y)):
        prof = _minmax(X[y == lab].mean(0)[idx])
        lines.append(f"{lab}: " + " ".join(f"{v:.3f}" for v in prof))
    return "\n".join(lines)


def _profiles_csv(train: pd.DataFrame, features, bins):
    idx = np.linspace(0, len(features) - 1, bins).astype(int)
    lines = []
    for lab, group in train.groupby("label", sort=True):
        prof = _minmax(group[features].to_numpy(float).mean(0)[idx])
        lines.append(f"{lab}: " + " ".join(f"{v:.3f}" for v in prof))
    return "\n".join(lines)


def _shell(task, meta, preview, input_block, n_features, n_train, n_test, labels):
    return f"""You are writing ONE standalone Python program for a supervised classification task.

TASK
{task}

INPUT (a directory path is passed as argv[1])
{input_block}
- Labels ({len(labels)}): {labels}

TRAINING-SET PREVIEW
Per-class mean spectrum, averaged into {meta['bins']} bins across the feature axis and
min-max scaled per class (so you can see the shape of each class):
{preview}

OUTPUT
Write predictions.json in the CURRENT WORKING DIRECTORY:
{{"predictions": [{{"sample_id": "...", "prediction": "..."}}, ...]}}
Exactly {n_test} rows; prediction must be one of the labels above.

CONSTRAINTS
1. Python 3.10 + numpy/pandas. No network, no subprocess, no reading any file other
   than the input files above. Test labels are not available.
2. Train only on the training data.
3. Deterministic, single process, under {meta['timeout']} seconds.
4. Output ONLY Python code. No Markdown fences, no commentary.

Your program will be executed exactly once and its predictions scored.
"""


def build_prompt(batch: Path, meta: dict, bins: int = 64, timeout: int = 300) -> str:
    task = _task_text(batch)
    labels = meta["labels"]
    n_feat = meta["n_features"]
    meta = dict(meta, bins=bins, timeout=timeout)

    if meta["format"] == "json":
        waves = meta["waves"]
        preview = _profiles_json(meta["ref"], waves, bins)
        input_block = (
            f"- reference_samples.json: {meta['n_train']} labeled samples, each\n"
            f"  {{\"sample_id\": str, \"spectrum\": {{\"<wavenumber>\": float, ...}}, \"label\": str}}\n"
            f"  with {n_feat} points per spectrum.\n"
            f"- samples.json: {meta['n_test']} unlabeled samples with the same spectrum format."
        )
        head = f"{waves[0]} ... {waves[-1]}"
    else:
        features = meta["features"]
        preview = _profiles_csv(meta["train"], features, bins)
        input_block = (
            f"- train.csv: {meta['n_train']} labeled spectra. Columns: sample_id, label,\n"
            f"  then {n_feat} numeric feature columns.\n"
            f"- test.csv: {meta['n_test']} unlabeled spectra. Columns: sample_id, then the same features."
        )
        head = f"{features[0]} ... {features[-1]}"

    return _shell(task, meta, preview, input_block, n_feat,
                  meta["n_train"], meta["n_test"], labels) + \
        f"\nFeature axis: {head}\n"
