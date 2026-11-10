"""Create a tiny synthetic batch (JSON and CSV) so the pipeline runs offline.

4 classes x 30 train + 10 test, 128 features, deterministic (seed 0).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
classes = ["alpha", "beta", "gamma", "delta"]
n_feat = 128
waves = [str(400 + i) for i in range(n_feat)]
x = np.arange(n_feat)

centers = {"alpha": [20, 55, 90], "beta": [35, 70, 105],
           "gamma": [15, 60, 110], "delta": [25, 80, 100]}


def spectrum(cls):
    y = np.zeros(n_feat)
    for c in centers[cls]:
        y += np.exp(-0.5 * ((x - c) / 2.5) ** 2) * rng.uniform(0.8, 1.2)
    y += rng.normal(0, 0.03, n_feat)
    y += rng.uniform(0.05, 0.15)
    return y


def split():
    ref, test = [], []
    for cls in classes:
        for i in range(30):
            ref.append((f"{cls}_tr{i:02d}", cls, spectrum(cls)))
        for i in range(10):
            test.append((f"{cls}_te{i:02d}", cls, spectrum(cls)))
    return ref, test


def write_json(root, ref, test):
    root.mkdir(parents=True, exist_ok=True)
    (root / "reference_samples.json").write_text(json.dumps(
        [{"sample_id": sid, "spectrum": {w: round(float(v), 6) for w, v in zip(waves, spec)},
          "label": cls} for sid, cls, spec in ref], ensure_ascii=False), encoding="utf-8")
    (root / "samples.json").write_text(json.dumps(
        [{"sample_id": sid, "spectrum": {w: round(float(v), 6) for w, v in zip(waves, spec)}}
         for sid, _, spec in test], ensure_ascii=False), encoding="utf-8")
    (root / "ground_truth.json").write_text(json.dumps(
        [{"sample_id": sid, "label": cls} for sid, cls, _ in test], ensure_ascii=False),
        encoding="utf-8")
    (root / "task.md").write_text(
        "Four-class classification of synthetic spectra.\n"
        "Each class has a fixed set of absorption bands plus noise; the task is to\n"
        "assign every test spectrum to one of the four classes.", encoding="utf-8")


def write_csv(root, ref, test):
    root.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame([[sid, cls] + list(np.round(spec, 6)) for sid, cls, spec in ref],
                         columns=["sample_id", "label"] + waves)
    test_df = pd.DataFrame([[sid] + list(np.round(spec, 6)) for sid, _, spec in test],
                           columns=["sample_id"] + waves)
    train.to_csv(root / "train.csv", index=False)
    test_df.to_csv(root / "test.csv", index=False)
    (root / "ground_truth.json").write_text(json.dumps(
        [{"sample_id": sid, "label": cls} for sid, cls, _ in test], ensure_ascii=False),
        encoding="utf-8")
    (root / "task.md").write_text(
        "Four-class classification of synthetic spectra.\n"
        "Each class has a fixed set of absorption bands plus noise; the task is to\n"
        "assign every test spectrum to one of the four classes.", encoding="utf-8")


if __name__ == "__main__":
    ref, test = split()
    write_json(HERE / "synthetic_json", ref, test)
    ref, test = split()
    write_csv(HERE / "synthetic_csv", ref, test)
    print("wrote examples/synthetic_json and examples/synthetic_csv")
