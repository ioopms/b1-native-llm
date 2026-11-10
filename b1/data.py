"""Batch loading for the two supported on-disk formats.

JSON format
    reference_samples.json : [{"sample_id", "spectrum": {wavenumber: intensity}, "label"}]
    samples.json           : [{"sample_id", "spectrum": {...}}]
    ground_truth.json      : [{"sample_id", "label"}]

CSV format
    train.csv : sample_id, label, <feature columns...>
    test.csv  : sample_id, <feature columns...>
    ground_truth.json : [{"sample_id", "label"}]
"""
import json
from pathlib import Path

import pandas as pd


def detect_format(batch: Path) -> str:
    if (batch / "train.csv").exists():
        return "csv"
    if (batch / "reference_samples.json").exists():
        return "json"
    raise FileNotFoundError(f"{batch} 既不是 JSON 批次也不是 CSV 批次")


def load(batch: Path) -> dict:
    fmt = detect_format(batch)
    truth = {r["sample_id"]: r["label"] for r in
             json.loads((batch / "ground_truth.json").read_text(encoding="utf-8"))}
    meta = {"format": fmt, "truth": truth, "n_test": len(truth), "batch": batch}

    if fmt == "json":
        ref = json.loads((batch / "reference_samples.json").read_text(encoding="utf-8"))
        test = json.loads((batch / "samples.json").read_text(encoding="utf-8"))
        waves = list(ref[0]["spectrum"].keys())
        meta.update(
            ref=ref, test=test, waves=waves,
            labels=sorted({r["label"] for r in ref}),
            n_train=len(ref), n_features=len(waves),
        )
    else:
        train = pd.read_csv(batch / "train.csv")
        test = pd.read_csv(batch / "test.csv")
        feat = [c for c in train.columns if c not in ("sample_id", "label")]
        meta.update(
            train=train, test=test, features=feat,
            labels=sorted(train["label"].astype(str).unique()),
            n_train=len(train), n_features=len(feat),
        )
    return meta
