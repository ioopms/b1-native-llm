"""Build a B1 CSV batch from the Raman API-compounds dataset.

Source (public, Scientific Data 2025):
  raman_spectra_api_compounds.csv
  3510 spectra x 3276 wavenumbers (150-3425 cm-1), 32 compound labels.

Usage:
  python tools/build_raman_batch.py --csv path/to/raman_spectra_api_compounds.csv \
      --out data/raman_b1
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.30)
    args = ap.parse_args()

    src = Path(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    labels = df["label"].astype(str).values
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = [], []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_test = max(1, min(int(round(len(idx) * args.test_frac)), len(idx) - 1))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    all_ids = np.array([f"api_{i:05d}" for i in range(len(df))])
    train = df.iloc[sorted(train_idx)].copy()
    test = df.iloc[sorted(test_idx)].copy()
    train.insert(0, "sample_id", all_ids[sorted(train_idx)])
    test.insert(0, "sample_id", all_ids[sorted(test_idx)])

    train.to_csv(out / "train.csv", index=False, float_format="%.4f")
    test.drop(columns=["label"]).to_csv(out / "test.csv", index=False, float_format="%.4f")
    truth = [{"sample_id": str(s), "label": str(l)}
             for s, l in zip(test["sample_id"], test["label"])]
    (out / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
    (out / "synonyms.json").write_text(
        json.dumps({"4-Methyl-2-pentanone": "Methyl Isobutyl Ketone"}, ensure_ascii=False),
        encoding="utf-8")
    (out / "task.md").write_text(
        "Multi-class classification of chemical compounds from Raman spectra.\n"
        "Medium: organic solvents and reagents used in active pharmaceutical ingredient (API) development.\n"
        "Spectrum: Raman, 785 nm excitation, 150-3425 cm-1, 1 cm-1 resolution, 3276 points per spectrum.",
        encoding="utf-8")

    meta = {
        "source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(df.shape[1] - 1),
        "labels": sorted(df["label"].astype(str).unique().tolist()),
        "split": {"method": "stratified_by_compound", "seed": args.seed,
                  "test_frac": args.test_frac},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train {len(train)} / test {len(test)} / features {meta['n_features']} / classes {len(meta['labels'])}")
    print("saved to", out)


if __name__ == "__main__":
    main()
