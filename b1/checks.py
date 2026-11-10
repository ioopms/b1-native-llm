"""Behavioural checks for generated programs.

order_invariance : shuffling the rows of the input files must not change the
                   per-sample predictions of a legitimate program.
label_sensitivity: cyclically permuting the training labels must change the
                   predictions; a program that ignores labels is hardcoded.
"""
import numpy as np

from . import sandbox


def _predict_map(preds):
    return {r["sample_id"]: r["prediction"] for r in preds.get("predictions", [])}


def _compare(base, other):
    a, b = _predict_map(base), _predict_map(other or {})
    changed = sum(1 for key in a if a[key] != b.get(key))
    return changed


def _run(batch, meta, code, timeout, ref=None, test=None, train=None, test_df=None):
    if meta["format"] == "json":
        box = sandbox.make_json_sandbox(batch, code, ref=ref, test=test)
    else:
        box = sandbox.make_csv_sandbox(batch, code, train=train, test=test_df)
    sandbox.run_sandbox(box, timeout)
    return sandbox.read_predictions(box)


def order_invariance(batch, meta, code, base_preds, timeout, seed=0):
    rng = np.random.default_rng(seed)
    if meta["format"] == "json":
        ref = list(meta["ref"])
        test = list(meta["test"])
        rng.shuffle(ref)
        rng.shuffle(test)
        preds = _run(batch, meta, code, timeout, ref=ref, test=test)
    else:
        train = meta["train"].iloc[rng.permutation(len(meta["train"]))].reset_index(drop=True)
        test = meta["test"].iloc[rng.permutation(len(meta["test"]))].reset_index(drop=True)
        preds = _run(batch, meta, code, timeout, train=train, test_df=test)
    if preds is None:
        return {"ok": False, "changed": None, "reason": "no output"}
    changed = _compare(base_preds, preds)
    return {"ok": changed == 0, "changed": changed}


def label_sensitivity(batch, meta, code, base_preds, timeout):
    labels = meta["labels"]
    mapping = {lab: labels[(i + 1) % len(labels)] for i, lab in enumerate(labels)}
    if meta["format"] == "json":
        ref = []
        for row in meta["ref"]:
            row = dict(row)
            row["label"] = mapping[row["label"]]
            ref.append(row)
        preds = _run(batch, meta, code, timeout, ref=ref)
    else:
        train = meta["train"].copy()
        train["label"] = train["label"].astype(str).map(mapping)
        preds = _run(batch, meta, code, timeout, train=train)
    if preds is None:
        return {"ok": False, "changed": None, "reason": "no output"}
    changed = _compare(base_preds, preds)
    return {"ok": changed > 0, "changed": changed}
