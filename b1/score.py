"""Local scoring. Test labels are read only here, after all model calls."""


def _canon(label, synonyms):
    return synonyms.get(label, label)


def evaluate(truth: dict, preds, labels: list, synonyms: dict = None) -> dict:
    synonyms = synonyms or {}
    rows = preds.get("predictions") if isinstance(preds, dict) else None
    if not isinstance(rows, list):
        raise ValueError('predictions.json must be {"predictions": [...]}')
    ids = [r.get("sample_id") for r in rows]
    if len(ids) != len(truth) or set(ids) != set(truth) or len(set(ids)) != len(ids):
        raise ValueError(f"prediction rows do not match test IDs: {len(ids)} vs {len(truth)}")

    canon_labels = sorted({_canon(v, synonyms) for v in truth.values()})
    correct_raw = 0
    correct = 0
    invalid = 0
    matrix = [[0] * len(canon_labels) for _ in canon_labels]
    errors = []
    for row in rows:
        sid = row["sample_id"]
        true_label = truth[sid]
        pred_label = row.get("prediction")
        if pred_label == true_label:
            correct_raw += 1
        if pred_label not in labels:
            invalid += 1
            errors.append((sid, true_label, pred_label))
            continue
        ct, cp = _canon(true_label, synonyms), _canon(pred_label, synonyms)
        if ct == cp:
            correct += 1
        else:
            errors.append((sid, true_label, pred_label))
        matrix[canon_labels.index(ct)][canon_labels.index(cp)] += 1

    per_class = []
    for i, label in enumerate(canon_labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(len(canon_labels))) - tp
        fn = sum(matrix[i]) - tp
        denom = 2 * tp + fp + fn
        per_class.append({"label": label, "f1": 2 * tp / denom if denom else 0.0,
                          "support": sum(matrix[i])})

    n = len(rows)
    return {
        "n": n,
        "correct": correct,
        "correct_raw": correct_raw,
        "accuracy": correct / n if n else 0.0,
        "accuracy_raw": correct_raw / n if n else 0.0,
        "macro_f1": sum(c["f1"] for c in per_class) / len(per_class),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": canon_labels,
        "invalid": invalid,
        "errors": errors,
    }
