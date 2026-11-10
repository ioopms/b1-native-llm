import sys
import os
import json
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def transform_X(X, kind):
    X = np.asarray(X, dtype=float)
    if kind == "raw":
        Z = X.copy()
    elif kind == "raw_l2":
        Z = X.copy()
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norm + 1e-12)
    elif kind == "center":
        Z = X - X.mean(axis=1, keepdims=True)
    elif kind == "center_l2":
        Z = X - X.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norm + 1e-12)
    elif kind == "median":
        Z = X - np.median(X, axis=1, keepdims=True)
    elif kind == "median_l2":
        Z = X - np.median(X, axis=1, keepdims=True)
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norm + 1e-12)
    elif kind == "diff_l2":
        Z = np.diff(X, axis=1)
        Z = Z - Z.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norm + 1e-12)
    elif kind == "smooth_center_l2":
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        kernel = kernel / kernel.sum()
        Z = np.empty_like(X)
        for i in range(X.shape[0]):
            Z[i] = np.convolve(X[i], kernel, mode="same")
        Z = Z - Z.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / (norm + 1e-12)
    else:
        Z = X.copy()
    return Z


def fit_predict(Xtr, ytr, Xte, nclass, typ, param):
    if typ == "centroid":
        mus = np.zeros((nclass, Xtr.shape[1]), dtype=float)
        for c in range(nclass):
            Xc = Xtr[ytr == c]
            if len(Xc) > 0:
                mus[c] = Xc.mean(axis=0)
        pred = np.zeros(len(Xte), dtype=int)
        for i in range(len(Xte)):
            d = np.sum((mus - Xte[i]) ** 2, axis=1)
            pred[i] = int(np.argmin(d))
        return pred

    elif typ == "lda":
        lam = float(param)
        p = Xtr.shape[1]
        n = len(ytr)
        mus = np.zeros((nclass, p), dtype=float)
        for c in range(nclass):
            Xc = Xtr[ytr == c]
            if len(Xc) > 0:
                mus[c] = Xc.mean(axis=0)

        R = Xtr.copy()
        for c in range(nclass):
            mask = (ytr == c)
            if np.any(mask):
                R[mask] = R[mask] - mus[c]

        denom = max(1, n - nclass)
        S = (R.T @ R) / denom
        scale = np.trace(S) / p
        if scale <= 0:
            scale = 1.0

        Sreg = (1.0 - lam) * S + lam * scale * np.eye(p)
        Sreg += np.eye(p) * (1e-10 * scale + 1e-12)
        invS = np.linalg.inv(Sreg)

        scores = np.zeros((len(Xte), nclass), dtype=float)
        for c in range(nclass):
            D = Xte - mus[c]
            scores[:, c] = -0.5 * np.sum(D * (D @ invS), axis=1)
        return np.argmax(scores, axis=1).astype(int)

    elif typ == "knn":
        k = int(param)
        nte = len(Xte)
        pred = np.zeros(nte, dtype=int)
        d2 = np.sum((Xte[:, None, :] - Xtr[None, :, :]) ** 2, axis=2)
        for i in range(nte):
            idx = np.argsort(d2[i])[:k]
            votes = np.zeros(nclass, dtype=float)
            for j in idx:
                votes[ytr[j]] += 1.0 / (np.sqrt(d2[i, j]) + 1e-9)
            pred[i] = int(np.argmax(votes))
        return pred

    else:
        return np.zeros(len(Xte), dtype=int)


def cross_validate(X, y, nclass, configs):
    K = 5
    n = len(y)
    fold = np.zeros(n, dtype=int)

    for c in range(nclass):
        idx = np.where(y == c)[0]
        for j, i in enumerate(idx):
            fold[i] = j % K

    accs = []
    for prep, typ, param in configs:
        Z = transform_X(X, prep)
        correct = 0
        for f in range(K):
            tr = fold != f
            va = fold == f
            pred = fit_predict(Z[tr], y[tr], Z[va], nclass, typ, param)
            correct += int(np.sum(pred == y[va]))
        accs.append(correct / float(n))
    return accs


def main():
    input_dir = sys.argv[1]
    ref = load_json(os.path.join(input_dir, "reference_samples.json"))
    test = load_json(os.path.join(input_dir, "samples.json"))

    ref_keys = sorted(ref[0]["spectrum"].keys(), key=lambda k: float(k))
    ref_floats = [float(k) for k in ref_keys]

    def get_values(rec):
        spec = rec["spectrum"]
        if all(k in spec for k in ref_keys):
            return [float(spec[k]) for k in ref_keys]
        mp = {float(k): float(v) for k, v in spec.items()}
        return [mp[f] for f in ref_floats]

    X = np.array([get_values(rec) for rec in ref], dtype=float)
    Xte = np.array([get_values(rec) for rec in test], dtype=float)

    y_labels = [rec["label"] for rec in ref]
    labels = sorted(set(y_labels))
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    y = np.array([lab2idx[lab] for lab in y_labels], dtype=int)
    nclass = len(labels)

    preps = [
        "center_l2",
        "median_l2",
        "center",
        "median",
        "raw",
        "raw_l2",
        "diff_l2",
        "smooth_center_l2",
    ]

    configs = []
    for prep in preps:
        configs.append((prep, "centroid", None))
    for prep in preps:
        for lam in [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]:
            configs.append((prep, "lda", lam))
    for prep in preps:
        for k in [1, 3, 5, 7]:
            configs.append((prep, "knn", k))

    accs = cross_validate(X, y, nclass, configs)
    best_idx = int(np.argmax(accs))
    prep, typ, param = configs[best_idx]

    Ztr = transform_X(X, prep)
    Zte = transform_X(Xte, prep)
    pred_idx = fit_predict(Ztr, y, Zte, nclass, typ, param)

    predictions = []
    for i, rec in enumerate(test):
        predictions.append({
            "sample_id": rec["sample_id"],
            "prediction": labels[int(pred_idx[i])],
        })

    with open("predictions.json", "w") as f:
        json.dump({"predictions": predictions}, f)


if __name__ == "__main__":
    main()