import sys
import os
import json
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        base = "."
    else:
        base = sys.argv[1]

    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")

    train = pd.read_csv(train_path, dtype={"sample_id": str})
    test = pd.read_csv(test_path, dtype={"sample_id": str})

    labels = [
        '1,3-Dimethyl-2-imidazolidinone',
        '2 - Propanol',
        '2, 2 - Dimethoxy Propane',
        '4-Methyl-2-pentanone',
        'Acetic acid',
        'Acetone',
        'Acetonitrile',
        'Benzaldehyde',
        'Benzyl bromide',
        'Butyl Acetate',
        'Chloroform',
        'Cyclohexane',
        'Dichloromethane',
        'Diethyl Malonate',
        'Diethylamine',
        'Diethylene Glycol',
        'Dimethyl Sulfoxide',
        'Ethanol',
        'Ethyl Acetate',
        'Ethylene Glycol',
        'Formic acid',
        'Isobutylamine',
        'Methanol',
        'Methyl Isobutyl Ketone',
        'N,N - Dimethylformahide',
        'Propyl acetate',
        'Tert-butanol',
        'Tert-butyl methyl ether',
        'Tetrahydrofuran',
        'Toluene',
        'n-Heptane',
        'n-Hexane'
    ]

    label_to_idx = {str(lab).strip(): i for i, lab in enumerate(labels)}
    y = np.empty(len(train), dtype=int)
    for i, lab in enumerate(train["label"].values):
        key = str(lab).strip()
        y[i] = label_to_idx.get(key, -1)

    if np.any(y < 0):
        unique_labels = []
        for lab in train["label"].values:
            s = str(lab).strip()
            if s not in unique_labels:
                unique_labels.append(s)
        labels = unique_labels
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        for i, lab in enumerate(train["label"].values):
            y[i] = label_to_idx[str(lab).strip()]

    C = len(labels)

    X_train_raw = train.iloc[:, 2:].values.astype(np.float32)
    X_test_raw = test.iloc[:, 1:].values.astype(np.float32)
    test_ids = test["sample_id"].astype(str).values

    width = 4
    n_train, n_cols = X_train_raw.shape
    n_bins = n_cols // width
    Xtr_bin = X_train_raw[:, : n_bins * width].reshape(n_train, n_bins, width).mean(axis=2)

    n_test = X_test_raw.shape[0]
    Xte_bin = X_test_raw[:, : n_bins * width].reshape(n_test, n_bins, width).mean(axis=2)

    def snv(X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std

    def l2norm(X):
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return X / norm

    variants = []
    variants.append(("snv", snv(Xtr_bin), snv(Xte_bin)))
    variants.append(("l2", l2norm(Xtr_bin), l2norm(Xte_bin)))

    def make_folds(y, n_folds=4, seed=42):
        rng = np.random.default_rng(seed)
        folds_val = [[] for _ in range(n_folds)]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            for i, j in enumerate(idx):
                folds_val[i % n_folds].append(j)

        all_idx = np.arange(len(y))
        folds = []
        for val_list in folds_val:
            val = np.array(val_list, dtype=int)
            mask = np.ones(len(y), dtype=bool)
            mask[val] = False
            tr = all_idx[mask]
            folds.append((tr, val))
        return folds

    folds = make_folds(y, n_folds=4, seed=42)

    def lda_predict(Xval, means, counts, S, reg):
        d = S.shape[0]
        tr = np.trace(S) / d
        S_reg = (1.0 - reg) * S + reg * tr * np.eye(d, dtype=S.dtype)
        S_reg += 1e-8 * (tr + 1e-12) * np.eye(d, dtype=S.dtype)
        W = np.linalg.solve(S_reg, means.T)
        log_prior = np.log((counts + 1e-12) / counts.sum())
        scores = Xval @ W - 0.5 * np.sum(means.T * W, axis=0) + log_prior
        return scores.argmax(axis=1)

    def run_cv(Ztr, y, C, folds, max_k):
        dims = [50, 100, 150]
        dims = [d for d in dims if d <= max_k]
        ks = [1, 3, 5, 7, 11]
        regs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7]
        smooths = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        correct = {}
        n_total = len(y)

        for dim in dims:
            Z = Ztr[:, :dim].astype(np.float64)

            for tr, val in folds:
                Xtr = Z[tr]
                ytr = y[tr]
                Xval = Z[val]
                yval = y[val]
                m = len(yval)

                means = np.zeros((C, dim), dtype=np.float64)
                counts = np.bincount(ytr, minlength=C).astype(np.float64)
                for c in range(C):
                    if counts[c] > 0:
                        means[c] = Xtr[ytr == c].mean(axis=0)

                d2 = (
                    np.sum(Xval ** 2, axis=1)[:, None]
                    + np.sum(means ** 2, axis=1)[None, :]
                    - 2.0 * Xval @ means.T
                )
                pred = d2.argmin(axis=1)
                key = (dim, "centroid")
                correct[key] = correct.get(key, 0) + int(np.sum(pred == yval))

                tr_sq = np.sum(Xtr ** 2, axis=1)
                va_sq = np.sum(Xval ** 2, axis=1)
                D = va_sq[:, None] + tr_sq[None, :] - 2.0 * Xval @ Xtr.T
                np.maximum(D, 0, out=D)
                rows = np.arange(m)

                for k in ks:
                    if k > Xtr.shape[0]:
                        continue
                    idx_k = np.argpartition(D, k - 1, axis=1)[:, :k]

                    votes = np.zeros((m, C), dtype=np.float64)
                    for j in range(k):
                        votes[rows, ytr[idx_k[:, j]]] += 1.0
                    pred = votes.argmax(axis=1)
                    key = (dim, "knn", k, False)
                    correct[key] = correct.get(key, 0) + int(np.sum(pred == yval))

                    dist = np.take_along_axis(D, idx_k, axis=1)
                    weights = 1.0 / (dist + 1e-6)
                    votes = np.zeros((m, C), dtype=np.float64)
                    for j in range(k):
                        votes[rows, ytr[idx_k[:, j]]] += weights[:, j]
                    pred = votes.argmax(axis=1)
                    key = (dim, "knn", k, True)
                    correct[key] = correct.get(key, 0) + int(np.sum(pred == yval))

                S = np.zeros((dim, dim), dtype=np.float64)
                for c in range(C):
                    if counts[c] > 1:
                        Xc = Xtr[ytr == c] - means[c]
                        S += Xc.T @ Xc
                denom = len(Xtr) - C
                if denom > 0:
                    S /= denom
                else:
                    S /= max(1, len(Xtr))

                for reg in regs:
                    pred = lda_predict(Xval, means, counts, S, reg)
                    key = (dim, "lda", reg)
                    correct[key] = correct.get(key, 0) + int(np.sum(pred == yval))

                pooled_var = np.var(Xtr, axis=0) + 1e-8
                class_vars = np.zeros((C, dim), dtype=np.float64)
                for c in range(C):
                    if counts[c] > 1:
                        class_vars[c] = np.var(Xtr[ytr == c], axis=0)
                    else:
                        class_vars[c] = pooled_var

                log_prior = np.log((counts + 1e-12) / counts.sum())

                for smooth in smooths:
                    scores = np.zeros((m, C), dtype=np.float64)
                    for c in range(C):
                        var_c = class_vars[c]
                        if smooth > 0:
                            var_c = (1.0 - smooth) * var_c + smooth * pooled_var
                        var_c = var_c + 1e-6 * pooled_var.mean()
                        logdet = np.sum(np.log(var_c))
                        diff = Xval - means[c]
                        score = (
                            -0.5 * np.sum(diff * diff / var_c, axis=1)
                            - 0.5 * logdet
                            + log_prior[c]
                        )
                        scores[:, c] = score
                    pred = scores.argmax(axis=1)
                    key = (dim, "qda", smooth)
                    correct[key] = correct.get(key, 0) + int(np.sum(pred == yval))

        best_acc = -1.0
        best_key = None
        for key, corr in correct.items():
            acc = corr / n_total
            if acc > best_acc:
                best_acc = acc
                best_key = key

        if best_key[1] == "centroid":
            config = {"dim": best_key[0], "method": "centroid", "acc": best_acc}
        elif best_key[1] == "knn":
            config = {
                "dim": best_key[0],
                "method": "knn",
                "k": best_key[2],
                "weighted": best_key[3],
                "acc": best_acc,
            }
        elif best_key[1] == "lda":
            config = {
                "dim": best_key[0],
                "method": "lda",
                "reg": best_key[2],
                "acc": best_acc,
            }
        else:
            config = {
                "dim": best_key[0],
                "method": "qda",
                "smooth": best_key[2],
                "acc": best_acc,
            }

        return config

    best_global = {"acc": -1.0}

    for name, Xtr_proc, Xte_proc in variants:
        mean_x = Xtr_proc.mean(axis=0, keepdims=True)
        Xc = (Xtr_proc - mean_x).astype(np.float64)
        U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
        max_k = min(180, Vh.shape[0])
        V = Vh[:max_k].T
        Ztr = Xc @ V

        config = run_cv(Ztr, y, C, folds, max_k)

        if config["acc"] > best_global["acc"]:
            best_global = config.copy()
            best_global["variant"] = name
            best_global["V"] = V
            best_global["mean_x"] = mean_x
            best_global["Ztr_full"] = Ztr
            best_global["Xte_proc"] = Xte_proc

    dim = best_global["dim"]
    Ztr_full = best_global["Ztr_full"][:, :dim]
    V = best_global["V"][:, :dim]
    mean_x = best_global["mean_x"]
    Zte = (best_global["Xte_proc"] - mean_x) @ V
    method = best_global["method"]

    if method == "centroid":
        means = np.zeros((C, dim), dtype=np.float64)
        counts = np.bincount(y, minlength=C).astype(np.float64)
        for c in range(C):
            means[c] = Ztr_full[y == c].mean(axis=0)
        d2 = (
            np.sum(Zte ** 2, axis=1)[:, None]
            + np.sum(means ** 2, axis=1)[None, :]
            - 2.0 * Zte @ means.T
        )
        pred_idx = d2.argmin(axis=1)

    elif method == "knn":
        k = best_global["k"]
        weighted = best_global["weighted"]
        tr_sq = np.sum(Ztr_full ** 2, axis=1)
        te_sq = np.sum(Zte ** 2, axis=1)
        D = te_sq[:, None] + tr_sq[None, :] - 2.0 * Zte @ Ztr_full.T
        np.maximum(D, 0, out=D)
        idx_k = np.argpartition(D, k - 1, axis=1)[:, :k]
        rows = np.arange(len(Zte))
        votes = np.zeros((len(Zte), C), dtype=np.float64)
        if weighted:
            dist = np.take_along_axis(D, idx_k, axis=1)
            weights = 1.0 / (dist + 1e-6)
            for j in range(k):
                votes[rows, y[idx_k[:, j]]] += weights[:, j]
        else:
            for j in range(k):
                votes[rows, y[idx_k[:, j]]] += 1.0
        pred_idx = votes.argmax(axis=1)

    elif method == "lda":
        means = np.zeros((C, dim), dtype=np.float64)
        counts = np.bincount(y, minlength=C).astype(np.float64)
        for c in range(C):
            means[c] = Ztr_full[y == c].mean(axis=0)
        S = np.zeros((dim, dim), dtype=np.float64)
        for c in range(C):
            Xc = Ztr_full[y == c] - means[c]
            S += Xc.T @ Xc
        S /= (len(y) - C)
        pred_idx = lda_predict(Zte, means, counts, S, best_global["reg"])

    else:
        smooth = best_global["smooth"]
        means = np.zeros((C, dim), dtype=np.float64)
        counts = np.bincount(y, minlength=C).astype(np.float64)
        pooled_var = np.var(Ztr_full, axis=0) + 1e-8
        class_vars = np.zeros((C, dim), dtype=np.float64)
        for c in range(C):
            Xc = Ztr_full[y == c]
            means[c] = Xc.mean(axis=0)
            class_vars[c] = np.var(Xc, axis=0)

        log_prior = np.log((counts + 1e-12) / counts.sum())
        scores = np.zeros((len(Zte), C), dtype=np.float64)

        for c in range(C):
            var_c = class_vars[c]
            if smooth > 0:
                var_c = (1.0 - smooth) * var_c + smooth * pooled_var
            var_c = var_c + 1e-6 * pooled_var.mean()
            logdet = np.sum(np.log(var_c))
            diff = Zte - means[c]
            score = (
                -0.5 * np.sum(diff * diff / var_c, axis=1)
                - 0.5 * logdet
                + log_prior[c]
            )
            scores[:, c] = score
        pred_idx = scores.argmax(axis=1)

    pred_labels = [labels[i] for i in pred_idx]
    predictions = [
        {"sample_id": sid, "prediction": lab}
        for sid, lab in zip(test_ids, pred_labels)
    ]

    with open("predictions.json", "w") as f:
        json.dump({"predictions": predictions}, f)


if __name__ == "__main__":
    main()