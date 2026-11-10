"""Reference candidate for CSV batches: SNV per spectrum + shrinkage LDA."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

batch = Path(sys.argv[1])
train = pd.read_csv(batch / "train.csv")
test = pd.read_csv(batch / "test.csv")
features = [c for c in train.columns if c not in ("sample_id", "label")]
labels = sorted(train["label"].astype(str).unique())

X = train[features].to_numpy(np.float64)
y = train["label"].astype(str).to_numpy()
Xt = test[features].to_numpy(np.float64)

X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
Xt = (Xt - Xt.mean(1, keepdims=True)) / (Xt.std(1, keepdims=True) + 1e-12)

k, d = len(labels), X.shape[1]
C = np.stack([X[y == lab].mean(0) for lab in labels])
S = np.zeros((d, d))
for lab in labels:
    Z = X[y == lab] - C[labels.index(lab)]
    S += Z.T @ Z
S /= len(y) - k
shrink = 0.1
S = (1 - shrink) * S + shrink * np.trace(S) / d * np.eye(d)
W = np.linalg.solve(S, C.T)
prior = np.log(np.bincount([labels.index(v) for v in y], minlength=k) / len(y))
scores = Xt @ W + prior - 0.5 * (C * W.T).sum(1)
pred = scores.argmax(1)

out = {"predictions": [{"sample_id": s, "prediction": labels[p]}
                       for s, p in zip(test["sample_id"].astype(str), pred)]}
Path("predictions.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(pred)} predictions over {k} classes")
