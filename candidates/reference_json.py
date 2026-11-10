"""Reference candidate for JSON batches: z-score + shrinkage LDA.

This is the bundled baseline used by `python -m b1 selftest`. It shows what a
competent generated program can achieve; it is not a model output.
"""
import json
import sys
from pathlib import Path

import numpy as np

batch = Path(sys.argv[1])
ref = json.loads((batch / "reference_samples.json").read_text(encoding="utf-8"))
test = json.loads((batch / "samples.json").read_text(encoding="utf-8"))
labels = sorted({r["label"] for r in ref})
waves = list(ref[0]["spectrum"].keys())

X = np.array([[r["spectrum"][w] for w in waves] for r in ref], float)
y = np.array([labels.index(r["label"]) for r in ref])
Xt = np.array([[s["spectrum"][w] for w in waves] for s in test], float)

mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1.0
Xz, Xtz = (X - mu) / sd, (Xt - mu) / sd

k, d = len(labels), Xz.shape[1]
C = np.stack([Xz[y == c].mean(0) for c in range(k)])
S = np.zeros((d, d))
for c in range(k):
    Z = Xz[y == c] - C[c]
    S += Z.T @ Z
S /= len(y) - k
shrink = 0.1
S = (1 - shrink) * S + shrink * np.trace(S) / d * np.eye(d)
W = np.linalg.solve(S, C.T)
prior = np.log(np.bincount(y, minlength=k) / len(y))
pred = (Xtz @ W + prior - 0.5 * (C * W.T).sum(1)).argmax(1)

out = {"predictions": [{"sample_id": s["sample_id"], "prediction": labels[p]}
                       for s, p in zip(test, pred)]}
Path("predictions.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(pred)} predictions over {k} classes")
