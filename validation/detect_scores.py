"""
Shared score loader for the improved detectors (Parts B-E). Builds the
per-timestep scores for bdd, ae (AE v2) and fusion_LR (refit on the new
AE score) over val and test, plus hour-of-day and the AE v2 per-channel
error matrices. Estimator is NOT run; all inputs are saved logs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import roc_auc

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
CORPUS = OUT.parent / "simulation" / "phase6_corpus"


def _logistic_fit(X, y, iters=4000, lr=0.5, l2=1e-4):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / n + l2 * w)
        b -= lr * np.mean(p - y)
    return w, b


def _hours(split):
    ts = pd.read_csv(CORPUS / f"{split}_telemetry.csv", usecols=["real_timestamp"])["real_timestamp"]
    return np.array([int(str(t)[11:13]) for t in ts])


def load_scores():
    vlab = pd.read_csv(FDIA / "val_labels.csv")
    tlab = pd.read_csv(FDIA / "test_labels.csv")
    nv, nt = len(vlab), len(tlab)

    vs = pd.read_csv(OUT / "val_scores.csv")                      # bdd on all val
    bdd_v = vs["bdd_score"].to_numpy()
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    bdd_t = np.full(nt, np.nan)
    for r, s in zip(phys["row"].astype(int), phys["bdd_score"].astype(float)):
        bdd_t[r] = s

    aev = pd.read_csv(OUT / "detect_ae_v2_val_pertimestep.csv")
    aet = pd.read_csv(OUT / "detect_ae_v2_test_pertimestep.csv")
    ae_v = np.full(nv, np.nan); ae_t = np.full(nt, np.nan)
    ae_v[aev["row"].astype(int)] = aev["score"].astype(float)
    ae_t[aet["row"].astype(int)] = aet["score"].astype(float)

    y_v = vlab["attacked"].to_numpy().astype(int)
    both_v = np.isfinite(bdd_v) & np.isfinite(ae_v)
    clean_v = both_v & (y_v == 0)
    mu = np.array([bdd_v[clean_v].mean(), ae_v[clean_v].mean()])
    sd = np.array([bdd_v[clean_v].std(), ae_v[clean_v].std()]) + 1e-12
    Xv = np.column_stack([(bdd_v[both_v] - mu[0]) / sd[0], (ae_v[both_v] - mu[1]) / sd[1]])
    w, b = _logistic_fit(Xv, y_v[both_v])
    auc_lr = roc_auc(1 / (1 + np.exp(-(Xv @ w + b))), y_v[both_v])

    def lr_score(bdd, ae):
        zb = (bdd - mu[0]) / sd[0]; za = (ae - mu[1]) / sd[1]
        return 1.0 / (1.0 + np.exp(-(zb * w[0] + za * w[1] + b)))

    fus_v = lr_score(bdd_v, ae_v)
    fus_t = lr_score(bdd_t, ae_t)

    err = np.load(OUT / "detect_ae_v2_errors.npz", allow_pickle=True)
    return {
        "vlab": vlab, "tlab": tlab,
        "val": {"bdd": bdd_v, "ae": ae_v, "fusion_LR": fus_v, "attacked": y_v, "hour": _hours("val")},
        "test": {"bdd": bdd_t, "ae": ae_t, "fusion_LR": fus_t,
                 "attacked": tlab["attacked"].to_numpy().astype(int), "hour": _hours("test")},
        "lr": {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(), "val_AUC": float(auc_lr)},
        "val_err": err["val_err"], "test_err": err["test_err"], "r_c": err["r_c"],
        "cols": list(err["cols"]),
        "detectors": ["bdd", "ae", "fusion_LR"],
    }
