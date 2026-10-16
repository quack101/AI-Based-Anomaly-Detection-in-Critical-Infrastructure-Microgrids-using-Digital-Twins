"""
Part A -- retrain the reconstruction AE to convergence (no 40-epoch cap;
early-stop on clean-val loss, patience 10, max 200 epochs). Sweep
d in {16,32,48}, pick by clean-val reconstruction loss. Reuses the
gradient-checked AE cell from validation/detect_ae.py (numpy only).

Saves: weights (detect_ae_v2_weights.npz), manifest, per-timestep
scores (s_max, s_mean) for val+test, and per-channel normalized
reconstruction errors e_{c,t}/r_c for val+test (detect_ae_v2_errors.npz,
float32) for the per-channel detector in Part C.

Run: python -m validation.detect_ae_v2 [--gradcheck-only]
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_ae import (init_params, loss_and_grad, forward, windows_of, predict_all,
                                   load_split, gradcheck, W, H, FLOOR)
from validation.detect_eval import threshold_99, roc_auc

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
D_CHOICES = [16, 32, 48]
LR = 1e-3
BATCH = 512
MAX_EPOCHS = 200
PATIENCE = 10
SEED = 1234


def train_one(d, Str, Sva, va_clean_idx, rng):
    D = Str.shape[1]
    P = init_params(D, H, d, rng)
    adam = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in P}
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr_idx = np.arange(W - 1, len(Str))
    step, bad = 0, 0
    best = (np.inf, None, 0)
    hist = []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(tr_idx)
        tl = nb = 0
        for b in range(0, len(tr_idx), BATCH):
            idx = tr_idx[b:b + BATCH]
            loss, grads = loss_and_grad(P, windows_of(Str, idx), Str[idx])
            tl += loss * idx.size; nb += idx.size; step += 1
            for k in P:
                m, v = adam[k]
                m[:] = b1 * m + (1 - b1) * grads[k]
                v[:] = b2 * v + (1 - b2) * grads[k] ** 2
                P[k] -= LR * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        vt = vn = 0.0
        for b in range(0, len(va_clean_idx), 4096):
            idx = va_clean_idx[b:b + 4096]
            out, _ = forward(P, windows_of(Sva, idx))
            vt += np.sum((out - Sva[idx]) ** 2); vn += idx.size * Str.shape[1]
        vl = vt / vn
        hist.append({"epoch": epoch, "train_mse": tl / nb, "clean_val_mse": vl})
        if vl < best[0] - 1e-7:
            best = (vl, {k: P[k].copy() for k in P}, epoch); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best[0], best[1], best[2], hist, time.time() - t0, epoch


def main():
    wc = gradcheck()
    print(f"AE gradient check worst relative error: {wc:.2e}", flush=True)
    assert wc < 1e-5
    if "--gradcheck-only" in sys.argv:
        return

    Xtr, _, cols = load_split("train")
    Xva, lva, _ = load_split("val")
    Xte, lte, _ = load_split("test")
    D = Xtr.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + FLOOR
    Str, Sva, Ste = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    va_att = lva["attacked"].to_numpy()
    va_idx = np.arange(W - 1, len(Sva))
    va_clean_idx = va_idx[va_att[va_idx] == 0]

    results = {}
    for d in D_CHOICES:
        vl, P, be, hist, secs, ep = train_one(d, Str, Sva, va_clean_idx, np.random.default_rng(SEED))
        nparams = sum(v.size for v in P.values())
        print(f"d={d}: best clean_val_mse {vl:.5f} @epoch {be}  (ran {ep}, {secs:.0f}s)  params {nparams}", flush=True)
        results[d] = dict(val_mse=vl, P=P, best_epoch=be, epochs=ep, hist=hist, params=nparams, secs=secs)
    d_best = min(results, key=lambda k: results[k]["val_mse"])
    P = results[d_best]["P"]
    print(f"chosen d={d_best}", flush=True)

    pred_tr = predict_all(P, Str)
    m = np.arange(W - 1, len(Str))
    r_c = (Xtr[m] - (pred_tr[m] * sd + mu)).std(0) + FLOOR

    def errors(S, Xraw):
        pred = predict_all(P, S)
        N = len(S)
        E = np.full((N, D), np.nan, dtype=np.float32)
        idx = np.arange(W - 1, N)
        E[idx] = (np.abs(Xraw[idx] - (pred[idx] * sd + mu)) / r_c).astype(np.float32)
        return E

    Eva, Ete = errors(Sva, Xva), errors(Ste, Xte)
    va_max, va_mean = np.nanmax(Eva, 1), np.nanmean(Eva, 1)
    te_max, te_mean = np.nanmax(Ete, 1), np.nanmean(Ete, 1)
    auc_max = roc_auc(va_max[va_idx], va_att[va_idx].astype(bool))
    auc_mean = roc_auc(va_mean[va_idx], va_att[va_idx].astype(bool))
    chosen = "s_max" if auc_max >= auc_mean else "s_mean"
    print(f"val AUC s_max {auc_max:.4f} s_mean {auc_mean:.4f} -> {chosen}", flush=True)

    np.savez(OUT / "detect_ae_v2_errors.npz", val_err=Eva, test_err=Ete, r_c=r_c,
             val_attacked=va_att, cols=np.array(cols, dtype=object))
    np.savez(OUT / "detect_ae_v2_weights.npz", mu=mu, sd=sd, r_c=r_c, d=d_best, chosen_score=chosen, **P)
    pd.DataFrame({"row": np.arange(len(Ste)), "timestamp": pd.read_csv(FDIA / "test_z.csv")["timestamp"],
                  "s_max": te_max, "s_mean": te_mean, "score": te_max if chosen == "s_max" else te_mean}
                 ).to_csv(OUT / "detect_ae_v2_test_pertimestep.csv", index=False)
    pd.DataFrame({"row": np.arange(len(Sva)), "attacked": va_att,
                  "s_max": va_max, "s_mean": va_mean, "score": va_max if chosen == "s_max" else va_mean}
                 ).to_csv(OUT / "detect_ae_v2_val_pertimestep.csv", index=False)

    def fh(p):
        h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest().upper()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUT.parent).stdout.strip()
    json.dump({"model": "GRU-AE-v2", "hidden": H, "window": W, "d_choices": D_CHOICES, "d_chosen": int(d_best),
               "params": results[d_best]["params"], "best_epoch": results[d_best]["best_epoch"],
               "epochs_run": results[d_best]["epochs"], "chosen_score": chosen,
               "val_mse_by_d": {k: results[k]["val_mse"] for k in results},
               "val_AUC_s_max": float(auc_max), "val_AUC_s_mean": float(auc_mean),
               "patience": PATIENCE, "max_epochs": MAX_EPOCHS, "seed": SEED, "gradcheck": wc,
               "code_commit": commit, "weights_hash": fh(OUT / "detect_ae_v2_weights.npz"),
               "history_best_d": results[d_best]["hist"]}, open(OUT / "detect_ae_v2_manifest.json", "w"), indent=2)
    print(f"saved; d={d_best}, {chosen}, val_mse {results[d_best]['val_mse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
