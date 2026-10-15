"""
Part B -- GRU next-step-prediction anomaly detector on dataset/fdia_v1.

Reads telemetry z only. A 1-layer GRU (pure numpy; no ML framework is
available and installing one would risk the frozen estimator's numpy)
predicts the next standardized telemetry vector from the previous W=10
standardized steps. Channels are standardized by TRAIN mean/std. The
model trains on CLEAN TRAIN only (next-step MSE, Adam), early-stopping
on CLEAN VAL prediction loss.

Score: per timestep t (t >= W), s_t = max over the 168 channels of
|z_t - zhat_t| / r_c, where r_c is the per-channel residual std on the
train split (argmax channel saved for localization). Threshold = 99th
percentile of s over clean val (the common 1% operating point); applied
to test. Metrics via validation/detect_eval.py.

Deterministic: fixed seeds; weights + manifest saved.

Run: python -m validation.detect_gru [--gradcheck-only]
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import evaluate_flags, threshold_99

BASE_DIR = Path(__file__).resolve().parent.parent
FDIA = BASE_DIR / "dataset" / "fdia_v1"
OUT = BASE_DIR / "validation"
W = 10
H = 96
LR = 1e-3
BATCH = 512
MAX_EPOCHS = 40
PATIENCE = 5
SEED = 1234
SIGMA_FLOOR = 1e-8


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(D, H, rng):
    def g(a, b):
        return rng.standard_normal((a, b)) * (1.0 / np.sqrt(b))
    P = {}
    for k in ["Wz", "Wr", "Wn"]:
        P[k] = g(D, H)
    for k in ["Uz", "Ur", "Un"]:
        P[k] = g(H, H)
    for k in ["bz", "br", "bn"]:
        P[k] = np.zeros(H)
    P["Wy"] = g(H, D)
    P["by"] = np.zeros(D)
    return P


def forward(P, Xb):
    B, Wn, D = Xb.shape
    h = np.zeros((B, P["bz"].shape[0]))
    cache = []
    for t in range(Wn):
        x = Xb[:, t, :]
        z = sig(x @ P["Wz"] + h @ P["Uz"] + P["bz"])
        r = sig(x @ P["Wr"] + h @ P["Ur"] + P["br"])
        hUn = h @ P["Un"]
        nn = np.tanh(x @ P["Wn"] + r * hUn + P["bn"])
        h_new = (1 - z) * nn + z * h
        cache.append((x, h, z, r, nn, hUn))
        h = h_new
    out = h @ P["Wy"] + P["by"]
    return out, cache, h


def backward(P, cache, dout, h_last):
    grads = {k: np.zeros_like(P[k]) for k in P}
    grads["Wy"] = h_last.T @ dout
    grads["by"] = dout.sum(0)
    dh = dout @ P["Wy"].T
    for t in range(len(cache) - 1, -1, -1):
        x, h_prev, z, r, nn, hUn = cache[t]
        dz = dh * (h_prev - nn)
        dn = dh * (1 - z)
        dh_prev = dh * z
        dan = dn * (1 - nn ** 2)
        grads["Wn"] += x.T @ dan
        grads["bn"] += dan.sum(0)
        dr = dan * hUn
        d_hUn = dan * r
        grads["Un"] += h_prev.T @ d_hUn
        dh_prev = dh_prev + d_hUn @ P["Un"].T
        daz = dz * z * (1 - z)
        grads["Wz"] += x.T @ daz
        grads["bz"] += daz.sum(0)
        grads["Uz"] += h_prev.T @ daz
        dh_prev = dh_prev + daz @ P["Uz"].T
        dar = dr * r * (1 - r)
        grads["Wr"] += x.T @ dar
        grads["br"] += dar.sum(0)
        grads["Ur"] += h_prev.T @ dar
        dh_prev = dh_prev + dar @ P["Ur"].T
        dh = dh_prev
    return grads


def mse_and_grad(P, Xb, Yb):
    out, cache, h_last = forward(P, Xb)
    diff = out - Yb
    loss = float(np.mean(diff ** 2))
    dout = 2.0 * diff / (Xb.shape[0] * Xb.shape[2])
    grads = backward(P, cache, dout, h_last)
    return loss, grads


def gradcheck():
    rng = np.random.default_rng(0)
    D, Hh, Ww, B = 4, 3, 3, 2
    P = init_params(D, Hh, rng)
    Xb = rng.standard_normal((B, Ww, D))
    Yb = rng.standard_normal((B, D))
    _, grads = mse_and_grad(P, Xb, Yb)
    eps = 1e-6
    worst = 0.0
    for k in P:
        flat = P[k].ravel()
        gflat = grads[k].ravel()
        for j in range(flat.size):
            o = flat[j]
            flat[j] = o + eps
            lp, _ = mse_and_grad(P, Xb, Yb)
            flat[j] = o - eps
            lm, _ = mse_and_grad(P, Xb, Yb)
            flat[j] = o
            fd = (lp - lm) / (2 * eps)
            denom = max(abs(fd) + abs(gflat[j]), 1e-9)
            worst = max(worst, abs(fd - gflat[j]) / denom)
    return worst


def load_split(split):
    z = pd.read_csv(FDIA / f"{split}_z.csv")
    lab = pd.read_csv(FDIA / f"{split}_labels.csv")
    cols = [c for c in z.columns if c != "timestamp"]
    return z[cols].to_numpy(dtype=float), lab, cols


def windows_of(Xstd, idx):
    return np.stack([Xstd[i - W:i] for i in idx], axis=0)


def predict_all(P, Xstd, batch=2048):
    """Predicted standardized next-step for t = W..N-1 (rows 0..W-1 left
    as NaN -- no history)."""
    N, D = Xstd.shape
    pred = np.full((N, D), np.nan)
    ts = np.arange(W, N)
    for b in range(0, len(ts), batch):
        idx = ts[b:b + batch]
        out, _, _ = forward(P, windows_of(Xstd, idx))
        pred[idx] = out
    return pred


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main():
    wc = gradcheck()
    print(f"gradient check worst relative error: {wc:.2e}", flush=True)
    assert wc < 1e-5, "gradient check failed"
    if "--gradcheck-only" in sys.argv:
        return

    rng = np.random.default_rng(SEED)
    Xtr, ltr, cols = load_split("train")
    Xva, lva, _ = load_split("val")
    Xte, lte, _ = load_split("test")
    D = Xtr.shape[1]
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + SIGMA_FLOOR
    Str, Sva, Ste = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    P = init_params(D, H, rng)
    nparams = sum(v.size for v in P.values())
    adam = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in P}
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr_idx = np.arange(W, len(Str))
    va_idx = np.arange(W, len(Sva))
    # clean-val prediction-loss indices (attacked==0 only)
    va_clean = va_idx[(lva["attacked"].to_numpy()[va_idx] == 0)]

    def val_loss(P):
        tot = 0.0
        nb = 0
        for b in range(0, len(va_clean), 4096):
            idx = va_clean[b:b + 4096]
            out, _, _ = forward(P, windows_of(Sva, idx))
            tot += np.sum((out - Sva[idx]) ** 2)
            nb += idx.size * D
        return tot / nb

    print(f"GRU 1-layer, hidden {H}, W {W}, params {nparams}; train windows {len(tr_idx)}", flush=True)
    best = (np.inf, None)
    history = []
    step = 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(tr_idx)
        tl = 0.0
        nb = 0
        for b in range(0, len(tr_idx), BATCH):
            idx = tr_idx[b:b + BATCH]
            loss, grads = mse_and_grad(P, windows_of(Str, idx), Str[idx])
            tl += loss * idx.size
            nb += idx.size
            step += 1
            for k in P:
                m, v = adam[k]
                m[:] = b1 * m + (1 - b1) * grads[k]
                v[:] = b2 * v + (1 - b2) * grads[k] ** 2
                mhat = m / (1 - b1 ** step)
                vhat = v / (1 - b2 ** step)
                P[k] -= LR * mhat / (np.sqrt(vhat) + eps)
        vl = val_loss(P)
        history.append({"epoch": epoch, "train_mse": tl / nb, "val_mse": vl})
        print(f"  epoch {epoch}: train_mse {tl/nb:.5f}  clean_val_mse {vl:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if vl < best[0] - 1e-6:
            best = (vl, {k: P[k].copy() for k in P})
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best clean_val_mse {best[0]:.5f})", flush=True)
                break
    P = best[1]

    # per-channel residual std on train (destandardized), for scoring
    pred_tr = predict_all(P, Str)
    m = np.arange(W, len(Str))
    resid_tr = Xtr[m] - (pred_tr[m] * sd + mu)
    r_c = resid_tr.std(0) + SIGMA_FLOOR

    def score_split(Xstd, Xraw):
        pred = predict_all(P, Xstd)
        N = len(Xstd)
        s = np.full(N, np.nan)
        arg = np.full(N, -1)
        idx = np.arange(W, N)
        resid = np.abs(Xraw[idx] - (pred[idx] * sd + mu)) / r_c
        s[idx] = resid.max(1)
        arg[idx] = resid.argmax(1)
        return s, arg

    s_va, _ = score_split(Sva, Xva)
    s_te, arg_te = score_split(Ste, Xte)
    clean_va = (lva["attacked"].to_numpy() == 0) & np.isfinite(s_va)
    thr = threshold_99(s_va[clean_va])
    print(f"GRU threshold (99th pct clean val score): {thr:.4f}", flush=True)

    scored_te = set(np.arange(W, len(Ste)).tolist())
    flagged = {int(i): bool(s_te[i] > thr) for i in scored_te}
    res = evaluate_flags(lte, flagged, scored_rows=scored_te, name="GRU")

    # save per-timestep test scores + localization
    pd.DataFrame({"row": np.arange(len(Ste)), "timestamp": pd.read_csv(FDIA / "test_z.csv")["timestamp"],
                  "score": s_te, "argmax_channel": [cols[a] if a >= 0 else "" for a in arg_te],
                  "flagged": [bool(s_te[i] > thr) if np.isfinite(s_te[i]) else False for i in range(len(Ste))]}
                 ).to_csv(OUT / "detect_gru_test_pertimestep.csv", index=False)
    res["window_table"].to_csv(OUT / "detect_gru_windows.csv", index=False)
    res["breakdown"].to_csv(OUT / "detect_gru_breakdown.csv", index=False)
    np.savez(OUT / "detect_gru_weights.npz", mu=mu, sd=sd, r_c=r_c, threshold=thr, **P)
    summ = {k: v for k, v in res.items() if k not in ("window_table", "breakdown")}
    summ["threshold"] = float(thr)
    json.dump(summ, open(OUT / "detect_gru_summary.json", "w"), indent=2)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
    manifest = {"model": "GRU-1layer", "hidden": H, "window": W, "n_params": int(nparams),
                "lr": LR, "batch": BATCH, "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "epochs_run": len(history), "best_clean_val_mse": best[0], "history": history,
                "code_commit": commit, "gradcheck_worst_rel_err": wc,
                "data_hashes": {f"{s}_z.csv": file_hash(FDIA / f"{s}_z.csv") for s in ["train", "val", "test"]},
                "weights_hash": file_hash(OUT / "detect_gru_weights.npz")}
    json.dump(manifest, open(OUT / "detect_gru_manifest.json", "w"), indent=2)

    print("\nGRU @ 1% operating point on test:")
    for k in ["clean_test_FAR", "attacked_timestep_detection", "windows_detected", "windows",
              "window_detection_rate", "median_delay", "clean_60block_FA", "clean_60block_total"]:
        print(f"  {k}: {summ[k]}")
    pd.set_option("display.width", 200)
    print(res["breakdown"].to_string(index=False))


if __name__ == "__main__":
    main()
