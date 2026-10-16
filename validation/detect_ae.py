"""
Part B -- GRU reconstruction autoencoder anomaly detector (pure numpy;
no ML framework, to keep the frozen estimator's numpy untouched).

A GRU encoder reads a window of W=10 standardized steps (the window
ENDS at the current step t, i.e. steps t-W+1..t) and produces its final
hidden state; a linear+tanh bottleneck of dim d compresses it to a code;
a linear decoder reconstructs the CURRENT step z_t (all 168 channels).
Because every step is reconstructed from its own window, a sustained
anomaly keeps producing a reconstruction error (unlike the next-step
predictor, whose error decays once it adapts).

Train on CLEAN TRAIN only (next-step... reconstruction MSE), Adam, fixed
seed, early-stop on CLEAN VAL reconstruction loss. d is chosen from
{8,16,32} by clean-val reconstruction loss. Score per timestep:
  s_max  = max over channels of |z_t - zhat_t| / r_c
  s_mean = mean over channels of the same
r_c = per-channel residual std on the train split. The score (max or
mean) with the higher clean-val-vs-attacked-val AUC is used. Threshold =
99th percentile over clean val (1% operating point); applied to test.

Run: python -m validation.detect_ae [--gradcheck-only]
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import evaluate_flags, threshold_99, roc_auc

BASE_DIR = Path(__file__).resolve().parent.parent
FDIA = BASE_DIR / "dataset" / "fdia_v1"
OUT = BASE_DIR / "validation"
W = 10
H = 96
D_CHOICES = [8, 16, 32]
LR = 1e-3
BATCH = 512
MAX_EPOCHS = 40
PATIENCE = 5
SEED = 1234
FLOOR = 1e-8


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(D, H, d, rng):
    def g(a, b):
        return rng.standard_normal((a, b)) * (1.0 / np.sqrt(b))
    P = {"Wz": g(D, H), "Wr": g(D, H), "Wn": g(D, H),
         "Uz": g(H, H), "Ur": g(H, H), "Un": g(H, H),
         "bz": np.zeros(H), "br": np.zeros(H), "bn": np.zeros(H),
         "Wc": g(H, d), "bc": np.zeros(d), "Wd": g(d, D), "bd": np.zeros(D)}
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
    code = np.tanh(h @ P["Wc"] + P["bc"])
    out = code @ P["Wd"] + P["bd"]
    return out, (cache, h, code)


def backward(P, state, dout):
    cache, h_last, code = state
    grads = {k: np.zeros_like(P[k]) for k in P}
    grads["Wd"] = code.T @ dout
    grads["bd"] = dout.sum(0)
    dcode = dout @ P["Wd"].T
    dcpre = dcode * (1 - code ** 2)
    grads["Wc"] = h_last.T @ dcpre
    grads["bc"] = dcpre.sum(0)
    dh = dcpre @ P["Wc"].T
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


def loss_and_grad(P, Xb, Yb):
    out, state = forward(P, Xb)
    diff = out - Yb
    loss = float(np.mean(diff ** 2))
    dout = 2.0 * diff / (Xb.shape[0] * Xb.shape[2])
    return loss, backward(P, state, dout)


def gradcheck():
    rng = np.random.default_rng(0)
    D, Hh, d, Ww, B = 4, 3, 2, 3, 2
    P = init_params(D, Hh, d, rng)
    Xb = rng.standard_normal((B, Ww, D))
    Yb = rng.standard_normal((B, D))
    _, grads = loss_and_grad(P, Xb, Yb)
    eps, worst = 1e-6, 0.0
    for k in P:
        flat, gflat = P[k].ravel(), grads[k].ravel()
        for j in range(flat.size):
            o = flat[j]
            flat[j] = o + eps; lp, _ = loss_and_grad(P, Xb, Yb)
            flat[j] = o - eps; lm, _ = loss_and_grad(P, Xb, Yb)
            flat[j] = o
            fd = (lp - lm) / (2 * eps)
            worst = max(worst, abs(fd - gflat[j]) / max(abs(fd) + abs(gflat[j]), 1e-9))
    return worst


def load_split(split):
    z = pd.read_csv(FDIA / f"{split}_z.csv")
    lab = pd.read_csv(FDIA / f"{split}_labels.csv")
    cols = [c for c in z.columns if c != "timestamp"]
    return z[cols].to_numpy(dtype=float), lab, cols


def windows_of(S, idx):
    return np.stack([S[i - W + 1:i + 1] for i in idx], axis=0)


def predict_all(P, S, batch=2048):
    N, D = S.shape
    pred = np.full((N, D), np.nan)
    ts = np.arange(W - 1, N)
    for b in range(0, len(ts), batch):
        idx = ts[b:b + batch]
        out, _ = forward(P, windows_of(S, idx))
        pred[idx] = out
    return pred


def train_one(d, Str, Sva, va_clean_idx, rng):
    D = Str.shape[1]
    P = init_params(D, H, d, rng)
    adam = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in P}
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr_idx = np.arange(W - 1, len(Str))
    step = 0
    best = (np.inf, None)
    bad = 0
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
        # clean-val reconstruction loss
        vt = vn = 0.0
        for b in range(0, len(va_clean_idx), 4096):
            idx = va_clean_idx[b:b + 4096]
            out, _ = forward(P, windows_of(Sva, idx))
            vt += np.sum((out - Sva[idx]) ** 2); vn += idx.size * D
        vl = vt / vn
        hist.append({"epoch": epoch, "train_mse": tl / nb, "clean_val_mse": vl})
        if vl < best[0] - 1e-6:
            best = (vl, {k: P[k].copy() for k in P}); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best[0], best[1], hist, time.time() - t0, epoch


def main():
    wc = gradcheck()
    print(f"AE gradient check worst relative error: {wc:.2e}", flush=True)
    assert wc < 1e-5, "gradient check failed"
    if "--gradcheck-only" in sys.argv:
        return

    Xtr, ltr, cols = load_split("train")
    Xva, lva, _ = load_split("val")
    Xte, lte, _ = load_split("test")
    D = Xtr.shape[1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + FLOOR
    Str, Sva, Ste = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    va_idx = np.arange(W - 1, len(Sva))
    va_att = lva["attacked"].to_numpy()
    va_clean_idx = va_idx[va_att[va_idx] == 0]

    results = {}
    for d in D_CHOICES:
        vl, P, hist, secs, ep = train_one(d, Str, Sva, va_clean_idx, np.random.default_rng(SEED))
        nparams = sum(v.size for v in P.values())
        print(f"d={d}: clean_val_mse {vl:.5f}  epochs {ep}  params {nparams}  {secs:.0f}s", flush=True)
        results[d] = {"val_mse": vl, "P": P, "hist": hist, "params": nparams, "epochs": ep}
    d_best = min(results, key=lambda k: results[k]["val_mse"])
    P = results[d_best]["P"]
    print(f"chosen d={d_best} (lowest clean-val reconstruction MSE)", flush=True)

    # per-channel residual std on train
    pred_tr = predict_all(P, Str)
    m = np.arange(W - 1, len(Str))
    r_c = (Xtr[m] - (pred_tr[m] * sd + mu)).std(0) + FLOOR

    def score_split(S, Xraw):
        pred = predict_all(P, S)
        N = len(S)
        idx = np.arange(W - 1, N)
        resid = np.abs(Xraw[idx] - (pred[idx] * sd + mu)) / r_c
        smax = np.full(N, np.nan); smean = np.full(N, np.nan); arg = np.full(N, -1)
        smax[idx] = resid.max(1); smean[idx] = resid.mean(1); arg[idx] = resid.argmax(1)
        return smax, smean, arg

    va_max, va_mean, _ = score_split(Sva, Xva)
    te_max, te_mean, te_arg = score_split(Ste, Xte)
    y_va = va_att[va_idx].astype(bool)
    auc_max = roc_auc(va_max[va_idx], y_va)
    auc_mean = roc_auc(va_mean[va_idx], y_va)
    chosen = "s_max" if auc_max >= auc_mean else "s_mean"
    print(f"val AUC: s_max {auc_max:.4f}, s_mean {auc_mean:.4f} -> using {chosen}", flush=True)
    s_va = va_max if chosen == "s_max" else va_mean
    s_te = te_max if chosen == "s_max" else te_mean

    thr = threshold_99(s_va[va_clean_idx])
    print(f"AE threshold (99th pct clean val {chosen}): {thr:.4f}", flush=True)
    scored = set(np.arange(W - 1, len(Ste)).tolist())
    flagged = {int(i): bool(s_te[i] > thr) for i in scored}
    res = evaluate_flags(lte, flagged, scored_rows=scored, name=f"AE_d{d_best}_{chosen}")
    y_te = lte["attacked"].to_numpy()[np.arange(W - 1, len(Ste))].astype(bool)
    auc_te = roc_auc(s_te[np.arange(W - 1, len(Ste))], y_te)

    # sustained-window check: a 240-step step-profile attacked window
    sust = lte[(lte["attacked"] == 1) & (lte["duration"] == 240) & (lte["profile"] == "step")]
    sustained = {}
    if len(sust):
        wid = sust["window_id"].iloc[0]
        g = lte[(lte["window_id"] == wid) & (lte["attacked"] == 1)]
        rows = [r for r in g.index if r in scored]
        fl = np.array([s_te[r] > thr for r in rows])
        k = max(1, len(rows) // 5)
        sustained = {"window_id": int(wid), "n_steps": len(rows), "flagged_frac": float(fl.mean()),
                     "onset_block_rate": float(fl[:k].mean()), "mid_block_rate": float(fl[len(rows)//2 - k//2: len(rows)//2 + k//2 + 1].mean()),
                     "end_block_rate": float(fl[-k:].mean())}
        print(f"sustained 240-step window {wid}: flagged {sustained['flagged_frac']*100:.1f}% "
              f"(onset {sustained['onset_block_rate']*100:.0f}%, mid {sustained['mid_block_rate']*100:.0f}%, "
              f"end {sustained['end_block_rate']*100:.0f}%)", flush=True)

    pd.DataFrame({"row": np.arange(len(Ste)), "timestamp": pd.read_csv(FDIA / "test_z.csv")["timestamp"],
                  "s_max": te_max, "s_mean": te_mean, "score": s_te,
                  "argmax_channel": [cols[a] if a >= 0 else "" for a in te_arg],
                  "flagged": [bool(s_te[i] > thr) if np.isfinite(s_te[i]) else False for i in range(len(Ste))]}
                 ).to_csv(OUT / "detect_ae_test_pertimestep.csv", index=False)
    # val scores saved too (for Part C/D)
    pd.DataFrame({"row": np.arange(len(Sva)), "s_max": va_max, "s_mean": va_mean, "score": s_va,
                  "attacked": va_att}).to_csv(OUT / "detect_ae_val_pertimestep.csv", index=False)
    res["window_table"].to_csv(OUT / "detect_ae_windows.csv", index=False)
    res["breakdown"].to_csv(OUT / "detect_ae_breakdown.csv", index=False)
    np.savez(OUT / "detect_ae_weights.npz", mu=mu, sd=sd, r_c=r_c, threshold=thr, d=d_best,
             chosen_score=chosen, **P)

    summ = {k: v for k, v in res.items() if k not in ("window_table", "breakdown")}
    summ.update({"d": int(d_best), "chosen_score": chosen, "threshold": float(thr),
                 "per_timestep_AUC": float(auc_te), "val_AUC_s_max": float(auc_max),
                 "val_AUC_s_mean": float(auc_mean), "sustained": sustained})
    json.dump(summ, open(OUT / "detect_ae_summary.json", "w"), indent=2)

    def fh(p):
        h = hashlib.sha256()
        h.update(Path(p).read_bytes()); return h.hexdigest().upper()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
    json.dump({"model": "GRU-AE", "hidden": H, "window": W, "d_choices": D_CHOICES, "d_chosen": int(d_best),
               "params": results[d_best]["params"], "lr": LR, "batch": BATCH, "seed": SEED,
               "epochs_run": results[d_best]["epochs"], "chosen_score": chosen,
               "val_mse_by_d": {k: results[k]["val_mse"] for k in results},
               "gradcheck_worst_rel_err": wc, "code_commit": commit,
               "data_hashes": {f"{s}_z.csv": fh(FDIA / f"{s}_z.csv") for s in ["train", "val", "test"]},
               "weights_hash": fh(OUT / "detect_ae_weights.npz"),
               "history": results[d_best]["hist"]}, open(OUT / "detect_ae_manifest.json", "w"), indent=2)

    print(f"\nAE @ 1% operating point on test (d={d_best}, {chosen}), per-timestep AUC {auc_te:.4f}:")
    for k in ["clean_test_FAR", "attacked_timestep_detection", "windows_detected", "windows",
              "window_detection_rate", "median_delay", "clean_60block_FA", "clean_60block_total"]:
        print(f"  {k}: {summ[k]}")
    pd.set_option("display.width", 200)
    print(res["breakdown"].to_string(index=False))


if __name__ == "__main__":
    main()
