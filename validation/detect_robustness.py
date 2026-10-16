"""
Final hardening (numpy only; no estimator runs).

A  FAIR W=1 comparison: a W=1 telemetry reconstruction AE trained on the
   SAME strided-clean train indices the residual-AE used (residuals_
   train.npz 'row'), vs a W=1 residual AE on the residual vectors. Both
   d=32, same cell, same seed, same steps. Mahalanobis-scored; ~1% FAR.
B  reseed robustness: the best detector's AE (telemetry, W=10, d=32,
   v3 40-epoch protocol) retrained with 5 seeds; per seed recompute the
   Mahalanobis-point score and the BDD+Mahalanobis fusion. Report
   mean+/-std of per-ts AUC, window AUC, window detection, realized FAR,
   A2 count.

Run: python -m validation.detect_robustness
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_ae import init_params, loss_and_grad, forward, windows_of, W, H, FLOOR
from validation.detect_eval import roc_auc
from validation.detect_mahalanobis import ledoit_wolf, segments

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
LR = 1e-3
BATCH = 512
SEEDS = [1234, 1, 2, 3, 4]


def win1(S, idx):
    return S[idx][:, None, :]


def load_tel(split):
    z = pd.read_csv(FDIA / f"{split}_z.csv").drop(columns="timestamp").to_numpy(float)
    lab = pd.read_csv(FDIA / f"{split}_labels.csv")
    return z, lab["attacked"].to_numpy().astype(int)


def train_ae(Str, train_idx, val_clean_idx, Sva, d, seed, window, max_epochs, patience):
    rng = np.random.default_rng(seed)
    D = Str.shape[1]
    P = init_params(D, H, d, rng)
    adam = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in P}
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr = np.array(train_idx); step = 0; best = (np.inf, None, 0); bad = 0

    def wins(S, idx):
        return win1(S, idx) if window == 1 else windows_of(S, idx)

    for epoch in range(1, max_epochs + 1):
        rng.shuffle(tr);
        for b in range(0, len(tr), BATCH):
            idx = tr[b:b + BATCH]
            _, g = loss_and_grad(P, wins(Str, idx), Str[idx]); step += 1
            for k in P:
                m, v = adam[k]; m[:] = b1 * m + (1 - b1) * g[k]; v[:] = b2 * v + (1 - b2) * g[k] ** 2
                P[k] -= LR * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        vt = vn = 0.0
        for b in range(0, len(val_clean_idx), 4096):
            idx = val_clean_idx[b:b + 4096]
            out, _ = forward(P, wins(Sva, idx)); vt += np.sum((out - Sva[idx]) ** 2); vn += idx.size * Str.shape[1]
        vl = vt / vn
        if vl < best[0] - 1e-7:
            best = (vl, {k: P[k].copy() for k in P}, epoch); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1], best[0], best[2]


def recon_err(P, S, window, batch=2048):
    N, D = S.shape
    E = np.full((N, D), np.nan)
    idx = np.arange(0, N) if window == 1 else np.arange(W - 1, N)
    for b in range(0, len(idx), batch):
        ii = idx[b:b + batch]
        out, _ = forward(P, (win1(S, ii) if window == 1 else windows_of(S, ii)))
        E[ii] = np.abs(S[ii] - out)
    return E


def maha_scores(Etr_clean, Eva, Ete, r_c):
    Etr = Etr_clean / r_c
    mu = Etr.mean(0)
    Sigma, _, rho = ledoit_wolf(Etr - mu)
    Minv = np.linalg.inv(Sigma)
    def sc(E):
        Z = E / r_c
        s = np.full(len(E), np.nan); fin = np.isfinite(E[:, 0]); d = Z[fin] - mu
        s[fin] = np.einsum("ij,ij->i", d @ Minv, d); return s
    return sc(Eva), sc(Ete), rho


def ev(score, lab, thr):
    att = lab["attacked"].to_numpy(); fin = np.isfinite(score)
    far = float(np.mean(score[fin & (att == 0)] > thr)); step = float(np.mean(score[fin & (att == 1)] > thr))
    wins, blocks = segments(lab); det = {}; fam = {}
    for w in wins:
        rows = w["rows"][np.isfinite(score[w["rows"]])]
        dd = bool(np.any(score[rows] > thr)) if len(rows) else False
        det[w["wid"]] = dd; fam.setdefault(w["type"], [0, 0]); fam[w["type"]][1] += 1; fam[w["type"]][0] += int(dd)
    def smax(rows):
        v = score[rows]; v = v[np.isfinite(v)]; return float(v.max()) if len(v) else np.nan
    pos = np.array([smax(w["rows"]) for w in wins]); neg = np.array([smax(b) for b in blocks])
    fam_ct = {t: a for t, (a, n) in fam.items()}
    return {"FAR": far, "step_det": step, "win": int(sum(det.values())), "n": len(wins),
            "A1": fam_ct.get("A1", 0), "A2": fam_ct.get("A2", 0), "A3": fam_ct.get("A3", 0),
            "per_ts_AUC": float(roc_auc(score[fin], att[fin].astype(bool))),
            "window_AUC": float(roc_auc(np.concatenate([pos, neg]),
                                        np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])))}


def localization(Ete, mu, Minv, r_c, lab, score, thr, cols):
    wins, _ = segments(lab); hits = tot = 0
    for w in wins:
        rows = w["rows"][np.isfinite(score[w["rows"]])]
        if not len(rows) or not np.any(score[rows] > thr):
            continue
        tot += 1; contrib = np.zeros(Ete.shape[1])
        for rr in rows:
            d = Ete[rr] / r_c - mu; contrib += np.where(np.isfinite(d), d * (Minv @ d), 0.0)
        hits += int(cols[int(np.argmax(contrib))] in set(str(w["targets"]).split(";")))
    return hits / tot if tot else float("nan")


def logistic(X, y, iters=4000, lr=0.5, l2=1e-4):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); w -= lr * (X.T @ (p - y) / len(y) + l2 * w); b -= lr * np.mean(p - y)
    return w, b


def main():
    tlab = pd.read_csv(FDIA / "test_labels.csv"); vlab = pd.read_csv(FDIA / "val_labels.csv")
    cols_tel = [c for c in pd.read_csv(FDIA / "test_z.csv", nrows=0).columns if c != "timestamp"]

    # ===== STEP A: fair W=1 =====
    rtr = np.load(OUT / "residuals_train.npz", allow_pickle=True)
    strided_idx = rtr["row"].astype(int)                     # the 15120 strided clean train indices
    # telemetry W=1 on those indices
    Ztr, atr = load_tel("train"); Zva, ava = load_tel("val"); Zte, ate = load_tel("test")
    mu_t = Ztr[strided_idx].mean(0); sd_t = Ztr[strided_idx].std(0) + FLOOR
    Str, Sva, Ste = (Ztr - mu_t) / sd_t, (Zva - mu_t) / sd_t, (Zte - mu_t) / sd_t
    va_clean = np.flatnonzero(ava == 0)
    Pt, vlt, ept = train_ae(Str, strided_idx, va_clean, Sva, 32, 1234, 1, 120, 10)
    Etr = recon_err(Pt, Str, 1); Eva = recon_err(Pt, Sva, 1); Ete = recon_err(Pt, Ste, 1)
    r_c = Etr[strided_idx].std(0) + FLOOR
    mv, mt, _ = maha_scores(Etr[strided_idx], Eva, Ete, r_c)
    thr = float(np.quantile(mv[ava == 0], 0.99))
    tel_w1 = ev(mt, tlab, thr)
    mu_m = (Etr[strided_idx] / r_c).mean(0); Sig, _, _ = ledoit_wolf(Etr[strided_idx] / r_c - mu_m)
    tel_w1["loc"] = localization(Ete, mu_m, np.linalg.inv(Sig), r_c, tlab, mt, thr, cols_tel)
    print(f"[A] telemetry W=1: win {tel_w1['win']}/18 A1 {tel_w1['A1']} A2 {tel_w1['A2']} A3 {tel_w1['A3']} "
          f"AUC {tel_w1['per_ts_AUC']:.4f} winAUC {tel_w1['window_AUC']:.4f} FAR {tel_w1['FAR']*100:.2f}% "
          f"loc {tel_w1['loc']}", flush=True)

    # residual W=1 on the same steps
    Rtr = rtr["R"].astype(float); rva = np.load(OUT / "residuals_val.npz", allow_pickle=True)
    rte = np.load(OUT / "residuals_test.npz", allow_pickle=True)
    Rva, avr = rva["R"].astype(float), rva["attacked"].astype(int)
    Rte = rte["R"].astype(float)
    cols_res = list(rtr["ids"])
    mu_r = Rtr.mean(0); sd_r = Rtr.std(0) + FLOOR
    RStr, RSva, RSte = (Rtr - mu_r) / sd_r, (Rva - mu_r) / sd_r, (Rte - mu_r) / sd_r
    rva_clean = np.flatnonzero(avr == 0)
    Pr, vlr, epr = train_ae(RStr, np.arange(len(RStr)), rva_clean, RSva, 32, 1234, 1, 120, 10)
    REtr = recon_err(Pr, RStr, 1); REva = recon_err(Pr, RSva, 1); REte = recon_err(Pr, RSte, 1)
    rrc = REtr.std(0) + FLOOR
    rmv, rmt, _ = maha_scores(REtr, REva, REte, rrc)
    rthr = float(np.quantile(rmv[avr == 0], 0.99))
    # residual test rows are every test step -> align
    sc_te = np.full(len(tlab), np.nan); sc_te[rte["row"].astype(int)] = rmt
    res_w1 = ev(sc_te, tlab, rthr)
    rmu_m = (REtr / rrc).mean(0); rSig, _, _ = ledoit_wolf(REtr / rrc - rmu_m)
    # localization aligned
    REte_full = np.full((len(tlab), Rte.shape[1]), np.nan); REte_full[rte["row"].astype(int)] = REte
    res_w1["loc"] = localization(REte_full, rmu_m, np.linalg.inv(rSig), rrc, tlab, sc_te, rthr, cols_res)
    print(f"[A] residual  W=1: win {res_w1['win']}/18 A1 {res_w1['A1']} A2 {res_w1['A2']} A3 {res_w1['A3']} "
          f"AUC {res_w1['per_ts_AUC']:.4f} winAUC {res_w1['window_AUC']:.4f} FAR {res_w1['FAR']*100:.2f}% "
          f"loc {res_w1['loc']}", flush=True)

    # ===== STEP B: reseed telemetry W=10 d=32 =====
    mu_f = Ztr.mean(0); sd_f = Ztr.std(0) + FLOOR
    Strf, Svaf, Stef = (Ztr - mu_f) / sd_f, (Zva - mu_f) / sd_f, (Zte - mu_f) / sd_f
    tr_all = np.arange(W - 1, len(Strf))
    va_clean10 = tr_all_val = np.arange(W - 1, len(Svaf))
    va_clean10 = va_clean10[ava[va_clean10] == 0]
    vs = pd.read_csv(OUT / "val_scores.csv"); bdd_v = vs["bdd_score"].to_numpy()
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    bdd_t = np.full(len(tlab), np.nan); bdd_t[phys["row"].astype(int)] = phys["bdd_score"].astype(float)

    maha_rows, fus_rows = [], []
    for seed in SEEDS:
        t0 = time.time()
        P, vl, ep = train_ae(Strf, tr_all, va_clean10, Svaf, 32, seed, W, 40, 40)
        Etr = recon_err(P, Strf, W); Eva = recon_err(P, Svaf, W); Ete = recon_err(P, Stef, W)
        clean_tr = (atr == 0) & np.isfinite(Etr[:, 0])
        r_c = Etr[clean_tr].std(0) + FLOOR
        mv, mt, rho = maha_scores(Etr[clean_tr], Eva, Ete, r_c)
        thr = float(np.quantile(mv[(ava == 0) & np.isfinite(mv)], 0.99))
        m = ev(mt, tlab, thr); m["seed"] = seed
        maha_rows.append(m)
        # fusion
        both = np.isfinite(bdd_v) & np.isfinite(mv); cvf = both & (ava == 0)
        muf = np.array([bdd_v[cvf].mean(), mv[cvf].mean()]); sdf = np.array([bdd_v[cvf].std(), mv[cvf].std()]) + 1e-12
        Xv = np.column_stack([(bdd_v[both] - muf[0]) / sdf[0], (mv[both] - muf[1]) / sdf[1]])
        w, b = logistic(Xv, ava[both].astype(float))
        def fus(bd, mm): return 1 / (1 + np.exp(-(((bd - muf[0]) / sdf[0]) * w[0] + ((mm - muf[1]) / sdf[1]) * w[1] + b)))
        fv = fus(bdd_v, mv); ftt = fus(bdd_t, mt)
        thrf = float(np.quantile(fv[(ava == 0) & np.isfinite(fv)], 0.99))
        f = ev(ftt, tlab, thrf); f["seed"] = seed
        fus_rows.append(f)
        print(f"[B] seed {seed}: maha win {m['win']}/18 AUC {m['per_ts_AUC']:.3f} A2 {m['A2']} FAR {m['FAR']*100:.2f}% "
              f"| fusion win {f['win']}/18 AUC {f['per_ts_AUC']:.3f} A2 {f['A2']} ({time.time()-t0:.0f}s)", flush=True)

    def agg(rows, label):
        keys = ["per_ts_AUC", "window_AUC", "win", "FAR", "A2"]
        return {label: {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in keys}}

    summary = {"stepA": {"telemetry_W1": tel_w1, "residual_W1": res_w1},
               **agg(maha_rows, "maha_point"), **agg(fus_rows, "fusion"),
               "maha_seeds": maha_rows, "fusion_seeds": fus_rows}
    json.dump(summary, open(OUT / "detect_robustness_summary.json", "w"), indent=2, default=float)
    pd.DataFrame(maha_rows).to_csv(OUT / "detect_robustness_maha.csv", index=False)
    pd.DataFrame(fus_rows).to_csv(OUT / "detect_robustness_fusion.csv", index=False)

    print("\n[B] reseed mean+/-std (5 seeds):")
    for label, rows in [("AE-Maha-point", maha_rows), ("fusion BDD+Maha", fus_rows)]:
        a = agg(rows, label)[label]
        print(f"  {label}: per-ts AUC {a['per_ts_AUC'][0]:.4f}+/-{a['per_ts_AUC'][1]:.4f}  "
              f"winAUC {a['window_AUC'][0]:.4f}+/-{a['window_AUC'][1]:.4f}  "
              f"win {a['win'][0]:.1f}+/-{a['win'][1]:.1f}/18  FAR {a['FAR'][0]*100:.2f}+/-{a['FAR'][1]*100:.2f}%  "
              f"A2 {a['A2'][0]:.1f}+/-{a['A2'][1]:.1f}")


if __name__ == "__main__":
    main()
