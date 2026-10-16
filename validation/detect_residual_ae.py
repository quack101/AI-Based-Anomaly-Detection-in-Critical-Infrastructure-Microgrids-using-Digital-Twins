"""
STEP B-D -- learned detector on the physics RESIDUAL features
(residuals_{split}.npz from detect_residual_log.py). numpy only.

DEVIATION (noted): train residuals are logged at stride 4 (cost), so no
10 consecutive steps exist to form the telemetry AE's W=10 temporal
window while test is every-step. A W=10 temporal AE therefore cannot be
trained consistently across splits. We use a W=1 reconstruction AE (the
same GRU-AE cell and d=32, one-step window) on the 361-dim standardized
residual vector -- stride-invariant and consistent train/val/test.

B  reconstruction AE on standardized residuals; train on clean train,
   early-stop on clean val; gradient-checked.
C  per-channel recon errors -> Mahalanobis-full (mu/Sigma on clean
   train, Ledoit-Wolf); 1% clean-val threshold; fusion LR [bdd, resid
   Mahalanobis]. Test scored once.
D  compare to the telemetry Mahalanobis (11/18 @ 0.88%) and fusions.

Run: python -m validation.detect_residual_ae [--gradcheck-only]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_ae import init_params, loss_and_grad, forward, H, FLOOR
from validation.detect_eval import roc_auc
from validation.detect_mahalanobis import ledoit_wolf, segments

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
D_CODE = 32
LR = 1e-3
BATCH = 512
MAX_EPOCHS = 120
PATIENCE = 10
SEED = 1234


def win1(S, idx):
    return S[idx][:, None, :]


def gradcheck():
    rng = np.random.default_rng(0)
    D, Hh, d = 5, 3, 2
    P = init_params(D, Hh, d, rng)
    Xb = rng.standard_normal((2, 1, D)); Yb = rng.standard_normal((2, D))
    _, g = loss_and_grad(P, Xb, Yb)
    eps, worst = 1e-6, 0.0
    for k in P:
        fl, gf = P[k].ravel(), g[k].ravel()
        for j in range(fl.size):
            o = fl[j]; fl[j] = o + eps; lp, _ = loss_and_grad(P, Xb, Yb)
            fl[j] = o - eps; lm, _ = loss_and_grad(P, Xb, Yb); fl[j] = o
            worst = max(worst, abs((lp - lm) / (2 * eps) - gf[j]) / max(abs((lp - lm) / (2 * eps)) + abs(gf[j]), 1e-9))
    return worst


def predict(P, S, batch=4096):
    out = np.empty_like(S)
    for b in range(0, len(S), batch):
        o, _ = forward(P, win1(S, np.arange(b, min(b + batch, len(S)))))
        out[b:b + len(o)] = o
    return out


def load(split):
    z = np.load(OUT / f"residuals_{split}.npz", allow_pickle=True)
    return z["R"].astype(float), z["attacked"].astype(int), z["row"].astype(int), list(z["ids"])


def main():
    wc = gradcheck()
    print(f"residual-AE (W=1) gradient check: {wc:.2e}", flush=True)
    assert wc < 1e-5
    if "--gradcheck-only" in sys.argv:
        return

    Rtr, atr, rowtr, ids = load("train")
    Rva, ava, rowva, _ = load("val")
    Rte, ate, rowte, _ = load("test")
    D = Rtr.shape[1]
    ctr = (atr == 0)
    mu_s = Rtr[ctr].mean(0); sd_s = Rtr[ctr].std(0) + FLOOR
    Str, Sva, Ste = (Rtr - mu_s) / sd_s, (Rva - mu_s) / sd_s, (Rte - mu_s) / sd_s
    cva = (ava == 0)

    # ---- B: train reconstruction AE
    rng = np.random.default_rng(SEED)
    P = init_params(D, H, D_CODE, rng)
    adam = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in P}
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr = np.flatnonzero(ctr); step = 0; best = (np.inf, None, 0); bad = 0; t0 = time.time()
    nparams = sum(v.size for v in P.values())
    for epoch in range(1, MAX_EPOCHS + 1):
        rng.shuffle(tr); tl = nb = 0
        for b in range(0, len(tr), BATCH):
            idx = tr[b:b + BATCH]
            loss, g = loss_and_grad(P, win1(Str, idx), Str[idx]); tl += loss * len(idx); nb += len(idx); step += 1
            for k in P:
                m, v = adam[k]; m[:] = b1 * m + (1 - b1) * g[k]; v[:] = b2 * v + (1 - b2) * g[k] ** 2
                P[k] -= LR * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        vp = predict(P, Sva[cva]); vl = float(np.mean((vp - Sva[cva]) ** 2))
        if vl < best[0] - 1e-7:
            best = (vl, {k: P[k].copy() for k in P}, epoch); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    P = best[1]
    print(f"[B] residual-AE: d={D_CODE} params {nparams} epochs {best[2]} clean_val_mse {best[0]:.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # per-channel recon errors (standardized units), r_c on train
    def errors(S, Xc):
        return np.abs(S - predict(P, S))
    Etr, Eva, Ete = errors(Str, None), errors(Sva, None), errors(Ste, None)
    r_c = Etr[ctr].std(0) + FLOOR
    Etr, Eva, Ete = Etr / r_c, Eva / r_c, Ete / r_c

    # ---- C: Mahalanobis on clean-train errors
    mu = Etr[ctr].mean(0)
    Sigma, S, rho = ledoit_wolf(Etr[ctr] - mu)
    Minv = np.linalg.inv(Sigma)
    def maha(E):
        d = E - mu
        return np.einsum("ij,ij->i", d @ Minv, d)
    mt = maha(Ete); mv = maha(Eva)
    thr = float(np.quantile(mv[cva], 0.99))
    print(f"[C] Ledoit-Wolf lambda={rho:.4f} cond={np.linalg.cond(Sigma):.2e}", flush=True)

    tlab = pd.read_csv(FDIA / "test_labels.csv")
    # test residual rows are every step -> align by row index
    sc_te = np.full(len(tlab), np.nan); sc_te[rowte] = mt
    res = ev(sc_te, tlab, thr)
    # localization
    cols = ids
    loc_h = loc_t = 0
    for w in segments(tlab)[0]:
        rows = w["rows"][np.isfinite(sc_te[w["rows"]])]
        if not len(rows) or not np.any(sc_te[rows] > thr):
            continue
        loc_t += 1; contrib = np.zeros(D)
        for rr in rows:
            idx_r = np.where(rowte == rr)[0]
            if len(idx_r):
                dv = Ete[idx_r[0]] - mu; contrib += dv * (Minv @ dv)
        loc_h += int(cols[int(np.argmax(contrib))] in set(str(w["targets"]).split(";")))
    res["localization"] = loc_h / loc_t if loc_t else float("nan")
    print(f"[C] residual Mahalanobis: FAR {res['FAR']*100:.2f}% step {res['step_det']*100:.1f}% "
          f"win {res['win_det']}/{res['n_win']} AUC {res['per_ts_AUC']:.4f} winAUC {res['window_AUC']:.4f} "
          f"fam {res['A1']},{res['A2']},{res['A3']} loc {loc_h}/{loc_t}", flush=True)

    # ---- fusion LR [bdd, residual maha] on VAL (rows logged in val)
    vs = pd.read_csv(OUT / "val_scores.csv"); bdd_v_all = vs["bdd_score"].to_numpy()
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    bdd_t = np.full(len(tlab), np.nan); bdd_t[phys["row"].astype(int)] = phys["bdd_score"].astype(float)
    bdd_v = bdd_v_all[rowva]
    fin = np.isfinite(bdd_v) & np.isfinite(mv)
    cvf = fin & (ava == 0)
    muf = np.array([bdd_v[cvf].mean(), mv[cvf].mean()]); sdf = np.array([bdd_v[cvf].std(), mv[cvf].std()]) + 1e-12
    Xv = np.column_stack([(bdd_v[fin] - muf[0]) / sdf[0], (mv[fin] - muf[1]) / sdf[1]])
    w, b = logistic(Xv, ava[fin].astype(float))
    auc_lr = roc_auc(1 / (1 + np.exp(-(Xv @ w + b))), ava[fin])
    def fus(bd, m): return 1 / (1 + np.exp(-(((bd - muf[0]) / sdf[0]) * w[0] + ((m - muf[1]) / sdf[1]) * w[1] + b)))
    fv = np.full(len(vs), np.nan); fv[rowva[fin]] = fus(bdd_v[fin], mv[fin])
    ft = fus(bdd_t, sc_te)
    thr_f = float(np.quantile(fv[np.isfinite(fv) & (vs["attacked"].to_numpy() == 0)], 0.99))
    fres = ev(ft, tlab, thr_f)
    print(f"[C] fusion BDD+residMaha: val AUC {auc_lr:.4f} w bdd {w[0]:.3f} resid {w[1]:.3f} -> FAR "
          f"{fres['FAR']*100:.2f}% win {fres['win_det']}/{fres['n_win']} AUC {fres['per_ts_AUC']:.4f} "
          f"fam {fres['A1']},{fres['A2']},{fres['A3']}", flush=True)

    tel = json.load(open(OUT / "detect_final_summary.json"))
    table = [
        {"detector": "telemetry Mahalanobis (point)", "FAR": 0.0088, "window": "11/18", "A1": "9/12", "A2": "0/3",
         "A3": "2/3", "per_ts_AUC": 0.874, "window_AUC": 0.770, "localization": 0.727},
        {"detector": "telemetry fusion BDD+Maha", "FAR": 0.0073, "window": "12/18", "A1": "9/12", "A2": "1/3",
         "A3": "2/3", "per_ts_AUC": 0.873, "window_AUC": 0.811, "localization": None},
        {"detector": "residual-AE Mahalanobis", "FAR": res["FAR"], "window": f"{res['win_det']}/{res['n_win']}",
         "A1": res["A1"], "A2": res["A2"], "A3": res["A3"], "per_ts_AUC": res["per_ts_AUC"],
         "window_AUC": res["window_AUC"], "localization": res["localization"]},
        {"detector": "fusion BDD+residual-Maha", "FAR": fres["FAR"], "window": f"{fres['win_det']}/{fres['n_win']}",
         "A1": fres["A1"], "A2": fres["A2"], "A3": fres["A3"], "per_ts_AUC": fres["per_ts_AUC"],
         "window_AUC": fres["window_AUC"], "localization": None},
    ]
    pd.DataFrame(table).to_csv(OUT / "detect_residual_table.csv", index=False)
    json.dump({"residual_maha": res, "fusion": fres, "fusion_val_AUC": auc_lr, "lw_lambda": rho,
               "ae_params": nparams, "ae_val_mse": best[0], "gradcheck": wc},
              open(OUT / "detect_residual_summary.json", "w"), indent=2, default=lambda o: None)
    np.savez(OUT / "detect_residual_ae_weights.npz", mu_s=mu_s, sd_s=sd_s, r_c=r_c, mu=mu, Sigma=Sigma, **P)
    pd.set_option("display.width", 230)
    print("\n" + pd.DataFrame(table).to_string(index=False))


def logistic(X, y, iters=4000, lr=0.5, l2=1e-4):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); w -= lr * (X.T @ (p - y) / len(y) + l2 * w); b -= lr * np.mean(p - y)
    return w, b


def _f(fam, t):
    a, n = fam.get(t, [0, 0]); return f"{a}/{n}"


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
    return {"FAR": far, "step_det": step, "win_det": int(sum(det.values())), "n_win": len(wins),
            "A1": _f(fam, "A1"), "A2": _f(fam, "A2"), "A3": _f(fam, "A3"),
            "per_ts_AUC": float(roc_auc(score[fin], att[fin].astype(bool))),
            "window_AUC": float(roc_auc(np.concatenate([pos, neg]),
                                        np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])))}


if __name__ == "__main__":
    main()
