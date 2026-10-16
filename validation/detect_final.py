"""
Finalize: window-smooth the AE Mahalanobis-full score and refit fusion
with it. numpy only; no estimator runs (v3 AE per-channel errors
recomputed from saved weights; Mahalanobis fit recomputed on clean
train, deterministic). Thresholds/fusion weights on VAL only; test
scored once per variant.

A  trailing-mean (L in {15,60}, picked by val window-AUC) and CUSUM of
   the Mahalanobis score; 1% clean-val threshold; test metrics + delay,
   vs the point (unsmoothed) Mahalanobis.
B  logistic-regression fusion on VAL with [bdd_score, mahalanobis_score].
C  final table + reports/detector_final.md.

Run: python -m validation.detect_final
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import roc_auc
from validation.detect_mahalanobis import ae_errors, ledoit_wolf, segments, PARAM_KEYS

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
L_CHOICES = [15, 60]
KS = [0.25, 0.5, 1.0]
L_FAR = 60


def trailing_mean(s, L):
    n = len(s); out = np.full(n, np.nan)
    fin = np.isfinite(s); sf = np.where(fin, s, 0.0)
    cs = np.concatenate([[0], np.cumsum(sf)]); cf = np.concatenate([[0], np.cumsum(fin)])
    for t in range(L - 1, n):
        if cf[t + 1] - cf[t + 1 - L] == L:
            out[t] = (cs[t + 1] - cs[t + 1 - L]) / L
    return out


def cusum1(x, k):
    g = 0.0; out = np.empty(len(x))
    for t, v in enumerate(x):
        g = max(0.0, g + (v - k if np.isfinite(v) else -k)); out[t] = g
    return out


def seg_max(score, rows):
    v = score[rows]; v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else np.nan


def win_auc(score, lab):
    wins, blocks = segments(lab)
    pos = np.array([seg_max(score, w["rows"]) for w in wins])
    neg = np.array([seg_max(score, b) for b in blocks])
    return roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))


def eval_point(score, lab, thr):
    att = lab["attacked"].to_numpy()
    fin = np.isfinite(score)
    far = float(np.mean(score[fin & (att == 0)] > thr))
    step = float(np.mean(score[fin & (att == 1)] > thr))
    wins, _ = segments(lab)
    det, fam, delays = {}, {}, []
    for w in wins:
        rows = w["rows"][np.isfinite(score[w["rows"]])]
        crossed = rows[score[rows] > thr] if len(rows) else []
        dd = len(crossed) > 0
        det[w["wid"]] = dd
        if dd:
            delays.append(int(crossed.min() - w["rows"].min()))
        fam.setdefault(w["type"], [0, 0]); fam[w["type"]][1] += 1; fam[w["type"]][0] += int(dd)
    return {"FAR": far, "step_det": step, "win_det": int(sum(det.values())), "n_win": len(wins),
            "A1": _f(fam, "A1"), "A2": _f(fam, "A2"), "A3": _f(fam, "A3"),
            "per_ts_AUC": float(roc_auc(score[fin], att[fin].astype(bool))),
            "window_AUC": float(win_auc(score, lab)),
            "median_delay": float(np.median(delays)) if delays else float("nan"), "det": det}


def _f(fam, t):
    a, n = fam.get(t, [0, 0]); return f"{a}/{n}"


def cusum_eval(sv, st, vlab, tlab, clean_v):
    mu = np.nanmean(sv[clean_v]); sd = np.nanstd(sv[clean_v]) + 1e-12
    svs, sts = (sv - mu) / sd, (st - mu) / sd
    win_v, blk_v = segments(vlab); win_t, blk_t = segments(tlab)
    best = None
    for k in KS:
        pos = np.array([np.max(cusum1(svs[w["rows"]], k)) for w in win_v])
        neg = np.array([np.max(cusum1(svs[b], k)) for b in blk_v])
        auc = roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        if best is None or auc > best[0]:
            best = (auc, k, neg)
    _, k, negv = best
    h = float(np.quantile(negv, 0.99))
    att = tlab["attacked"].to_numpy()
    det, fam, delays = {}, {}, []
    for w in win_t:
        g = cusum1(sts[w["rows"]], k)
        cr = np.where(g > h)[0]
        dd = len(cr) > 0
        det[w["wid"]] = dd
        if dd:
            delays.append(int(cr.min()))
        fam.setdefault(w["type"], [0, 0]); fam[w["type"]][1] += 1; fam[w["type"]][0] += int(dd)
    fb = sum(int(np.any(cusum1(sts[b], k) > h)) for b in blk_t)
    return {"k": k, "FAR": fb / len(blk_t), "win_det": int(sum(det.values())), "n_win": len(win_t),
            "A1": _f(fam, "A1"), "A2": _f(fam, "A2"), "A3": _f(fam, "A3"),
            "median_delay": float(np.median(delays)) if delays else float("nan")}


def logistic_fit(X, y, iters=4000, lr=0.5, l2=1e-4):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w); b -= lr * np.mean(p - y)
    return w, b


def main():
    wz = np.load(OUT / "detect_ae_weights.npz", allow_pickle=True)
    P = {k: wz[k] for k in PARAM_KEYS}
    mu_ae, sd_ae, r_c = wz["mu"], wz["sd"], wz["r_c"]
    Xtr = pd.read_csv(FDIA / "train_z.csv").drop(columns="timestamp").to_numpy(float)
    Xva = pd.read_csv(FDIA / "val_z.csv").drop(columns="timestamp").to_numpy(float)
    Xte = pd.read_csv(FDIA / "test_z.csv").drop(columns="timestamp").to_numpy(float)
    vlab = pd.read_csv(FDIA / "val_labels.csv"); tlab = pd.read_csv(FDIA / "test_labels.csv")

    Etr = ae_errors(P, mu_ae, sd_ae, r_c, Xtr)
    Eva = ae_errors(P, mu_ae, sd_ae, r_c, Xva)
    Ete = ae_errors(P, mu_ae, sd_ae, r_c, Xte)
    tr = Etr[np.isfinite(Etr[:, 0])]
    mu = tr.mean(0)
    Sigma, S, rho = ledoit_wolf(tr - mu)
    Minv = np.linalg.inv(Sigma)

    def maha(E):
        s = np.full(len(E), np.nan); fin = np.isfinite(E[:, 0]); d = E[fin] - mu
        s[fin] = np.einsum("ij,ij->i", d @ Minv, d); return s
    mv, mt = maha(Eva), maha(Ete)
    att_v, att_t = vlab["attacked"].to_numpy(), tlab["attacked"].to_numpy()
    clean_v = (att_v == 0) & np.isfinite(mv)

    # point (unsmoothed)
    thr_pt = float(np.quantile(mv[clean_v], 0.99))
    pt = eval_point(mt, tlab, thr_pt)
    print(f"point Mahalanobis: FAR {pt['FAR']*100:.2f}% win {pt['win_det']}/18 AUC {pt['per_ts_AUC']:.4f} "
          f"winAUC {pt['window_AUC']:.4f} fam {pt['A1']},{pt['A2']},{pt['A3']} delay {pt['median_delay']}", flush=True)

    # STEP A: trailing mean, pick L by val window-AUC
    bestL = None
    for L in L_CHOICES:
        a = win_auc(trailing_mean(mv, L), vlab)
        if bestL is None or a > bestL[0]:
            bestL = (a, L)
    L = bestL[1]
    tmv, tmt = trailing_mean(mv, L), trailing_mean(mt, L)
    thr_tm = float(np.quantile(tmv[clean_v & np.isfinite(tmv)], 0.99))
    sm = eval_point(tmt, tlab, thr_tm)
    cu = cusum_eval(mv, mt, vlab, tlab, clean_v)
    print(f"[A] trailing-mean L={L}: FAR {sm['FAR']*100:.2f}% win {sm['win_det']}/18 winAUC {sm['window_AUC']:.4f} "
          f"fam {sm['A1']},{sm['A2']},{sm['A3']} delay {sm['median_delay']}", flush=True)
    print(f"[A] CUSUM k={cu['k']}: FAR {cu['FAR']*100:.2f}% win {cu['win_det']}/18 fam {cu['A1']},{cu['A2']},{cu['A3']} "
          f"delay {cu['median_delay']}", flush=True)

    # STEP B: fusion BDD + Mahalanobis
    vs = pd.read_csv(OUT / "val_scores.csv"); bdd_v = vs["bdd_score"].to_numpy()
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    bdd_t = np.full(len(tlab), np.nan); bdd_t[phys["row"].astype(int)] = phys["bdd_score"].astype(float)
    cvf = (att_v == 0) & np.isfinite(bdd_v) & np.isfinite(mv)
    muf = np.array([bdd_v[cvf].mean(), mv[cvf].mean()]); sdf = np.array([bdd_v[cvf].std(), mv[cvf].std()]) + 1e-12
    both = np.isfinite(bdd_v) & np.isfinite(mv)
    Xv = np.column_stack([(bdd_v[both] - muf[0]) / sdf[0], (mv[both] - muf[1]) / sdf[1]])
    w, b = logistic_fit(Xv, att_v[both].astype(float))
    auc_lr = roc_auc(1 / (1 + np.exp(-(Xv @ w + b))), att_v[both])
    def fus(bd, m): return 1 / (1 + np.exp(-(((bd - muf[0]) / sdf[0]) * w[0] + ((m - muf[1]) / sdf[1]) * w[1] + b)))
    fv, ft = fus(bdd_v, mv), fus(bdd_t, mt)
    thr_f = float(np.quantile(fv[(att_v == 0) & np.isfinite(fv)], 0.99))
    fu = eval_point(ft, tlab, thr_f)
    print(f"[B] fusion BDD+Maha: val AUC {auc_lr:.4f} weights bdd {w[0]:.3f} maha {w[1]:.3f} -> test FAR "
          f"{fu['FAR']*100:.2f}% win {fu['win_det']}/18 AUC {fu['per_ts_AUC']:.4f} fam {fu['A1']},{fu['A2']},{fu['A3']}",
          flush=True)

    # localization for point & smoothed (top Mahalanobis contribution)
    def loc(score, thr):
        wins, _ = segments(tlab); hits = tot = 0
        cols = [c for c in pd.read_csv(FDIA / "test_z.csv", nrows=0).columns if c != "timestamp"]
        for wd in wins:
            rows = wd["rows"][np.isfinite(score[wd["rows"]])]
            if not len(rows) or not np.any(score[rows] > thr):
                continue
            tot += 1; contrib = np.zeros(Ete.shape[1])
            for rr in rows:
                dv = Ete[rr] - mu; contrib += np.where(np.isfinite(dv), dv * (Minv @ dv), 0.0)
            hits += int(cols[int(np.argmax(contrib))] in set(str(wd["targets"]).split(";")))
        return (hits / tot) if tot else float("nan"), hits, tot
    loc_pt = loc(mt, thr_pt); loc_sm = loc(tmt, thr_tm)

    table = [
        {"detector": "BDD any-step", "FAR": 0.0113, "step_det": None, "window": "10/18",
         "A1": "8/12", "A2": "1/3", "A3": "1/3", "per_ts_AUC": 0.607, "window_AUC": 0.784, "localization": None},
        {"detector": "AE Mahalanobis (point)", "FAR": pt["FAR"], "step_det": pt["step_det"],
         "window": f"{pt['win_det']}/18", "A1": pt["A1"], "A2": pt["A2"], "A3": pt["A3"],
         "per_ts_AUC": pt["per_ts_AUC"], "window_AUC": pt["window_AUC"], "localization": loc_pt[0]},
        {"detector": f"AE Mahalanobis (trailing-mean L={L})", "FAR": sm["FAR"], "step_det": sm["step_det"],
         "window": f"{sm['win_det']}/18", "A1": sm["A1"], "A2": sm["A2"], "A3": sm["A3"],
         "per_ts_AUC": sm["per_ts_AUC"], "window_AUC": sm["window_AUC"], "localization": loc_sm[0]},
        {"detector": "fusion BDD+Mahalanobis", "FAR": fu["FAR"], "step_det": fu["step_det"],
         "window": f"{fu['win_det']}/18", "A1": fu["A1"], "A2": fu["A2"], "A3": fu["A3"],
         "per_ts_AUC": fu["per_ts_AUC"], "window_AUC": fu["window_AUC"], "localization": None},
    ]
    pd.DataFrame(table).to_csv(OUT / "detect_final_table.csv", index=False)
    json.dump({"point": pt, "trailing_mean": {**sm, "L": L}, "cusum": cu, "fusion": fu,
               "fusion_val_AUC": float(auc_lr), "fusion_weights": {"bdd": float(w[0]), "maha": float(w[1]), "b": float(b)},
               "loc_point": loc_pt[0], "loc_smoothed": loc_sm[0], "lw_lambda": rho},
              open(OUT / "detect_final_summary.json", "w"), indent=2, default=lambda o: None)
    pd.set_option("display.width", 230)
    print("\n" + pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
