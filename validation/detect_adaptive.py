"""
v5 -- diagnose the clean val->test score drift and fix the operating
point with a LABEL-FREE adaptive rolling threshold. numpy only; no
estimator runs. Detection uses the v3 AE (d=32); the v4 d=48 AE is
retired for detection (it reconstructs anomalies too well -> lower
detection AUC, see reports/detector_compare_v4.md). The v3 AE's
per-channel errors are recomputed here from its saved weights
(detect_ae_weights.npz) -- an AE forward pass, not an estimator run.

Part 1  clean-score distribution across train(AE)/val/test (mean, std,
        p90, p99) for bdd and v3 ae; clean test score vs day-of-series
        and hour-of-day; drift diagnosis.
Part 2  adaptive rolling threshold: flag s_t > median+k*MAD over the
        trailing W_adapt scores (robust to ~12% attacked). k set on VAL
        for 1% clean-val FAR, applied to test. Evaluated for BDD
        any-step, AE per-channel (adaptive per-channel), fusion_LR;
        static vs adaptive side by side.
Part 3  final comparison + reports/detector_compare_v5.md.

Run: python -m validation.detect_adaptive
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_ae import forward, windows_of, W, FLOOR
from validation.detect_eval import roc_auc

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
CORPUS = OUT.parent / "simulation" / "phase6_corpus"
PARAM_KEYS = ["Wz", "Wr", "Wn", "Uz", "Ur", "Un", "bz", "br", "bn", "Wc", "bc", "Wd", "bd"]
W_ADAPT = [720, 1440]


def hours(split):
    ts = pd.read_csv(CORPUS / f"{split}_telemetry.csv", usecols=["real_timestamp"])["real_timestamp"]
    return np.array([int(str(t)[11:13]) for t in ts])


def logistic_fit(X, y, iters=4000, lr=0.5, l2=1e-4):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w); b -= lr * np.mean(p - y)
    return w, b


def ae_errors(P, mu, sd, r_c, X):
    """Per-channel normalized reconstruction error e_{c,t}/r_c (NaN head)."""
    S = (X - mu) / sd
    N, D = S.shape
    E = np.full((N, D), np.nan)
    idx = np.arange(W - 1, N)
    for b in range(0, len(idx), 2048):
        ii = idx[b:b + 2048]
        out, _ = forward(P, windows_of(S, ii))
        E[ii] = np.abs(X[ii] - (out * sd + mu)) / r_c
    return E


def load():
    vlab = pd.read_csv(FDIA / "val_labels.csv"); tlab = pd.read_csv(FDIA / "test_labels.csv")
    vs = pd.read_csv(OUT / "val_scores.csv")
    bdd_v = vs["bdd_score"].to_numpy()
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    bdd_t = np.full(len(tlab), np.nan); bdd_t[phys["row"].astype(int)] = phys["bdd_score"].astype(float)
    aev = pd.read_csv(OUT / "detect_ae_val_pertimestep.csv")        # v3 AE d=32
    aet = pd.read_csv(OUT / "detect_ae_test_pertimestep.csv")
    ae_v = np.full(len(vlab), np.nan); ae_v[aev["row"].astype(int)] = aev["score"].astype(float)
    ae_t = np.full(len(tlab), np.nan); ae_t[aet["row"].astype(int)] = aet["score"].astype(float)
    y_v = vlab["attacked"].to_numpy()
    cv = (y_v == 0) & np.isfinite(bdd_v) & np.isfinite(ae_v)
    mu = np.array([bdd_v[cv].mean(), ae_v[cv].mean()]); sd = np.array([bdd_v[cv].std(), ae_v[cv].std()]) + 1e-12
    both = np.isfinite(bdd_v) & np.isfinite(ae_v)
    Xv = np.column_stack([(bdd_v[both] - mu[0]) / sd[0], (ae_v[both] - mu[1]) / sd[1]])
    w, b = logistic_fit(Xv, y_v[both].astype(float))
    def lr(bd, ae): return 1 / (1 + np.exp(-(((bd - mu[0]) / sd[0]) * w[0] + ((ae - mu[1]) / sd[1]) * w[1] + b)))
    return dict(vlab=vlab, tlab=tlab,
                bdd=dict(val=bdd_v, test=bdd_t), ae=dict(val=ae_v, test=ae_t),
                fusion_LR=dict(val=lr(bdd_v, ae_v), test=lr(bdd_t, ae_t)),
                lr=dict(w=w.tolist(), b=float(b)))


def dist_stats(x):
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()),
                p90=float(np.quantile(x, 0.90)), p99=float(np.quantile(x, 0.99)))


def roll_threshold(s, Wa, k, stride=1):
    n = len(s); thr = np.full(n, np.inf); last = None
    for t in range(n):
        if t < Wa:
            continue
        if last is None or (t % stride) == 0:
            win = s[t - Wa:t]; win = win[np.isfinite(win)]
            if len(win) < Wa // 2:
                last = None; continue
            med = np.median(win); mad = np.median(np.abs(win - med)) + 1e-12
            last = (med, mad)
        if last is not None:
            thr[t] = last[0] + k * last[1]
    return thr


def tune_k(s, attacked, Wa, target=0.01, stride=1):
    clean = (attacked == 0)
    lo, hi = 0.0, 50.0
    for _ in range(40):
        k = 0.5 * (lo + hi)
        thr = roll_threshold(s, Wa, k, stride)
        m = clean & np.isfinite(thr) & np.isfinite(s)
        far = float(np.mean(s[m] > thr[m])) if m.any() else 1.0
        if far > target:
            lo = k
        else:
            hi = k
    return hi


def windows(lab):
    att = lab["attacked"].to_numpy().astype(bool)
    return [dict(wid=int(wid), rows=np.array(sorted(g.index)), type=g.iloc[0]["type"],
                 targets=g.iloc[0]["targets"]) for wid, g in lab[att].groupby("window_id")]


def eval_flags(flag, lab):
    att = lab["attacked"].to_numpy().astype(bool)
    wins = windows(lab)
    det = {w["wid"]: bool(flag[w["rows"]].any()) for w in wins}
    fam = {}
    for w in wins:
        fam.setdefault(w["type"], [0, 0]); fam[w["type"]][1] += 1; fam[w["type"]][0] += int(det[w["wid"]])
    return (sum(det.values()), len(wins), {t: f"{a}/{n}" for t, (a, n) in fam.items()})


def main():
    d = load()
    vlab, tlab = d["vlab"], d["tlab"]
    att_v, att_t = vlab["attacked"].to_numpy(), tlab["attacked"].to_numpy()

    # recompute v3 AE per-channel errors
    wz = np.load(OUT / "detect_ae_weights.npz", allow_pickle=True)
    P = {k: wz[k] for k in PARAM_KEYS}
    mu_ae, sd_ae, r_c = wz["mu"], wz["sd"], wz["r_c"]
    Xva = pd.read_csv(FDIA / "val_z.csv").drop(columns="timestamp").to_numpy(float)
    Xte = pd.read_csv(FDIA / "test_z.csv").drop(columns="timestamp").to_numpy(float)
    cols = [c for c in pd.read_csv(FDIA / "test_z.csv", nrows=0).columns if c != "timestamp"]
    Eva, Ete = ae_errors(P, mu_ae, sd_ae, r_c, Xva), ae_errors(P, mu_ae, sd_ae, r_c, Xte)

    # ---- Part 1: drift
    Xtr = pd.read_csv(FDIA / "train_z.csv").drop(columns="timestamp").to_numpy(float)
    Str = (Xtr - mu_ae) / sd_ae
    tr_idx = np.arange(W - 1, len(Str))[::10]            # subsample for the distribution
    ae_tr = np.full(len(Str), np.nan)
    for b in range(0, len(tr_idx), 2048):
        ii = tr_idx[b:b + 2048]; out, _ = forward(P, windows_of(Str, ii))
        ae_tr[ii] = (np.abs(Xtr[ii] - (out * sd_ae + mu_ae)) / r_c).max(1)
    drift = {"bdd": {"val_clean": dist_stats(d["bdd"]["val"][att_v == 0]),
                     "test_clean": dist_stats(d["bdd"]["test"][att_t == 0])},
             "ae": {"train_clean": dist_stats(ae_tr[tr_idx]),
                    "val_clean": dist_stats(d["ae"]["val"][att_v == 0]),
                    "test_clean": dist_stats(d["ae"]["test"][att_t == 0])}}
    # test clean by day and hour
    ht = hours("test")
    rows_day = []
    for day in range(len(tlab) // 1440 + 1):
        m = (np.arange(len(tlab)) // 1440 == day) & (att_t == 0)
        if m.any():
            rows_day.append({"day": day, "bdd_mean": float(np.nanmean(d["bdd"]["test"][m])),
                             "ae_mean": float(np.nanmean(d["ae"]["test"][m]))})
    rows_hr = []
    for h in range(24):
        m = (ht == h) & (att_t == 0)
        rows_hr.append({"hour": h, "bdd_mean": float(np.nanmean(d["bdd"]["test"][m])),
                        "ae_mean": float(np.nanmean(d["ae"]["test"][m]))})
    pd.DataFrame(rows_day).to_csv(OUT / "detect_v5_drift_by_day.csv", index=False)
    pd.DataFrame(rows_hr).to_csv(OUT / "detect_v5_drift_by_hour.csv", index=False)
    json.dump(drift, open(OUT / "detect_v5_drift.json", "w"), indent=2)
    print("Part 1 drift (clean score distribution):")
    for det in ["bdd", "ae"]:
        for split, st in drift[det].items():
            print(f"  {det:4s} {split:11s} mean {st['mean']:.4f} std {st['std']:.4f} "
                  f"p90 {st['p90']:.4f} p99 {st['p99']:.4f}")

    # ---- Part 2: adaptive vs static
    static_thr = {det: float(np.quantile(d[det]["val"][(att_v == 0) & np.isfinite(d[det]["val"])], 0.99))
                  for det in ["bdd", "ae", "fusion_LR"]}
    results = []
    for det in ["bdd", "ae", "fusion_LR"]:
        sv, st = d[det]["val"], d[det]["test"]
        # static
        fin_t = np.isfinite(st)
        s_flag = np.zeros(len(st), bool); s_flag[fin_t] = st[fin_t] > static_thr[det]
        s_far = float(np.mean(s_flag[(att_t == 0) & fin_t]))
        s_det, s_n, s_fam = eval_flags(s_flag, tlab)
        # adaptive: pick W_adapt by val window detection at tuned 1%
        best = None
        for Wa in W_ADAPT:
            k = tune_k(sv, att_v, Wa, 0.01, stride=1)
            thr_v = roll_threshold(sv, Wa, k, 1)
            fv = np.isfinite(thr_v) & np.isfinite(sv)
            vflag = np.zeros(len(sv), bool); vflag[fv] = sv[fv] > thr_v[fv]
            vdet, vn, _ = eval_flags(vflag, vlab)
            if best is None or vdet > best[0]:
                best = (vdet, Wa, k)
        _, Wa, k = best
        thr_t = roll_threshold(st, Wa, k, 1)
        ft = np.isfinite(thr_t) & fin_t
        a_flag = np.zeros(len(st), bool); a_flag[ft] = st[ft] > thr_t[ft]
        a_far = float(np.mean(a_flag[(att_t == 0) & ft]))
        a_step = float(np.mean(a_flag[(att_t == 1) & ft]))
        a_det, a_n, a_fam = eval_flags(a_flag, tlab)
        auc = roc_auc(st[fin_t], att_t[fin_t].astype(bool))
        results.append({"detector": det, "static_FAR": s_far, "static_win": f"{s_det}/{s_n}", "static_fam": s_fam,
                        "adaptive_W": Wa, "adaptive_k": k, "adaptive_FAR": a_far, "adaptive_step_det": a_step,
                        "adaptive_win": f"{a_det}/{a_n}", "adaptive_fam": a_fam, "per_timestep_AUC": auc})
        print(f"[2] {det}: static FAR {s_far*100:.2f}% win {s_det}/{s_n} | adaptive(W={Wa}) FAR {a_far*100:.2f}% "
              f"win {a_det}/{a_n} fam {a_fam}", flush=True)

    # ---- Part 2b: per-channel adaptive (v3 AE errors), k global tuned on val for 1% feeder FAR
    def perchannel_flags(E, Wa, k, stride=30):
        n, C = E.shape
        flag = np.zeros(n, bool); topch = np.full(n, -1)
        thr = np.full((n, C), np.inf); last = None
        for t in range(n):
            if t < Wa:
                continue
            if last is None or (t % stride) == 0:
                win = E[t - Wa:t]
                med = np.nanmedian(win, 0)
                mad = np.nanmedian(np.abs(win - med), 0) + 1e-9
                last = (med, mad)
            h = last[0] + k * last[1]
            e = E[t]
            cross = np.isfinite(e) & (e > h)
            if cross.any():
                flag[t] = True
                margin = np.where(cross, e - h, -np.inf)
                topch[t] = int(np.nanargmax(margin))
        return flag, topch

    def pc_far(k, Wa):
        fl, _ = perchannel_flags(Eva, Wa, k)
        m = (att_v == 0) & np.isfinite(Eva[:, 0])
        return float(np.mean(fl[m]))
    Wa_pc = 1440
    lo, hi = 0.0, 50.0
    for _ in range(22):
        k = 0.5 * (lo + hi)
        if pc_far(k, Wa_pc) > 0.01:
            lo = k
        else:
            hi = k
    k_pc = hi
    fl_t, top_t = perchannel_flags(Ete, Wa_pc, k_pc)
    pc_far_t = float(np.mean(fl_t[(att_t == 0) & np.isfinite(Ete[:, 0])]))
    pc_det, pc_n, pc_fam = eval_flags(fl_t, tlab)
    # localization
    lh = lt = 0
    for w in windows(tlab):
        if fl_t[w["rows"]].any():
            lt += 1
            tch = [cols[top_t[r]] for r in w["rows"] if top_t[r] >= 0]
            tgt = set(str(w["targets"]).split(";"))
            # most frequent top channel in the window
            if tch:
                from collections import Counter
                top = Counter(tch).most_common(1)[0][0]
                lh += int(top in tgt)
    pc_loc = lh / lt if lt else float("nan")
    print(f"[2b] per-channel ADAPTIVE (W={Wa_pc}, k={k_pc:.3f}): test FAR {pc_far_t*100:.2f}% "
          f"win {pc_det}/{pc_n} fam {pc_fam} localization {lh}/{lt}={pc_loc*100:.1f}%", flush=True)

    v4 = json.load(open(OUT / "detect_v4_summary.json"))
    pc_static = v4["perchannel_tuned1pct"]
    print(f"     (static per-channel v4 tuned@1%: test FAR {pc_static['clean_timestep_FAR']*100:.2f}% "
          f"win {pc_static['windows_detected']}/18)")

    summary = {"drift": drift, "adaptive": results, "lr": d["lr"],
               "perchannel_adaptive": {"W": Wa_pc, "k": k_pc, "test_FAR": pc_far_t, "windows": f"{pc_det}/{pc_n}",
                                       "by_family": pc_fam, "localization": pc_loc, "loc_hits": lh, "loc_tot": lt},
               "perchannel_static_v4": pc_static, "static_thr": static_thr}
    json.dump(summary, open(OUT / "detect_v5_summary.json", "w"), indent=2, default=float)
    pd.DataFrame(results).to_csv(OUT / "detect_v5_adaptive.csv", index=False)
    print("\nsaved detect_v5_summary.json")


if __name__ == "__main__":
    main()
