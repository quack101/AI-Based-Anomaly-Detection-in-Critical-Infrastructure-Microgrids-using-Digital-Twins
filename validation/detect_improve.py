"""
Parts B-E: improved decision rules over the saved detector scores
(bdd, ae [AE v2], fusion_LR). numpy only; estimator not run.

B  CUSUM on the per-timestep score (segment-reset evaluation: CUSUM
   accumulates within a segment -- an attack window, or a non-overlapping
   60-step clean block -- from 0). k in {0.25,0.5,1.0} and threshold h
   calibrated on clean val to 1% window-FAR; k picked by val window-AUC.
C  per-channel CUSUM on the AE v2 normalized errors, per-channel
   threshold by Bonferroni (per-channel alpha = 1%/168) on clean val;
   flag if any channel crosses. Localization = top-CUSUM channel in the
   window's attacked targets.
D  per-hour (24) thresholds on clean val at 1%, applied to test by hour.
E  final comparison + reports/detector_compare_v4.md.

Run: python -m validation.detect_improve
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import roc_auc
from validation.detect_scores import load_scores

OUT = Path(__file__).resolve().parent
L_FAR = 60
KS = [0.25, 0.5, 1.0]
FAR_TARGETS = [0.005, 0.01, 0.02, 0.05]
W = 10


def segments(lab):
    att = lab["attacked"].to_numpy().astype(bool)
    n = len(lab)
    windows = []
    for wid, g in lab[att].groupby("window_id"):
        r0 = g.iloc[0]
        windows.append({"wid": int(wid), "rows": np.array(sorted(g.index)), "type": r0["type"],
                        "class": r0["class"], "level": r0["level"], "profile": r0["profile"],
                        "duration": int(r0["duration"]), "targets": r0["targets"]})
    blocks = []
    i = 0
    while i < n:
        if att[i]:
            i += 1; continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j - L_FAR + 1, L_FAR):
            blocks.append(np.arange(b, b + L_FAR))
        i = j
    return windows, blocks


def cusum1(x, k):
    g = 0.0; out = np.empty(len(x))
    for t, v in enumerate(x):
        g = max(0.0, g + (v - k if np.isfinite(v) else -k))
        out[t] = g
    return out


def std_by_clean(score_v, score_t, clean_mask):
    mu = np.nanmean(score_v[clean_mask]); sd = np.nanstd(score_v[clean_mask]) + 1e-12
    return (score_v - mu) / sd, (score_t - mu) / sd


# ---------- Part B: CUSUM ----------
def cusum_detector(name, S, clean_v):
    sv, st = std_by_clean(S["val"][name], S["test"][name], clean_v)
    win_v, blk_v = segments(S["vlab"])
    win_t, blk_t = segments(S["tlab"])

    def maxg(seq_std, rows, k):
        return float(np.max(cusum1(seq_std[rows], k)))

    best = None
    for k in KS:
        pos = np.array([maxg(sv, w["rows"], k) for w in win_v])
        neg = np.array([maxg(sv, b, k) for b in blk_v])
        auc = roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        if best is None or auc > best[0]:
            best = (auc, k, neg)
    auc_k, k, neg_v = best
    h = float(np.quantile(neg_v, 0.99))

    rows = []
    delays = []
    for w in win_t:
        g = cusum1(st[w["rows"]], k)
        det = bool(np.any(g > h))
        if det:
            delays.append(int(np.argmax(g > h)))
        rows.append({**{kk: w[kk] for kk in ("wid", "type", "class", "level", "profile", "duration")},
                     "detected": det})
    win_df = pd.DataFrame(rows)
    fb = sum(int(np.any(cusum1(st[b], k) > h)) for b in blk_t)
    far = fb / len(blk_t)
    return {"k": k, "h": h, "val_window_AUC": auc_k, "win_df": win_df,
            "window_rate": float(win_df["detected"].mean()), "windows_detected": int(win_df["detected"].sum()),
            "clean_block_FAR": far, "median_delay": float(np.median(delays)) if delays else float("nan"),
            "sv_std": sv, "st_std": st}


# ---------- mean-window (v3 rule, recomputed on v2 scores) for comparison ----------
def trailing_mean(s, L):
    n = len(s); out = np.full(n, np.nan)
    fin = np.isfinite(s); sf = np.where(fin, s, 0.0)
    cs = np.concatenate([[0], np.cumsum(sf)]); cf = np.concatenate([[0], np.cumsum(fin)])
    for t in range(L - 1, n):
        if cf[t + 1] - cf[t + 1 - L] == L:
            out[t] = (cs[t + 1] - cs[t + 1 - L]) / L
    return out


def meanwin_detector(name, S, clean_v, L=15):
    """Trailing-mean rule with a CONSISTENT per-timestep 1% threshold:
    h = 99th pct of the trailing mean over clean-val timesteps; a window
    is detected if any attacked step's trailing mean exceeds h; realized
    FAR is the clean-test per-timestep rate. Window-level AUC = max
    trailing-mean over a window vs over clean 60-step blocks."""
    sv, st = S["val"][name], S["test"][name]
    tm_v, tm_t = trailing_mean(sv, L), trailing_mean(st, L)
    win_t, blk_t = segments(S["tlab"])
    cv = clean_v & np.isfinite(tm_v)
    h = float(np.quantile(tm_v[cv], 0.99))
    att_t = S["test"]["attacked"].astype(bool)
    fin_t = np.isfinite(tm_t)
    far = float(np.mean(tm_t[fin_t & ~att_t] > h))
    det = []
    for w in win_t:
        rows = [r for r in w["rows"] if np.isfinite(tm_t[r])]
        det.append(bool(np.any(tm_t[rows] > h)) if rows else False)
    pos = np.array([np.max(tm_t[[r for r in w["rows"] if np.isfinite(tm_t[r])] or [w["rows"][0]]]) for w in win_t])
    neg = np.array([np.max(tm_t[b][np.isfinite(tm_t[b])]) if np.isfinite(tm_t[b]).any() else np.nan for b in blk_t])
    auc = roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
    return {"window_rate": float(np.mean(det)), "windows_detected": int(np.sum(det)),
            "realized_FAR": far, "window_AUC": float(auc), "L": L}


# ---------- Part C: per-channel CUSUM on AE errors ----------
def perchannel(S, k=0.5):
    Ev, Et = S["val_err"], S["test_err"]            # (N,168) normalized errors, NaN head
    cols = S["cols"]
    vlab, tlab = S["vlab"], S["tlab"]
    att_v = vlab["attacked"].to_numpy().astype(bool)
    # standardize each channel by clean-val mean/std
    cv = (~att_v) & np.isfinite(Ev[:, 0])
    mu = np.nanmean(Ev[cv], 0); sd = np.nanstd(Ev[cv], 0) + 1e-12
    Zt = (Et - mu) / sd
    Zv = (Ev - mu) / sd
    win_v, blk_v = segments(vlab)
    win_t, blk_t = segments(tlab)
    alpha = 0.01 / Ev.shape[1]                       # Bonferroni per-channel

    def run_cusum2(Z, rows):
        g = np.zeros(Z.shape[1]); G = np.empty((len(rows), Z.shape[1]))
        for i, r in enumerate(rows):
            v = Z[r]; v = np.where(np.isfinite(v), v, 0.0)
            g = np.maximum(0.0, g + v - k); G[i] = g
        return G

    # per-channel clean-val CUSUM values (all clean-block timesteps), for calibration
    Gv_all = np.vstack([run_cusum2(Zv, b) for b in blk_v])                 # (Nsteps, C)
    # Bonferroni: per-channel (1-alpha) quantile
    h_bonf = np.quantile(Gv_all, 1 - alpha, axis=0)
    # tuned global level q so union per-timestep FAR on clean val = 1%
    lo, hi = 0.90, 1.0 - 1e-6
    for _ in range(40):
        q = 0.5 * (lo + hi)
        tc = np.quantile(Gv_all, q, axis=0)
        far = float(np.mean((Gv_all > tc).any(1)))
        if far > 0.01:
            lo = q
        else:
            hi = q
    h_tuned = np.quantile(Gv_all, hi, axis=0)
    h_c = h_bonf   # primary report = Bonferroni (principled); tuned reported alongside

    rows_out = []
    loc_hits = loc_tot = 0
    for w in win_t:
        G = run_cusum2(Zt, w["rows"])
        crossed = G > h_c
        det = bool(crossed.any())
        rows_out.append({**{kk: w[kk] for kk in ("wid", "type", "class", "level", "profile", "duration")},
                         "detected": det})
        if det:
            top = cols[int(G.max(0).argmax())]
            tgt = set(str(w["targets"]).split(";")) if isinstance(w["targets"], str) else set()
            loc_tot += 1
            if top in tgt:
                loc_hits += 1
    win_df = pd.DataFrame(rows_out)

    def eval_thr(h_vec):
        wins = 0
        lh = lt = 0
        fb = fs = ts = 0
        wrows = []
        for w in win_t:
            G = run_cusum2(Zt, w["rows"])
            det = bool((G > h_vec).any())
            wins += det
            wrows.append({**{kk: w[kk] for kk in ("wid", "type", "class", "level", "profile", "duration")},
                          "detected": det})
            if det:
                lt += 1
                top = cols[int(G.max(0).argmax())]
                tgt = set(str(w["targets"]).split(";")) if isinstance(w["targets"], str) else set()
                lh += int(top in tgt)
        for b in blk_t:
            anyc = (run_cusum2(Zt, b) > h_vec).any(1)
            fb += int(anyc.any()); fs += int(anyc.sum()); ts += len(b)
        return {"win_df": pd.DataFrame(wrows), "windows_detected": wins, "window_rate": wins / len(win_t),
                "clean_block_FAR": fb / len(blk_t), "clean_timestep_FAR": fs / ts,
                "localization_acc": (lh / lt) if lt else float("nan"), "loc_hits": lh, "loc_tot": lt}

    bonf = eval_thr(h_bonf)
    tuned = eval_thr(h_tuned)
    return {"bonferroni": bonf, "tuned_1pct": tuned, "k": k, "alpha": alpha,
            "win_df": bonf["win_df"], **{kk: bonf[kk] for kk in
            ("window_rate", "windows_detected", "clean_block_FAR", "clean_timestep_FAR",
             "localization_acc", "loc_hits", "loc_tot")}}


# ---------- Part D: diurnal thresholds ----------
def diurnal(name, S, clean_v, far=0.01):
    sv, st = S["val"][name], S["test"][name]
    hv, ht = S["val"]["hour"], S["test"]["hour"]
    att_t = S["test"]["attacked"].astype(bool)
    thr_by_hour = {}
    for h in range(24):
        m = clean_v & (hv == h) & np.isfinite(sv)
        thr_by_hour[h] = float(np.quantile(sv[m], 1 - far)) if m.sum() > 20 else np.inf
    thr_t = np.array([thr_by_hour[h] for h in ht])
    fin = np.isfinite(st)
    realized_far = float(np.mean(st[fin & ~att_t] > thr_t[fin & ~att_t]))
    # window detection (any attacked step over its hourly threshold)
    win_t, _ = segments(S["tlab"])
    det = []
    for w in win_t:
        rows = [r for r in w["rows"] if np.isfinite(st[r])]
        det.append(bool(np.any([st[r] > thr_by_hour[ht[r]] for r in rows])) if rows else False)
    # single-threshold baseline at same far
    sglthr = float(np.quantile(sv[clean_v & np.isfinite(sv)], 1 - far))
    realized_far_single = float(np.mean(st[fin & ~att_t] > sglthr))
    det_single = []
    for w in win_t:
        rows = [r for r in w["rows"] if np.isfinite(st[r])]
        det_single.append(bool(np.any(st[rows] > sglthr)) if rows else False)
    return {"realized_FAR_diurnal": realized_far, "window_rate_diurnal": float(np.mean(det)),
            "realized_FAR_single": realized_far_single, "window_rate_single": float(np.mean(det_single))}


def by_family(win_df):
    return {t: f"{int(g['detected'].sum())}/{len(g)}" for t, g in win_df.groupby("type")}


def main():
    S = load_scores()
    y_v = S["val"]["attacked"]
    both_v = {name: np.isfinite(S["val"][name]) for name in S["detectors"]}
    print(f"fusion_LR refit val AUC {S['lr']['val_AUC']:.4f}, weights {S['lr']['w']}, bias {S['lr']['b']:.3f}",
          flush=True)

    # ---- Part B
    partB = {}
    for name in S["detectors"]:
        clean_v = (y_v == 0) & both_v[name]
        cu = cusum_detector(name, S, clean_v)
        mw = meanwin_detector(name, S, clean_v)
        partB[name] = {"cusum": cu, "meanwin": mw}
        cu["win_df"].to_csv(OUT / f"detect_v4_cusum_{name}.csv", index=False)
        print(f"[B] {name}: CUSUM k={cu['k']} h={cu['h']:.3f} val_winAUC {cu['val_window_AUC']:.4f} -> "
              f"test windows {cu['windows_detected']}/18 (FAR {cu['clean_block_FAR']*100:.2f}%, delay "
              f"{cu['median_delay']}); mean-window {mw['windows_detected']}/18 (FAR {mw['realized_FAR']*100:.2f}%, winAUC {mw['window_AUC']:.3f})",
              flush=True)
        print(f"     CUSUM by family: {by_family(cu['win_df'])}", flush=True)

    # ---- Part C
    pc = perchannel(S)
    pc["win_df"].to_csv(OUT / "detect_v4_perchannel.csv", index=False)
    print(f"\n[C] per-channel AE CUSUM (k={pc['k']}, Bonferroni alpha {pc['alpha']:.2e}): "
          f"windows {pc['windows_detected']}/18 (per-timestep FAR {pc['clean_timestep_FAR']*100:.2f}%, "
          f"block-FAR {pc['clean_block_FAR']*100:.1f}%), "
          f"localization {pc['loc_hits']}/{pc['loc_tot']} = "
          f"{pc['localization_acc']*100:.1f}%  by family {by_family(pc['win_df'])}", flush=True)
    t = pc['tuned_1pct']
    print(f"    tuned@1%: windows {t['windows_detected']}/18 (per-timestep FAR {t['clean_timestep_FAR']*100:.2f}%, "
          f"block-FAR {t['clean_block_FAR']*100:.1f}%), localization {t['loc_hits']}/{t['loc_tot']}="
          f"{t['localization_acc']*100:.1f}%  by family {by_family(t['win_df'])}", flush=True)

    # ---- Part D
    partD = {}
    for name in S["detectors"]:
        clean_v = (y_v == 0) & both_v[name]
        partD[name] = diurnal(name, S, clean_v)
        d = partD[name]
        print(f"[D] {name}: diurnal FAR {d['realized_FAR_diurnal']*100:.2f}% win {d['window_rate_diurnal']*100:.1f}% "
              f"vs single FAR {d['realized_FAR_single']*100:.2f}% win {d['window_rate_single']*100:.1f}%", flush=True)

    # ---- Part E: AUCs + multi-FAR (CUSUM rule) + old/new
    att_t = S["test"]["attacked"].astype(bool)
    auc_pt = {}
    for name in S["detectors"]:
        st = S["test"][name]; fin = np.isfinite(st)
        auc_pt[name] = roc_auc(st[fin], att_t[fin])
    # window-level AUC via CUSUM maxg (k chosen) over windows vs clean blocks on test
    auc_win = {}
    multiFAR = []
    for name in S["detectors"]:
        clean_v = (y_v == 0) & both_v[name]
        cu = partB[name]["cusum"]
        st_std = cu["st_std"]; k = cu["k"]
        win_t, blk_t = segments(S["tlab"])
        pos = np.array([np.max(cusum1(st_std[w["rows"]], k)) for w in win_t])
        neg = np.array([np.max(cusum1(st_std[b], k)) for b in blk_t])
        auc_win[name] = roc_auc(np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        # multi-FAR: per-timestep attacked detection (raw score) + CUSUM window detection
        sv = cu["sv_std"]; cvmask = clean_v
        for far in FAR_TARGETS:
            # per-timestep threshold on raw standardized score
            pt_thr = float(np.quantile(sv[cvmask & np.isfinite(sv)], 1 - far))
            fin = np.isfinite(st_std)
            realized = float(np.mean(st_std[fin & ~att_t] > pt_thr))
            step_det = float(np.mean(st_std[fin & att_t] > pt_thr))
            # CUSUM window detection at this far (h from clean val blocks)
            negb = np.array([np.max(cusum1(sv[b], k)) for b in segments(S["vlab"])[1]])
            h = float(np.quantile(negb, 1 - far))
            wd = int(np.sum([np.any(cusum1(st_std[w["rows"]], k) > h) for w in win_t]))
            multiFAR.append({"detector": name, "far_target": far, "realized_clean_FAR": realized,
                             "attacked_step_detection": step_det, "cusum_windows_detected": wd,
                             "cusum_window_rate": wd / len(win_t)})
    pd.DataFrame(multiFAR).to_csv(OUT / "detect_v4_multiFAR.csv", index=False)

    # old (v3) vs improved summary at 1%
    v3 = json.load(open(OUT / "detect_fusion_v3_summary.json"))
    summary = {"lr": S["lr"], "per_timestep_AUC": auc_pt, "window_AUC_cusum": auc_win,
               "partB": {n: {kk: partB[n]["cusum"][kk] for kk in
                             ("k", "h", "window_rate", "windows_detected", "clean_block_FAR", "median_delay",
                              "val_window_AUC")} for n in S["detectors"]},
               "meanwin": {n: partB[n]["meanwin"] for n in S["detectors"]},
               "perchannel_bonferroni": {kk: pc["bonferroni"][kk] for kk in ("window_rate","windows_detected","clean_block_FAR","clean_timestep_FAR","localization_acc","loc_hits","loc_tot")},
               "perchannel_tuned1pct": {kk: pc["tuned_1pct"][kk] for kk in ("window_rate","windows_detected","clean_block_FAR","clean_timestep_FAR","localization_acc","loc_hits","loc_tot")},
               "perchannel_by_family_bonf": by_family(pc["bonferroni"]["win_df"]),
               "perchannel_by_family_tuned": by_family(pc["tuned_1pct"]["win_df"]),
               "partD": partD,
               "v3_window_AUC": v3.get("window_AUC"), "v3_per_timestep_AUC": v3.get("per_timestep_AUC"),
               "cusum_by_family": {n: by_family(partB[n]["cusum"]["win_df"]) for n in S["detectors"]}}
    json.dump(summary, open(OUT / "detect_v4_summary.json", "w"), indent=2, default=float)

    print("\n[E] per-timestep AUC:", {n: round(auc_pt[n], 4) for n in S["detectors"]})
    print("[E] window AUC (CUSUM):", {n: round(auc_win[n], 4) for n in S["detectors"]})
    print("[E] v3 window AUC (mean rule):", v3.get("window_AUC"))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(multiFAR).to_string(index=False))


if __name__ == "__main__":
    main()
