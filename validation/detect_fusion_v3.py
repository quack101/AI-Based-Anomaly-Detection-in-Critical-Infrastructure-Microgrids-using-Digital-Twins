"""
Parts B-D of the detector comparison (numpy only).

Inputs (all saved; estimator not run here):
  validation/val_scores.csv                 -- timestamp, attacked, bdd_score, ae_score (ALL val)
  validation/detect_phys_test_scores.csv    -- bdd_score on all test rows
  validation/detect_ae_test_pertimestep.csv -- ae score on test (t>=W-1)
  dataset/fdia_v1/{val,test}_labels.csv

B: logistic regression on VAL, features [bdd_score, ae_score] standardized
   by CLEAN-VAL mean/std, label=attacked. fusion_LR test score = logistic
   output; threshold at 99th pct over clean val (1% val FAR).
C: window score = trailing mean of the per-timestep score over L steps
   (L in {15,60}, picked by val window-AUC per detector); window threshold
   = 99th pct of non-overlapping clean-val L-block means (1% window FAR);
   a test attack window is detected if its trailing-mean crosses the
   threshold at any step inside it.
D: at per-timestep clean-val FAR targets {0.5,1,2,5}% report attacked-step
   detection, window-mean detection, realized clean test FAR, for bdd / ae /
   fusion_OR / fusion_LR; plus per-timestep AUC and window-level AUC.

Run: python -m validation.detect_fusion_v3
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import roc_auc, CLEAN_BLOCK

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "validation"
FDIA = BASE_DIR / "dataset" / "fdia_v1"
W = 10
L_CHOICES = [15, 60]
FAR_TARGETS = [0.005, 0.01, 0.02, 0.05]


def logistic_fit(X, y, iters=4000, lr=0.5, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        gw = X.T @ (p - y) / n + l2 * w
        gb = np.mean(p - y)
        w -= lr * gw
        b -= lr * gb
    return w, b


def trailing_mean(s, L):
    """Trailing mean over L steps; NaN where the window is not fully
    finite. s is a per-row score array (NaN allowed at the head)."""
    n = len(s)
    out = np.full(n, np.nan)
    finite = np.isfinite(s)
    csum = np.zeros(n + 1)
    cfin = np.zeros(n + 1)
    sfill = np.where(finite, s, 0.0)
    csum[1:] = np.cumsum(sfill)
    cfin[1:] = np.cumsum(finite)
    for t in range(L - 1, n):
        if cfin[t + 1] - cfin[t + 1 - L] == L:
            out[t] = (csum[t + 1] - csum[t + 1 - L]) / L
        else:
            out[t] = np.nan
    return out


def clean_block_scores(score_tm, attacked, L):
    """Window score (the trailing mean, = mean over the preceding L steps)
    at the END of each non-overlapping L-block within contiguous clean
    stretches. One value per block; its 99th percentile is the 1%
    window-FAR threshold."""
    att = attacked.astype(bool)
    n = len(score_tm)
    vals = []
    i = 0
    while i < n:
        if att[i]:
            i += 1
            continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j - L + 1, L):
            v = score_tm[b + L - 1]
            if np.isfinite(v):
                vals.append(float(v))
        i = j
    return np.array(vals)


def window_detect(score_tm, labels, thr):
    """Detection per attack window = trailing-mean crosses thr at any
    attacked step; window-FAR = clean non-overlapping L-blocks (of the
    trailing mean) crossing thr. Returns (window_table, far)."""
    att = labels["attacked"].values.astype(bool)
    rows = []
    for wid, g in labels[att].groupby("window_id"):
        idx = [r for r in g.index if np.isfinite(score_tm[r])]
        det = bool(np.any(score_tm[idx] > thr)) if idx else False
        r0 = g.iloc[0]
        rows.append({"window_id": int(wid), "type": r0["type"], "class": r0["class"], "level": r0["level"],
                     "profile": r0["profile"], "duration": int(r0["duration"]), "detected": det})
    return pd.DataFrame(rows)


def window_far(score_tm, labels, L, thr):
    att = labels["attacked"].values.astype(bool)
    n = len(score_tm)
    nb = fb = 0
    i = 0
    while i < n:
        if att[i]:
            i += 1; continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j - L + 1, L):
            blk = score_tm[b:b + L]
            if np.all(np.isfinite(blk)):
                nb += 1
                if np.any(blk > thr):
                    fb += 1
        i = j
    return fb / nb if nb else float("nan"), fb, nb


def window_auc(score_tm, labels, L):
    att = labels["attacked"].values.astype(bool)
    pos = [float(np.nanmax(score_tm[[r for r in g.index if np.isfinite(score_tm[r])] or [g.index[0]]]))
           for _, g in labels[att].groupby("window_id")]
    neg = clean_block_means_tm(score_tm, att, L)
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return roc_auc(s, y)


def clean_block_means_tm(score_tm, att, L):
    n = len(score_tm)
    out = []
    i = 0
    while i < n:
        if att[i]:
            i += 1; continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j - L + 1, L):
            blk = score_tm[b:b + L]
            if np.all(np.isfinite(blk)):
                out.append(float(blk.max()))
        i = j
    return np.array(out)


def load():
    vs = pd.read_csv(OUT / "val_scores.csv")
    vlab = pd.read_csv(FDIA / "val_labels.csv")
    tlab = pd.read_csv(FDIA / "test_labels.csv")
    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")
    ae = pd.read_csv(OUT / "detect_ae_test_pertimestep.csv")
    n_te = len(tlab)
    bdd_te = np.full(n_te, np.nan); ae_te = np.full(n_te, np.nan)
    for r, s in zip(phys["row"].astype(int), phys["bdd_score"].astype(float)):
        bdd_te[r] = s
    for r, s in zip(ae["row"].astype(int), ae["score"].astype(float)):
        ae_te[r] = s
    return vs, vlab, tlab, bdd_te, ae_te


def far_threshold(clean_scores, far):
    return float(np.quantile(clean_scores[np.isfinite(clean_scores)], 1 - far))




def main():
    vs, vlab, tlab, bdd_te, ae_te = load()
    y_val = vs["attacked"].to_numpy().astype(int)
    both_val = np.isfinite(vs["bdd_score"]) & np.isfinite(vs["ae_score"])
    clean_val = both_val & (y_val == 0)

    # ---- feature standardization by CLEAN-VAL stats
    feats = ["bdd_score", "ae_score"]
    mu = vs.loc[clean_val, feats].mean().to_numpy()
    sd = vs.loc[clean_val, feats].std().to_numpy() + 1e-12
    Xval = ((vs.loc[both_val, feats].to_numpy() - mu) / sd)
    yv = y_val[both_val.to_numpy()]
    w, b = logistic_fit(Xval, yv)
    p_val = 1.0 / (1.0 + np.exp(-(Xval @ w + b)))
    auc_lr_val = roc_auc(p_val, yv)
    print(f"Part B LR weights (standardized): bdd {w[0]:.4f}, ae {w[1]:.4f}, bias {b:.4f}; val AUC {auc_lr_val:.4f}",
          flush=True)

    # ---- test per-timestep scores for each detector
    n_te = len(tlab)
    att_te = tlab["attacked"].to_numpy().astype(bool)
    zb = (bdd_te - mu[0]) / sd[0]
    za = (ae_te - mu[1]) / sd[1]
    scores_te = {
        "bdd": bdd_te,
        "ae": ae_te,
        "fusion_OR": np.maximum(zb, za),
        "fusion_LR": 1.0 / (1.0 + np.exp(-(zb * w[0] + za * w[1] + b))),
    }
    # clean-val per-timestep scores (same transforms) for thresholds
    zb_v = (vs["bdd_score"].to_numpy() - mu[0]) / sd[0]
    za_v = (vs["ae_score"].to_numpy() - mu[1]) / sd[1]
    scores_val = {
        "bdd": vs["bdd_score"].to_numpy(),
        "ae": vs["ae_score"].to_numpy(),
        "fusion_OR": np.maximum(zb_v, za_v),
        "fusion_LR": 1.0 / (1.0 + np.exp(-(zb_v * w[0] + za_v * w[1] + b))),
    }
    cvmask = clean_val.to_numpy()

    # ---- Part C: pick L per detector by val window-AUC, calibrate 1% window-FAR, report
    partC = []
    chosen_L = {}
    for name in scores_te:
        sv = scores_val[name]
        aucs = {}
        for L in L_CHOICES:
            tm_v = trailing_mean(sv, L)
            aucs[L] = window_auc(tm_v, vlab, L)
        L = max(aucs, key=lambda k: (aucs[k] if np.isfinite(aucs[k]) else -1))
        chosen_L[name] = L
        # window threshold: 99th pct of non-overlapping clean-val L-block means of trailing mean
        tm_v = trailing_mean(sv, L)
        cb = clean_block_scores(tm_v, vlab["attacked"].to_numpy(), L)
        wthr = float(np.quantile(cb, 0.99))
        st = scores_te[name]
        tm_t = trailing_mean(st, L)
        win = window_detect(tm_t, tlab, wthr)
        far, fb, nb = window_far(tm_t, tlab, L, wthr)
        # any-step rule at 1% per-timestep (for comparison)
        pt_thr = far_threshold(sv[cvmask], 0.01)
        anystep = window_detect_anystep(st, tlab, pt_thr)
        partC.append({"detector": name, "L": L, "val_window_AUC_L15": aucs.get(15), "val_window_AUC_L60": aucs.get(60),
                      "winmean_detected": int(win["detected"].sum()), "winmean_rate": float(win["detected"].mean()),
                      "winmean_FAR": far, "winmean_FA_blocks": fb, "winmean_blocks": nb,
                      "anystep_detected": int(anystep["detected"].sum()), "anystep_rate": float(anystep["detected"].mean())})
        win.to_csv(OUT / f"detect_v3_winmean_{name}.csv", index=False)
    pc = pd.DataFrame(partC)
    pc.to_csv(OUT / "detect_v3_partC.csv", index=False)
    print("\nPart C window-mean vs any-step detection:")
    print(pc[["detector", "L", "winmean_detected", "winmean_rate", "winmean_FAR",
              "anystep_detected", "anystep_rate"]].to_string(index=False))

    # ---- per-timestep AUC and window-level AUC (test)
    aucs_pt = {name: roc_auc(scores_te[name][np.isfinite(scores_te[name])],
                             att_te[np.isfinite(scores_te[name])]) for name in scores_te}
    aucs_win = {}
    for name in scores_te:
        L = chosen_L[name]
        tm_t = trailing_mean(scores_te[name], L)
        aucs_win[name] = window_auc(tm_t, tlab, L)

    # ---- Part D: multi-FAR
    rows = []
    for name in scores_te:
        L = chosen_L[name]
        sv = scores_val[name]
        st = scores_te[name]
        tm_t = trailing_mean(st, L)
        for far in FAR_TARGETS:
            thr = far_threshold(sv[cvmask], far)
            fin = np.isfinite(st)
            realized = float(np.mean(st[fin & ~att_te] > thr))
            det = float(np.mean(st[fin & att_te] > thr))
            win = window_detect(tm_t, tlab, thr)
            rows.append({"detector": name, "far_target": far, "realized_clean_test_FAR": realized,
                         "attacked_step_detection": det, "winmean_detected": int(win["detected"].sum()),
                         "winmean_rate": float(win["detected"].mean()),
                         "per_timestep_AUC": aucs_pt[name], "window_AUC": aucs_win[name], "L": L})
    pd_tab = pd.DataFrame(rows)
    pd_tab.to_csv(OUT / "detect_v3_multiFAR.csv", index=False)
    json.dump({"lr_weights_std": {"bdd": float(w[0]), "ae": float(w[1]), "bias": float(b)},
               "lr_val_AUC": float(auc_lr_val), "chosen_L": chosen_L,
               "per_timestep_AUC": {k: float(v) for k, v in aucs_pt.items()},
               "window_AUC": {k: float(v) for k, v in aucs_win.items()}},
              open(OUT / "detect_fusion_v3_summary.json", "w"), indent=2)
    pd.set_option("display.width", 220)
    print("\nthreshold-free AUC (test):")
    for name in scores_te:
        print(f"  {name}: per-timestep {aucs_pt[name]:.4f}, window-level {aucs_win[name]:.4f}")
    print("\nPart D multi-FAR (attacked-step detection / window-mean rate / realized clean FAR):")
    for far in FAR_TARGETS:
        sub = pd_tab[pd_tab["far_target"] == far]
        print(f"\n FAR target {far*100:.1f}%:")
        print(sub[["detector", "realized_clean_test_FAR", "attacked_step_detection",
                   "winmean_detected", "winmean_rate"]].to_string(index=False))


def window_detect_anystep(score_te, labels, thr):
    att = labels["attacked"].values.astype(bool)
    rows = []
    for wid, g in labels[att].groupby("window_id"):
        idx = [r for r in g.index if np.isfinite(score_te[r])]
        det = bool(np.any(score_te[idx] > thr)) if idx else False
        rows.append({"window_id": int(wid), "detected": det})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
