"""
Shared evaluation protocol for the FDIA v1 anomaly detectors (Part A
residual baseline, Part B GRU). Detectors read only telemetry z; labels
are used only to pick clean timesteps for calibration and to score.

Operating point: 1.0% per-timestep false-alarm rate, set on the CLEAN
val timesteps (label attacked == 0). For a single per-timestep score,
the threshold is the 99th percentile of the score over clean val; that
exact threshold is applied to test.

Metrics on test (evaluate_flags): per-timestep clean FAR; per-timestep
detection rate on attacked steps; window-level detection (a window is
detected if ANY of its attacked steps is flagged); median detection
delay (timesteps); window-level breakdown by type/class/level/profile/
duration; and, for context, the number of 60-step clean test blocks
with any flag.
"""

import numpy as np
import pandas as pd

CLEAN_BLOCK = 60


def threshold_99(clean_scores):
    return float(np.quantile(np.asarray(clean_scores, dtype=float), 0.99))


def _avg_ranks(a):
    """Average ranks (1-based) with tie handling."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j < n and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0  # average of 1-based ranks i+1..j
        i = j
    return ranks


def roc_auc(scores, y):
    """Mann-Whitney AUC; y==1 is the positive (attacked) class. NaN scores
    are dropped. Returns NaN if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _avg_ranks(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def roc_curve(scores, y, n_points=200):
    """Per-timestep ROC as a DataFrame (threshold, tpr, fpr). Thresholds
    swept over the observed score range."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    P, N = int(y.sum()), int((~y).sum())
    lo, hi = float(np.min(s)), float(np.max(s))
    thr = np.unique(np.concatenate([[lo - 1e-9], np.linspace(lo, hi, n_points), [hi + 1e-9]]))
    rows = []
    for t in thr:
        flag = s > t
        tp = int((flag & y).sum())
        fp = int((flag & ~y).sum())
        rows.append({"threshold": float(t), "tpr": tp / P if P else float("nan"),
                     "fpr": fp / N if N else float("nan")})
    return pd.DataFrame(rows)


def _window_table(att, flagged):
    """att: attacked test rows (DataFrame with window_id/type/class/level/
    profile/duration and a positional 'pos'); flagged: bool Series aligned
    to att.index."""
    rows, delays = [], []
    for wid, g in att.groupby("window_id"):
        fl = flagged.loc[g.index]
        detected = bool(fl.any())
        if detected:
            first_flag_pos = g.loc[fl[fl].index, "pos"].min()
            delays.append(int(first_flag_pos - g["pos"].min()))
        r0 = g.iloc[0]
        rows.append({"window_id": int(wid), "type": r0["type"], "class": r0["class"], "level": r0["level"],
                     "profile": r0["profile"], "duration": int(r0["duration"]),
                     "n_attacked": len(g), "n_flagged": int(fl.sum()), "detected": detected})
    return pd.DataFrame(rows), delays


def breakdown(win):
    out = [{"dimension": "ALL", "value": "all", "n": len(win), "detected": int(win["detected"].sum()),
            "detection_rate": float(win["detected"].mean()) if len(win) else float("nan")}]
    for dim in ["type", "class", "level", "profile", "duration"]:
        for val, g in win.groupby(dim):
            out.append({"dimension": dim, "value": val, "n": len(g), "detected": int(g["detected"].sum()),
                        "detection_rate": float(g["detected"].mean())})
    return pd.DataFrame(out)


def clean_block_fa(labels_sorted, flagged_by_pos):
    """Number of 60-step blocks of contiguous clean test timesteps with
    >= 1 flag, and the total number of such blocks."""
    att = labels_sorted["attacked"].values.astype(bool)
    flag = np.array([flagged_by_pos.get(i, False) for i in range(len(labels_sorted))])
    n_blocks = n_flagged_blocks = 0
    i = 0
    n = len(att)
    while i < n:
        if att[i]:
            i += 1
            continue
        j = i
        while j < n and not att[j]:
            j += 1
        # clean run [i, j); chunk into CLEAN_BLOCK blocks
        for b in range(i, j, CLEAN_BLOCK):
            blk = slice(b, min(b + CLEAN_BLOCK, j))
            n_blocks += 1
            if flag[blk].any():
                n_flagged_blocks += 1
        i = j
    return n_flagged_blocks, n_blocks


def evaluate_flags(labels, flagged_rows, scored_rows=None, name=""):
    """labels: the split's {split}_labels.csv (ordered by timestep).
    flagged_rows: set/dict of row-position -> bool for rows that were
    scored and flagged. scored_rows: the set of row positions actually
    scored (None = all rows in labels). Returns a metrics dict."""
    labels = labels.reset_index(drop=True)
    n = len(labels)
    scored = set(range(n)) if scored_rows is None else set(scored_rows)
    flagged_by_pos = {i: bool(flagged_rows.get(i, False)) for i in scored}

    att_mask = labels["attacked"].values.astype(bool)
    clean_scored = [i for i in scored if not att_mask[i]]
    att_scored = [i for i in scored if att_mask[i]]
    clean_far = float(np.mean([flagged_by_pos[i] for i in clean_scored])) if clean_scored else float("nan")
    att_det = float(np.mean([flagged_by_pos[i] for i in att_scored])) if att_scored else float("nan")

    att = labels[att_mask].copy()
    att["pos"] = att.index
    att = att[att["pos"].isin(scored)]
    flagged = pd.Series({i: flagged_by_pos[i] for i in att.index})
    win, delays = _window_table(att, flagged)
    bd = breakdown(win)
    fb, tb = clean_block_fa(labels, flagged_by_pos)
    return {"name": name, "n_scored": len(scored), "clean_scored": len(clean_scored), "att_scored": len(att_scored),
            "clean_test_FAR": clean_far, "attacked_timestep_detection": att_det,
            "windows": len(win), "windows_detected": int(win["detected"].sum()),
            "window_detection_rate": float(win["detected"].mean()) if len(win) else float("nan"),
            "median_delay": float(np.median(delays)) if delays else float("nan"),
            "clean_60block_FA": fb, "clean_60block_total": tb,
            "window_table": win, "breakdown": bd}
