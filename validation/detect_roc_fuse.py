"""
Part C (ROC/AUC) and Part D (fusion) for the FDIA v1 detectors.

Inputs (all already produced; the estimator is NOT re-run):
  validation/detect_phys_test_scores.csv   -- bdd_score on all test rows
  validation/detect_ae_test_pertimestep.csv-- ae score on test (t>=W-1)
  validation/detect_ae_val_pertimestep.csv -- ae score on val (+ attacked)
  validation/val_residuals.csv             -- CLEAN val J/LNR (physics)
  config/bdd_thresholds_val1pct.json       -- J99_val, LNR99_val
  dataset/fdia_v1/{val,test}_labels.csv

Part C: per-timestep ROC + AUC for bdd_score and ae_score on test, and a
window-detection vs window-false-alarm curve for each. Saved as CSVs.

Part D: fusion of [bdd_score, ae_score].
  Rule 1 (OR): flag if bdd_score > bdd_1%thr OR ae_score > ae_1%thr.
  Rule 2 (logistic regression on VAL): REQUIRES bdd_score on attacked
  val timesteps. val_residuals.csv holds CLEAN val only, and the rules
  forbid re-running the estimator, so the attacked-val physics score is
  unavailable -- Rule 2 cannot be trained. This is detected and reported;
  the script STOPs Rule 2 rather than fabricate or re-run the estimator.

Run: python -m validation.detect_roc_fuse
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import evaluate_flags, threshold_99, roc_auc, roc_curve, CLEAN_BLOCK

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "validation"
FDIA = BASE_DIR / "dataset" / "fdia_v1"
THRESH = BASE_DIR / "config" / "bdd_thresholds_val1pct.json"
W = 10


def window_fa_curve(labels, score_by_row, n_points=100):
    """Sweep threshold; return (threshold, window_detection_rate,
    window_false_alarm_rate). Window detection = attacked windows with any
    attacked step > t. Window FA = clean 60-step blocks with any step > t."""
    att = labels["attacked"].values.astype(bool)
    n = len(labels)
    scored = np.array([r in score_by_row for r in range(n)])
    s = np.array([score_by_row.get(r, np.nan) for r in range(n)])
    finite = s[np.isfinite(s)]
    lo, hi = float(finite.min()), float(finite.max())
    thr = np.linspace(lo, hi, n_points)
    # attacked windows
    win_rows = {}
    for wid, g in labels[att].groupby("window_id"):
        win_rows[wid] = [r for r in g.index if scored[r]]
    # clean 60-step blocks
    blocks = []
    i = 0
    while i < n:
        if att[i]:
            i += 1; continue
        j = i
        while j < n and not att[j]:
            j += 1
        for b in range(i, j, CLEAN_BLOCK):
            blk = [r for r in range(b, min(b + CLEAN_BLOCK, j)) if scored[r]]
            if blk:
                blocks.append(blk)
        i = j
    rows = []
    for t in thr:
        det = np.mean([any(s[r] > t for r in rs) for rs in win_rows.values()]) if win_rows else np.nan
        fa = np.mean([any(s[r] > t for r in bl) for bl in blocks]) if blocks else np.nan
        rows.append({"threshold": float(t), "window_detection_rate": float(det), "window_fa_rate": float(fa)})
    return pd.DataFrame(rows)


def main():
    th = json.load(open(THRESH))
    j99v, lnr99v = th["J99"], th["LNR99"]
    tlab = pd.read_csv(FDIA / "test_labels.csv")
    vlab = pd.read_csv(FDIA / "val_labels.csv")

    phys = pd.read_csv(OUT / "detect_phys_test_scores.csv")           # row, bdd_score (all test)
    ae_te = pd.read_csv(OUT / "detect_ae_test_pertimestep.csv")       # row, score (t>=W-1)
    ae_va = pd.read_csv(OUT / "detect_ae_val_pertimestep.csv")        # row, score, attacked
    valres = pd.read_csv(OUT / "val_residuals.csv")                   # clean val J/LNR

    # align test scores by row
    bdd_te = dict(zip(phys["row"].astype(int), phys["bdd_score"].astype(float)))
    ae_te_s = {int(r): float(s) for r, s in zip(ae_te["row"], ae_te["score"]) if np.isfinite(s)}
    common = sorted(set(bdd_te) & set(ae_te_s))                       # rows scored by BOTH (t>=W-1)
    y = tlab["attacked"].to_numpy()

    # ---- Part C: ROC/AUC on test
    yb = np.array([y[r] for r in sorted(bdd_te)])
    auc_bdd = roc_auc([bdd_te[r] for r in sorted(bdd_te)], yb)
    ya = np.array([y[r] for r in sorted(ae_te_s)])
    auc_ae = roc_auc([ae_te_s[r] for r in sorted(ae_te_s)], ya)
    roc_curve([bdd_te[r] for r in sorted(bdd_te)], yb).to_csv(OUT / "detect_roc_bdd.csv", index=False)
    roc_curve([ae_te_s[r] for r in sorted(ae_te_s)], ya).to_csv(OUT / "detect_roc_ae.csv", index=False)
    window_fa_curve(tlab, bdd_te).to_csv(OUT / "detect_wincurve_bdd.csv", index=False)
    window_fa_curve(tlab, ae_te_s).to_csv(OUT / "detect_wincurve_ae.csv", index=False)
    print(f"Part C per-timestep AUC: BDD {auc_bdd:.4f}  AE {auc_ae:.4f}")

    # ---- Part D Rule 1: OR at each detector's own 1% clean-val point
    bdd_thr = threshold_99(np.maximum(valres["J"] / j99v, valres["max_lnr"] / lnr99v))
    ae_thr = float(json.load(open(OUT / "detect_ae_summary.json"))["threshold"])
    scored = set(common)
    def metrics_for(flag_fn, name):
        flagged = {r: bool(flag_fn(r)) for r in scored}
        m = evaluate_flags(tlab, flagged, scored_rows=scored, name=name)
        yy = np.array([y[r] for r in sorted(scored)])
        # fusion score for AUC (OR uses a combined score = max of standardized ratios)
        return m

    or_flag = lambda r: (bdd_te[r] > bdd_thr) or (ae_te_s[r] > ae_thr)
    m_or = metrics_for(or_flag, "fusion_OR")
    # a combined OR score for AUC: max(bdd/bdd_thr, ae/ae_thr)
    s_or = np.array([max(bdd_te[r] / bdd_thr, ae_te_s[r] / ae_thr) for r in sorted(scored)])
    auc_or = roc_auc(s_or, np.array([y[r] for r in sorted(scored)]))

    # BDD-alone and AE-alone restricted to the SAME common rows, for a fair table
    m_bdd = evaluate_flags(tlab, {r: bdd_te[r] > bdd_thr for r in scored}, scored_rows=scored, name="BDD")
    m_ae = evaluate_flags(tlab, {r: ae_te_s[r] > ae_thr for r in scored}, scored_rows=scored, name="AE")
    auc_bdd_c = roc_auc([bdd_te[r] for r in sorted(scored)], np.array([y[r] for r in sorted(scored)]))
    auc_ae_c = roc_auc([ae_te_s[r] for r in sorted(scored)], np.array([y[r] for r in sorted(scored)]))

    # ---- Part D Rule 2: logistic regression on VAL -- feasibility check
    # needs bdd_score on ALL val (incl attacked); val_residuals is clean-only.
    val_attacked = int((vlab["attacked"] == 1).sum())
    have_attacked_val_bdd = False   # val_residuals.csv is clean val only
    rule2_status = ("BLOCKED: logistic-regression fusion needs bdd_score on the "
                    f"{val_attacked} attacked val timesteps; val_residuals.csv holds clean val only and "
                    "the rules forbid re-running the estimator. Rule 2 not trained.")

    def row(m, auc, name):
        return {"detector": name, "clean_test_FAR": m["clean_test_FAR"],
                "attacked_timestep_detection": m["attacked_timestep_detection"],
                "windows_detected": m["windows_detected"], "window_detection_rate": m["window_detection_rate"],
                "median_delay": m["median_delay"], "per_timestep_AUC": auc,
                "clean_60block_FA": m["clean_60block_FA"]}
    table = pd.DataFrame([row(m_bdd, auc_bdd_c, "BDD"), row(m_ae, auc_ae_c, "AE"),
                          row(m_or, auc_or, "fusion_OR")])
    table.to_csv(OUT / "detect_compare_v2_table.csv", index=False)
    for m, tag in [(m_bdd, "BDD"), (m_ae, "AE"), (m_or, "fusion_OR")]:
        m["breakdown"].to_csv(OUT / f"detect_compare_v2_breakdown_{tag}.csv", index=False)

    json.dump({"part_c_auc": {"BDD": auc_bdd, "AE": auc_ae},
               "common_scored_rows": len(common),
               "thresholds": {"bdd_1pct": bdd_thr, "ae_1pct": ae_thr},
               "fusion_OR": {k: m_or[k] for k in ["clean_test_FAR", "attacked_timestep_detection",
                             "windows_detected", "window_detection_rate", "median_delay", "clean_60block_FA"]},
               "fusion_OR_AUC": auc_or, "rule2_status": rule2_status},
              open(OUT / "detect_roc_fuse_summary.json", "w"), indent=2)

    pd.set_option("display.width", 220)
    print("\nComparison table (matched 1% val FAR; common rows t>=W-1):")
    print(table.to_string(index=False))
    print(f"\nthresholds: bdd_1% {bdd_thr:.4f}, ae_1% {ae_thr:.4f}")
    print("Part D Rule 2:", rule2_status)


if __name__ == "__main__":
    main()
