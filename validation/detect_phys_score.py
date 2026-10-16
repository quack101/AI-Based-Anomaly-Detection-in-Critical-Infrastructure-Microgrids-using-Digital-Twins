"""
Part A -- physics (BDD) baseline as a SINGLE per-timestep score.

bdd_score_t = max(J_t / J99_val, maxLNR_t / LNR99_val), with J99_val /
LNR99_val the val-1% thresholds in config/bdd_thresholds_val1pct.json.
Threshold = 99th percentile of bdd_score over CLEAN val (exactly 1% val
FAR for this single score). Applied to test.

Reuses the saved residual logs ONLY -- the estimator is not re-run.
Clean val residuals: validation/val_residuals.csv (attacked==0 only).
Test residuals: validation/fdia_v1_bdd_test_v5_pertimestep.csv (all test).

Run: python -m validation.detect_phys_score
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.detect_eval import evaluate_flags, threshold_99, roc_auc

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "validation"
FDIA = BASE_DIR / "dataset" / "fdia_v1"
THRESH = BASE_DIR / "config" / "bdd_thresholds_val1pct.json"


def bdd_score(J, lnr, j99, lnr99):
    return np.maximum(np.asarray(J) / j99, np.asarray(lnr) / lnr99)


def main():
    th = json.load(open(THRESH))
    j99, lnr99 = th["J99"], th["LNR99"]
    val = pd.read_csv(OUT / "val_residuals.csv")              # clean val only
    test = pd.read_csv(OUT / "fdia_v1_bdd_test_v5_pertimestep.csv")
    tlab = pd.read_csv(FDIA / "test_labels.csv")

    s_val = bdd_score(val["J"], val["max_lnr"], j99, lnr99)   # all clean
    thr = threshold_99(s_val)
    s_test = bdd_score(test["J"], test["max_lnr"], j99, lnr99)
    test = test.assign(bdd_score=s_test)
    test[["row", "timestamp", "J", "max_lnr", "bdd_score"]].to_csv(OUT / "detect_phys_test_scores.csv", index=False)

    flagged = {int(r): bool(s > thr) for r, s in zip(test["row"].astype(int), s_test)}
    scored = set(test["row"].astype(int))
    res = evaluate_flags(tlab, flagged, scored_rows=scored, name="BDD_single")
    # per-timestep AUC on test (attacked vs clean), using bdd_score
    lab_by_row = tlab["attacked"].to_numpy()
    y = np.array([lab_by_row[r] for r in test["row"].astype(int)])
    auc = roc_auc(s_test, y)

    res["window_table"].to_csv(OUT / "detect_phys_windows.csv", index=False)
    res["breakdown"].to_csv(OUT / "detect_phys_breakdown.csv", index=False)
    summ = {k: v for k, v in res.items() if k not in ("window_table", "breakdown")}
    summ.update({"threshold": float(thr), "clean_val_FAR_target": 0.01, "per_timestep_AUC": float(auc),
                 "J99_val": j99, "LNR99_val": lnr99})
    json.dump(summ, open(OUT / "detect_phys_summary.json", "w"), indent=2)

    print(f"BDD single score: threshold {thr:.4f} (99th pct clean val), per-timestep AUC {auc:.4f}")
    for k in ["clean_test_FAR", "attacked_timestep_detection", "windows_detected", "windows",
              "window_detection_rate", "median_delay", "clean_60block_FA", "clean_60block_total"]:
        print(f"  {k}: {summ[k]}")
    pd.set_option("display.width", 200)
    print(res["breakdown"].to_string(index=False))


if __name__ == "__main__":
    main()
