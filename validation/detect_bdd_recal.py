"""
Part A -- recalibrate the residual (BDD) baseline to a 1% per-timestep
false-alarm operating point on CLEAN val.

A1: run the frozen estimator (estimator-v5-hfix defaults) over the clean
    val timesteps (attacked == 0) -> val_residuals.csv (per-step J, LNR).
A2: J99/LNR99 = 99th percentile of those clean-val residuals; save to
    config/bdd_thresholds_val1pct.json (old v5 values reported too).
A3: re-score the ALREADY-SAVED test per-timestep residual log
    (validation/fdia_v1_bdd_test_v5_pertimestep.csv -- the estimator is
    NOT re-run on test) with the new OR rule, via the common protocol.

z comes from dataset/fdia_v1/val_z.csv; the pseudo-bucket hour comes
from the corpus val split's own real_timestamp. Labels pick clean rows
and score only. One process.

Run: python -m validation.detect_bdd_recal
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr
from validation.detect_eval import evaluate_flags

BASE_DIR = Path(__file__).resolve().parent.parent
FDIA = BASE_DIR / "dataset" / "fdia_v1"
CORPUS = BASE_DIR / "simulation" / "phase6_corpus"
OUT = BASE_DIR / "validation"
V5_THRESH = BASE_DIR / "config" / "bdd_thresholds_v5.json"
NEW_THRESH = BASE_DIR / "config" / "bdd_thresholds_val1pct.json"
TEST_LOG = OUT / "fdia_v1_bdd_test_v5_pertimestep.csv"
VAL_RESID = OUT / "val_residuals.csv"
CHECKPOINT = 500


def main():
    vz = pd.read_csv(FDIA / "val_z.csv")
    vlab = pd.read_csv(FDIA / "val_labels.csv")
    vcorp = pd.read_csv(CORPUS / "val_telemetry.csv", usecols=["timestamp", "real_timestamp"])
    assert (vz["timestamp"].values == vcorp["timestamp"].values).all()
    z_cols = [c for c in vz.columns if c != "timestamp"]
    clean_rows = vlab.index[vlab["attacked"] == 0].tolist()
    print(f"clean val timesteps: {len(clean_rows)} of {len(vlab)}", flush=True)

    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    real_ts = vcorp["real_timestamp"].tolist()

    recs = []
    t0 = time.time()
    for n, i in enumerate(clean_rows):
        zr = {c: float(vz.iat[i, vz.columns.get_loc(c)]) for c in z_cols}
        ts = str(real_ts[i])
        res = est.estimate(zr, ts)
        st = compute_chi2_and_lnr(est, res, zr, ts)
        recs.append({"row": i, "timestamp": int(vz.iat[i, 0]), "converged": res.converged,
                     "iterations": res.iterations, "J": st["J"], "max_lnr": st["lnr"]})
        if (n + 1) % CHECKPOINT == 0:
            pd.DataFrame(recs).to_csv(VAL_RESID, index=False)
            el = time.time() - t0
            print(f"  {n+1}/{len(clean_rows)}  {el:.0f}s ({el/(n+1):.3f}s/row)", flush=True)
    val_df = pd.DataFrame(recs)
    val_df.to_csv(VAL_RESID, index=False)
    print(f"A1 done: {len(val_df)} clean val rows, converged {val_df['converged'].mean()*100:.2f}%, "
          f"{time.time()-t0:.0f}s", flush=True)

    j99 = float(np.quantile(val_df["J"], 0.99))
    lnr99 = float(np.quantile(val_df["max_lnr"], 0.99))
    old = json.load(open(V5_THRESH))
    json.dump({"J99": j99, "LNR99": lnr99, "n": len(val_df),
               "source": "99th percentile of J / max-LNR over clean val timesteps (attacked==0), "
                         "dataset/fdia_v1/val_z.csv, estimator-v5-hfix defaults.",
               "operating_point": "1% per-timestep FAR on clean val, each score at its 99th percentile",
               "prev_v5": {"J99": old["J99"], "LNR99": old["LNR99"]}}, open(NEW_THRESH, "w"), indent=2)
    print(f"A2 thresholds: J99 {old['J99']:.4f} -> {j99:.4f};  LNR99 {old['LNR99']:.4f} -> {lnr99:.4f}", flush=True)

    # clean-val achieved FAR of the OR rule at these thresholds
    val_or = ((val_df["J"] > j99) | (val_df["max_lnr"] > lnr99)).mean()
    print(f"A2 clean-val OR-rule FAR at new thresholds: {val_or*100:.2f}% "
          f"(J alone {np.mean(val_df['J']>j99)*100:.2f}%, LNR alone {np.mean(val_df['max_lnr']>lnr99)*100:.2f}%)",
          flush=True)

    # A3: re-score the saved test log
    test = pd.read_csv(TEST_LOG)
    tlab = pd.read_csv(FDIA / "test_labels.csv")
    test_flag = ((test["J"] > j99) | (test["max_lnr"] > lnr99)).values
    flagged = {int(r): bool(f) for r, f in zip(test["row"].values, test_flag)}
    m = evaluate_flags(tlab, flagged, scored_rows=set(test["row"].astype(int)), name="BDD_val1pct")
    m["window_table"].to_csv(OUT / "detect_bdd_val1pct_windows.csv", index=False)
    m["breakdown"].to_csv(OUT / "detect_bdd_val1pct_breakdown.csv", index=False)
    summ = {k: v for k, v in m.items() if k not in ("window_table", "breakdown")}
    summ.update({"J99": j99, "LNR99": lnr99, "clean_val_OR_FAR": float(val_or)})
    json.dump(summ, open(OUT / "detect_bdd_val1pct_summary.json", "w"), indent=2)
    print("\nA3 BDD baseline @ val-1% operating point on test:")
    for k in ["clean_test_FAR", "attacked_timestep_detection", "windows_detected", "windows",
              "window_detection_rate", "median_delay", "clean_60block_FA", "clean_60block_total"]:
        print(f"  {k}: {summ[k]}")
    pd.set_option("display.width", 200)
    print(m["breakdown"].to_string(index=False))


if __name__ == "__main__":
    main()
