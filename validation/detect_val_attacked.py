"""
Part A -- physics scores for the ATTACKED val timesteps (the one-time
estimator pass this task permits; the model stays frozen at
estimator-v5-hfix and is run nowhere else).

Runs the frozen estimator over the 1,580 attacked val rows of
dataset/fdia_v1/val_z.csv -> val_residuals_attacked.csv (J, maxLNR).
Then assembles val_scores.csv over ALL val timesteps:
  bdd_score = max(J/J99_val, maxLNR/LNR99_val)   [clean from the saved
              val_residuals.csv, attacked from the new pass]
  ae_score  = the AE s_max score (saved detect_ae_val_pertimestep.csv;
              AE reads z, already covers clean+attacked val)
columns: timestamp, attacked, bdd_score, ae_score.

Run: python -m validation.detect_val_attacked
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from estimation.bad_data_detection import compute_chi2_and_lnr

BASE_DIR = Path(__file__).resolve().parent.parent
FDIA = BASE_DIR / "dataset" / "fdia_v1"
CORPUS = BASE_DIR / "simulation" / "phase6_corpus"
OUT = BASE_DIR / "validation"
THRESH = BASE_DIR / "config" / "bdd_thresholds_val1pct.json"
CHECKPOINT = 400


def main():
    vz = pd.read_csv(FDIA / "val_z.csv")
    vlab = pd.read_csv(FDIA / "val_labels.csv")
    vcorp = pd.read_csv(CORPUS / "val_telemetry.csv", usecols=["timestamp", "real_timestamp"])
    assert (vz["timestamp"].values == vcorp["timestamp"].values).all()
    z_cols = [c for c in vz.columns if c != "timestamp"]
    att_rows = vlab.index[vlab["attacked"] == 1].tolist()
    print(f"attacked val timesteps: {len(att_rows)}", flush=True)

    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    real_ts = vcorp["real_timestamp"].tolist()

    recs = []
    t0 = time.time()
    for n, i in enumerate(att_rows):
        zr = {c: float(vz.iat[i, vz.columns.get_loc(c)]) for c in z_cols}
        ts = str(real_ts[i])
        res = est.estimate(zr, ts)
        st = compute_chi2_and_lnr(est, res, zr, ts)
        recs.append({"row": i, "timestamp": int(vz.iat[i, 0]), "converged": res.converged,
                     "iterations": res.iterations, "J": st["J"], "max_lnr": st["lnr"]})
        if (n + 1) % CHECKPOINT == 0:
            pd.DataFrame(recs).to_csv(OUT / "val_residuals_attacked.csv", index=False)
            el = time.time() - t0
            print(f"  {n+1}/{len(att_rows)}  {el:.0f}s ({el/(n+1):.3f}s/row)", flush=True)
    att_df = pd.DataFrame(recs)
    att_df.to_csv(OUT / "val_residuals_attacked.csv", index=False)
    print(f"A done: {len(att_df)} attacked val rows, converged {att_df['converged'].mean()*100:.2f}%, "
          f"{time.time()-t0:.0f}s", flush=True)

    # assemble val_scores.csv over ALL val
    th = json.load(open(THRESH))
    j99v, lnr99v = th["J99"], th["LNR99"]
    clean = pd.read_csv(OUT / "val_residuals.csv")[["row", "J", "max_lnr"]]
    both = pd.concat([clean, att_df[["row", "J", "max_lnr"]]], ignore_index=True).sort_values("row")
    bdd = np.maximum(both["J"].to_numpy() / j99v, both["max_lnr"].to_numpy() / lnr99v)
    bdd_by_row = dict(zip(both["row"].astype(int), bdd))
    ae = pd.read_csv(OUT / "detect_ae_val_pertimestep.csv")
    ae_by_row = dict(zip(ae["row"].astype(int), ae["score"].astype(float)))

    n = len(vlab)
    out = pd.DataFrame({"timestamp": vz["timestamp"].values, "attacked": vlab["attacked"].values,
                        "bdd_score": [bdd_by_row.get(i, np.nan) for i in range(n)],
                        "ae_score": [ae_by_row.get(i, np.nan) for i in range(n)]})
    out.to_csv(OUT / "val_scores.csv", index=False)
    nb = out["bdd_score"].notna().sum()
    na = out["ae_score"].notna().sum()
    print(f"val_scores.csv: {n} rows, bdd non-null {nb}, ae non-null {na} "
          f"(ae NaN for first W-1 rows with no history)", flush=True)
    print(f"bdd_score attacked mean {out.loc[out.attacked==1,'bdd_score'].mean():.3f} "
          f"vs clean {out.loc[out.attacked==0,'bdd_score'].mean():.3f}; "
          f"ae_score attacked mean {out.loc[out.attacked==1,'ae_score'].mean():.3f} "
          f"vs clean {out.loc[out.attacked==0,'ae_score'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
