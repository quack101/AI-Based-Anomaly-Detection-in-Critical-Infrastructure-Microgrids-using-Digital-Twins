"""
Phase 1 GATE -- verify streaming/detector_core.py reproduces the frozen
offline detector. No Kafka.

(1) 20 TEST timesteps: score_vector's J / maxLNR / bdd_score vs the
    committed per-timestep logs (validation/fdia_v1_bdd_test_v5_
    pertimestep.csv, detect_phys_test_scores.csv) -- independent scripts.
(2) score_vector's ae_mahalanobis / fusion_score / alarm at those 20 vs
    the offline pipeline recomputed over the FULL test the same way
    validation/detect_final.py did (committed bdd + AE Mahalanobis fit +
    fusion weights), and reproduce detect_final_summary.json's full-test
    point-maha (11/18) and fusion (12/18) window detection, AUCs and
    per-window det dicts.

Run: python -m streaming.verify_phase1
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from streaming.detector_core import DetectorCore, AE_W
from validation.detect_ae import W  # ==10
from validation.detect_mahalanobis import ae_errors, segments
from validation.detect_eval import roc_auc

BASE = Path(__file__).resolve().parent.parent
FDIA = BASE / "dataset" / "fdia_v1"
V = BASE / "validation"


def main():
    core = DetectorCore(registry_variant="multiphase")
    tz = pd.read_csv(FDIA / "test_z.csv")
    cols = core.cols
    Xte = tz[cols].to_numpy(float)
    corp = pd.read_csv(BASE / "simulation" / "phase6_corpus" / "test_telemetry.csv", usecols=["real_timestamp"])
    tlab = pd.read_csv(FDIA / "test_labels.csv")
    comm = pd.read_csv(V / "fdia_v1_bdd_test_v5_pertimestep.csv").set_index("row")
    bdd_comm = pd.read_csv(V / "detect_phys_test_scores.csv").set_index("row")["bdd_score"]

    att = tlab.index[tlab["attacked"] == 1].tolist()
    clean = [r for r in range(W - 1, len(tlab)) if tlab.iloc[r]["attacked"] == 0]
    rng = np.random.default_rng(0)
    rows = sorted(set([r for r in rng.choice([a for a in att if a >= W - 1], 10, replace=False)] +
                      [r for r in rng.choice(clean, 10, replace=False)]))

    recs = []
    for t in rows:
        window = Xte[t - W + 1:t + 1]
        out = core.score_vector(window, str(corp.iloc[t]["real_timestamp"]))
        recs.append({"row": t, **out, "J_comm": float(comm.loc[t, "J"]),
                     "lnr_comm": float(comm.loc[t, "max_lnr"]), "bdd_comm": float(bdd_comm.loc[t])})
    df = pd.DataFrame(recs)
    dJ = np.abs(df["J"] - df["J_comm"]); relJ = dJ / np.maximum(np.abs(df["J_comm"]), 1e-12)
    dL = np.abs(df["maxLNR"] - df["lnr_comm"]); dB = np.abs(df["bdd_score"] - df["bdd_comm"])
    print(f"(1) 20 test timesteps vs committed logs:")
    print(f"    J   max|abs| {dJ.max():.3e}  max|rel| {relJ.max():.3e}")
    print(f"    LNR max|abs| {dL.max():.3e}")
    print(f"    bdd max|abs| {dB.max():.3e}")

    # (2) full-test offline recompute (same as detect_final): committed bdd + AE maha + fusion
    J_all = comm["J"].reindex(range(len(tlab))).to_numpy()
    L_all = comm["max_lnr"].reindex(range(len(tlab))).to_numpy()
    bdd_all = np.maximum(J_all / core.J99, L_all / core.LNR99)
    Ete = ae_errors(core.P, core.mu_ae, core.sd_ae, core.r_c, Xte)
    d = Ete - core.mu
    maha_all = np.full(len(tlab), np.nan)
    fin = np.isfinite(Ete[:, 0])
    maha_all[fin] = np.einsum("ij,ij->i", d[fin] @ core.Minv, d[fin])
    zb = (bdd_all - core.muf[0]) / core.sdf[0]; zm = (maha_all - core.muf[1]) / core.sdf[1]
    fus_all = 1.0 / (1.0 + np.exp(-(zb * core.w_bdd + zm * core.w_maha + core.w_b)))

    # per-timestep match of core vs offline recompute at the 20 rows
    dM = np.abs(df["ae_mahalanobis"].to_numpy() - maha_all[df["row"]])
    dF = np.abs(df["fusion_score"].to_numpy() - fus_all[df["row"]])
    alarm_off = fus_all[df["row"]] > core.thr_f
    print(f"(2) core vs offline recompute at the 20 rows:")
    print(f"    maha   max|abs| {np.nanmax(dM):.3e}")
    print(f"    fusion max|abs| {np.nanmax(dF):.3e}")
    print(f"    alarms identical: {bool((df['alarm'].to_numpy() == alarm_off).all())}")

    # reproduce detect_final_summary window detection / AUC
    summ = json.load(open(V / "detect_final_summary.json"))
    att_t = tlab["attacked"].to_numpy()

    def windet(score, thr):
        wins, _ = segments(tlab); det = {}
        for w in wins:
            r = w["rows"][np.isfinite(score[w["rows"]])]
            det[w["wid"]] = bool(np.any(score[r] > thr)) if len(r) else False
        return det
    det_p = windet(maha_all, core.maha_thr); det_f = windet(fus_all, core.thr_f)
    auc_p = roc_auc(maha_all[fin], att_t[fin].astype(bool))
    auc_f = roc_auc(fus_all[fin], att_t[fin].astype(bool))
    off_p = {int(k): v for k, v in summ["point"]["det"].items()}
    off_f = {int(k): v for k, v in summ["fusion"]["det"].items()}
    print(f"(2) full-test reproduction of detect_final:")
    print(f"    point maha: win {sum(det_p.values())}/18 (committed 11)  AUC {auc_p:.6f} "
          f"(committed {summ['point']['per_ts_AUC']:.6f}) det-dict match {det_p == off_p}")
    print(f"    fusion:     win {sum(det_f.values())}/18 (committed 12)  AUC {auc_f:.6f} "
          f"(committed {summ['fusion']['per_ts_AUC']:.6f}) det-dict match {det_f == off_f}")

    df.to_csv(V / "verify_phase1_20rows.csv", index=False)
    gate = (relJ.max() <= 1e-6 and dB.max() <= 1e-6 and np.nanmax(dM) <= 1e-6 and np.nanmax(dF) <= 1e-6
            and bool((df["alarm"].to_numpy() == alarm_off).all())
            and sum(det_f.values()) == summ["fusion"]["win_det"] and det_f == off_f
            and abs(auc_f - summ["fusion"]["per_ts_AUC"]) <= 1e-6)
    print(f"\nGATE: {'PASS' if gate else 'FAIL'}")
    return gate


if __name__ == "__main__":
    main()
