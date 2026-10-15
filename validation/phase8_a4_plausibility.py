"""
A4 plausibility: how large is the kept-direction A4 attack
(attacks/a4_kept.py) relative to the clean readings it is added to?

Points 46, 94, 154; sizes 0.005, 0.01, 0.02 pu. The attack vectors are
re-created exactly as validation/phase8_a4_a5_scan_kept.py draws them
(per-point seed SEED+400+k, same rng call order including both
controls). For every channel with a != 0: a, the clean reading z,
|a|/|z| and |a|/sigma.

Run: python -m validation.phase8_a4_plausibility
"""

from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from attacks.a4_synthesis import random_sign_control
from attacks.a4_kept import kept_subspace_basis, synthesize_kept_attack, random_weighted_control
from validation import phase8_a4_a5_scan_kept as scan

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "validation" / "phase8_a4_plausibility.csv"
OUT_SUMMARY = BASE_DIR / "validation" / "phase8_a4_plausibility_summary.csv"
POINTS = [46, 94, 154]
SIZES = [0.005, 0.01, 0.02]


def main():
    telem_cols = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(scan.CORPUS_DIR / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols,
                     nrows=1440)
    strata = scan.stratified_picks(df, [c for c in df.columns if c.endswith("_P")])
    points = sorted(set(strata["low"] + strata["mid"] + strata["high"]))[:scan.N_POINTS]

    model = build_estimator_model("multiphase")
    est = StateEstimator(model)
    rows, summ = [], []
    for i in POINTS:
        row = df.iloc[i]
        ts = str(row["real_timestamp"])
        z = {c: float(row[c]) for c in z_cols}
        clean = est.estimate(z, ts)
        basis = kept_subspace_basis(est, clean, ts)
        rng = np.random.default_rng(scan.SEED + 400 + points.index(i))
        ids, sig = basis["meas_ids"], basis["sigma_meas"]
        for mag in scan.MAGNITUDES:
            atk = synthesize_kept_attack(basis, rng, mag)
            a = atk["a"]
            random_weighted_control(a, sig, rng)  # keep the scan's rng sequence
            random_sign_control(a, rng)
            if mag not in SIZES:
                continue
            ratios = []
            for k in np.flatnonzero(a != 0):
                zk = z[ids[k]]
                r = abs(a[k]) / abs(zk) if zk != 0 else np.inf
                ratios.append(r)
                rows.append({"i": i, "size_pu": mag, "channel": ids[k], "a": float(a[k]), "z_clean": zk,
                             "abs_a_over_abs_z": r, "abs_a_over_sigma": abs(a[k]) / sig[k]})
            ratios = np.array(ratios)
            sub = pd.DataFrame([r for r in rows if r["i"] == i and r["size_pu"] == mag])
            worst = sub.loc[sub["abs_a_over_abs_z"].idxmax()]
            summ.append({"i": i, "size_pu": mag, "n_nonzero": int(len(ratios)),
                         "max_abs_a_over_abs_z": float(ratios.max()), "argmax_channel": worst["channel"],
                         "n_channels_over_50pct": int((ratios > 0.5).sum()),
                         "all_under_50pct": bool((ratios < 0.5).all()),
                         "max_abs_a_over_sigma": float(sub["abs_a_over_sigma"].max())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    s = pd.DataFrame(summ)
    s.to_csv(OUT_SUMMARY, index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False))
    print()
    big = out.sort_values("abs_a_over_abs_z", ascending=False)
    for (i, mag), g in out.groupby(["i", "size_pu"]):
        g = g.reindex(g["abs_a_over_abs_z"].sort_values(ascending=False).index)
        print(f"--- i={i} size={mag}: {len(g)} nonzero channels; top 10 by |a|/|z|")
        print(g.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
