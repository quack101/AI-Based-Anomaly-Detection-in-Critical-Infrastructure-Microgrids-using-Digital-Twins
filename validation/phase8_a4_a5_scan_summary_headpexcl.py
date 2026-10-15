"""
Summary of validation/phase8_a4_a5_scan_headpexcl.csv per (attack,
size, kind). Detection rates are over CONVERGED solves only (the scan
leaves detected_by_* empty for non-converged ones); n_converged is
reported alongside. Also counts how many of the clean points the
active thresholds (validation/bdd_thresholds.py) flag.

Run: python -m validation.phase8_a4_a5_scan_summary_headpexcl
"""

import numpy as np
import pandas as pd

from validation.bdd_thresholds import load_active_thresholds
from validation.phase8_a4_a5_scan_frozen import OUT_PATH as SCAN_PATH, BASE_DIR

OUT_SUMMARY = BASE_DIR / "validation" / "phase8_a4_a5_scan_summary_headpexcl.csv"


def _rate(s):
    s = s.dropna().astype(bool)
    return float(s.mean()) if len(s) else float("nan")


def main():
    j99, lnr99 = load_active_thresholds()
    df = pd.read_csv(SCAN_PATH)
    rows = []
    for (part, mag, kind), g in df.groupby(["part", "magnitude_pu", "kind"], sort=True):
        conv = g[g["converged"].astype(bool)]
        either = conv["detected_by_J99"].astype(bool) | conv["detected_by_LNR99"].astype(bool)
        rows.append({
            "part": part, "magnitude_pu": mag, "kind": kind, "n": len(g), "n_converged": len(conv),
            "convergence_rate": len(conv) / len(g),
            "det_rate_J99": _rate(conv["detected_by_J99"]),
            "det_rate_LNR99": _rate(conv["detected_by_LNR99"]),
            "det_rate_either": float(either.mean()) if len(conv) else float("nan"),
            "median_dJ": g["dJ"].median(), "p95_dJ": g["dJ"].quantile(0.95),
            "median_dLNR": g["dLNR"].median(), "p95_dLNR": g["dLNR"].quantile(0.95),
            "median_achieved_over_intended": g["achieved_over_intended"].median(),
            "median_unattack_resid": g["unattack_residual_final"].median(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUMMARY, index=False)

    clean = df.drop_duplicates("i")[["i", "J_before", "lnr_before"]]
    flag_j = clean["J_before"] > j99
    flag_l = clean["lnr_before"] > lnr99
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    print()
    print(f"thresholds: J99={j99:.4f} LNR99={lnr99:.4f}")
    print(f"clean points: {len(clean)}  flagged by J99: {int(flag_j.sum())}  by LNR99: {int(flag_l.sum())}  "
          f"by either: {int((flag_j | flag_l).sum())}")
    print(f"clean J: min={clean['J_before'].min():.2f} median={clean['J_before'].median():.2f} "
          f"max={clean['J_before'].max():.2f};  clean LNR: min={clean['lnr_before'].min():.3f} "
          f"median={clean['lnr_before'].median():.3f} max={clean['lnr_before'].max():.3f}")
    if (flag_j | flag_l).any():
        print("flagged clean points:", clean[flag_j | flag_l].to_string(index=False))
    print("saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
