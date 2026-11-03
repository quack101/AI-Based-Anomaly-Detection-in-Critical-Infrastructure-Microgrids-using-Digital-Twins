"""
A4/A5 at realistic (small) sizes, by scaling the OpenDSS-generated vectors.

The library's A4/A5 at 0.01-0.05 pu move meter P by 3,000-10,000 kW (the whole
feeder carries ~3,490 kW): invisible to the residual test but absurd to any
detector that knows normal loads (all four detectors catch 30/30). A careful
attacker would use a much smaller state shift.

A4 is linear by construction (a = H c, c = size * unit direction from the
row's kept subspace), so a_scaled = a * (m_new / m_old) IS the A4 attack of
size m_new on the same rows. A5 = A4 + a Newton correction that vanishes as
the size goes to 0; measured ||a|| / size is the same at 0.01, 0.02 and 0.05
pu for both types (validation/eval_a45_scaled.py prints it), so scaling is a
close approximation for A5 at smaller sizes. Stated as such in the report.

Each of the 15 A4 + 15 A5 windows in dataset/attack_bench is scaled to every
size in SIZES and scored like validation/eval_attack_bench.py (K = 1 and 3).

Run: python -m validation.eval_a45_scaled
"""

import json

import numpy as np
import pandas as pd

from validation import detect_a2_peer as V
from validation import detect_peer_power as Pp
from validation.a2_persist_eval import persist
from validation.detect_ae import W
from validation.eval_attack_bench import ae_maha, BASE, OUT, BENCH, CONTEXT

SIZES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def main():
    vdet, pdet = V.fit_default(), Pp.fit_default()
    ae_score, ae_thr = ae_maha()
    cols = vdet.cols
    order = [json.load(open(BENCH / "channels.json")).index(c) for c in cols]
    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    Pi = [j for j, c in enumerate(cols) if Pp._P.match(c)]
    Vi = vdet.v_idx
    wins = [json.loads(l) for l in open(BENCH / "windows.jsonl")]
    wins = [w for w in wins if w.get("ok") and w["type"] in ("A4", "A5")]
    rows = []
    for w in wins:
        a0 = np.load(BENCH / f"a_{w['id']}.npy")[:, order]
        s, d = w["start"], w["duration"]
        s0 = max(0, s - max(CONTEXT, W))
        for m in SIZES:
            a = a0 * (m / w["magnitude"])
            seg = X[s0:s + d].copy(); seg[s - s0:] += a
            al = {"V-peer": vdet.score(seg) > vdet.thr, "P-peer": pdet.score(seg) > pdet.thr,
                  "AE-Maha": ae_score(seg) > ae_thr}
            al["V or P"] = al["V-peer"] | al["P-peer"]
            row = {"id": w["id"], "type": w["type"], "size_pu": m,
                   "max_abs_dP_kW": float(np.abs(a[:, Pi]).max()),
                   "median_abs_dP_kW": float(np.median(np.abs(a[:, Pi]))),
                   "max_abs_dV_pu": float(np.abs(a[:, Vi]).max())}
            for name, x in al.items():
                for K in (1, 3):
                    row[f"{name} K{K}"] = bool(persist(x, K)[s - s0:].any())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a45_scaled_eval.csv", index=False)
    dets = ["V-peer", "P-peer", "V or P", "AE-Maha"]
    out = {}
    for (t, m), g in df.groupby(["type", "size_pu"]):
        out[f"{t} {m:g} pu"] = {"n": len(g), "median max|dP| kW": round(g.max_abs_dP_kW.median(), 1),
                                "median max|dV| pu": round(g.max_abs_dV_pu.median(), 5),
                                **{f"{dd} K{K}": f"{int(g[f'{dd} K{K}'].sum())}/{len(g)}" for dd in dets for K in (1, 3)}}
        r = out[f"{t} {m:g} pu"]
        print(f"{t} {m:8g} pu | max|dP| ~{r['median max|dP| kW']:8.1f} kW  max|dV| ~{r['median max|dV| pu']:.5f} pu | "
              + "  ".join(f"{dd}: {r[dd + ' K3']}" for dd in dets))
    json.dump(out, open(OUT / "a45_scaled_eval_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
