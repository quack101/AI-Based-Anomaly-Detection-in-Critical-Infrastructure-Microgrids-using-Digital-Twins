"""
Step 0 of the placement-improvement plan: does placement matter for
accuracy once the dominant bias is removed?

Finding that motivates it: the estimator's HEAD_P measurement function
(observability/jacobian.py build_branch_or_head_row, P through the
Vsource element) evaluates to exactly 0 even at the TRUE state, because
the source-bus angle is pinned to the fixed source-EMF angle, so
sin(theta) = 0. The recorded HEAD_P (~2 MW total) can never be matched:
the "+95 sigma HEAD_P bias" carrying ~95% of J is this, not placement.
The estimator is frozen, so this script does NOT change it. It only
varies inputs:

  V0  as deployed (HEAD_P enabled)
  V1  HEAD_P_A/B/C disabled in a temporary copy of each registry
  V2  V1 + oracle pseudo-measurements: each unmetered load's pseudo value
      is its TRUE P,Q at that timestamp (same sigma), applied by wrapping
      the estimator instance's pseudo-entry builder at runtime -- the
      best case for any pseudo-measurement improvement.

Both sets are scored on the same 60 points (validation/paired_ab.py
neutral_picks) on the same physics. Temporary registries go to a temp
directory, never config/.

Run: python -m validation.bias_control
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator
from validation.paired_ab import neutral_picks

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "reports" / "measurements" / "bias_control.json"
SETS = {
    "A": (BASE / "config/sensor_registry_multiphase_Aall91.json", BASE / "simulation/phase6_corpus_Aall91"),
    "B": (BASE / "config/sensor_registry_multiphase_Ball91.json", BASE / "simulation/phase6_corpus_Ball91"),
}
HEAD_P = {"HEAD_P_A", "HEAD_P_B", "HEAD_P_C"}
DATASET = BASE / "dataset" / "node_loads_91"


def registry_variant(path, tmp, head_p):
    reg = json.loads(Path(path).read_text())
    for e in reg:
        if e["id"] in HEAD_P:
            e["enabled"] = head_p
    out = Path(tmp) / f"{Path(path).stem}_headp{int(head_p)}.json"
    out.write_text(json.dumps(reg))
    return out


def true_loads(load_names, timestamps):
    files = {f.stem[len("node_"):].lower(): f for f in DATASET.glob("node_*.csv")}
    out = {}
    for n in load_names:
        d = pd.read_csv(files[n.lower()]).set_index("timestamp")
        out[n] = d.loc[timestamps, ["P_kW", "Q_kvar"]]
    return out


def run(registry, corpus, rows, oracle):
    cols = [c for c in pd.read_csv(corpus / "test_telemetry.csv", nrows=0).columns
            if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(corpus / "test_telemetry.csv", nrows=max(rows) + 2)
    npz = np.load(corpus / "test_x_true.npz", allow_pickle=True)
    x_v = npz["x_v"]
    model = build_estimator_model("multiphase", registry_path=registry, corpus_dir=corpus)
    st, est = model.state, StateEstimator(model)
    v_base = np.array([st.v_base[n] for n in st.all_node_names])
    flat_v, flat_d = list(st.x0_v), list(st.x0_delta)

    current = {"ts": None}
    if oracle:
        ts_all = sorted({str(df.iloc[i]["real_timestamp"]) for i in rows} |
                        {str(df.iloc[i + 1]["real_timestamp"]) for i in rows})
        truth = true_loads(list(model.bus_phase_by_load), ts_all)
        base_builder = est._weighted_entries_for_hour

        def oracle_builder(hour):
            entries = base_builder(hour)
            out = []
            for e in entries:
                if e["id"].startswith("PSEUDO_"):
                    e = dict(e)
                    p, q = truth[e["load_name"]].loc[current["ts"]]
                    e["z_value"] = float(p if e["quantity"] == "P_injection" else q)
                out.append(e)
            return out

        est._weighted_entries_for_hour = oracle_builder

    errs, conv, iters, Js = [], [], [], []
    for i in rows:
        st.x0_v, st.x0_delta = list(flat_v), list(flat_d)
        for k, j in enumerate((i, i + 1)):
            row = df.iloc[j]
            current["ts"] = str(row["real_timestamp"])
            r = est.estimate({c: float(row[c]) for c in cols if c in {e["id"] for e in model.meas_entries}},
                             current["ts"])
            errs.append(np.abs(r.x_v - x_v[j]) / v_base)
            conv.append(r.converged)
            iters.append(r.iterations)
            Js.append(r.J)
            st.x0_v, st.x0_delta = list(r.x_v), list(r.x_delta)
    return np.array(errs), conv, iters, Js


def main():
    rows = neutral_picks(SETS["A"][1])
    result = {"source": "validation/bias_control.py", "rows": rows, "variants": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, head_p, oracle in (("V0_as_deployed", True, False), ("V1_headP_off", False, False),
                                     ("V2_headP_off_oracle_pseudo", False, True)):
            res = {}
            for s, (reg, corpus) in SETS.items():
                e, conv, it, Js = run(registry_variant(reg, tmp, head_p), corpus.resolve(), rows, oracle)
                res[s] = (e, conv, it, Js)
                print(f"{name} {s}: median {np.median(e):.5f} converged {sum(conv)}/{len(conv)}", flush=True)
            eA, eB = res["A"][0], res["B"][0]
            mA, mB = np.median(eA, axis=1), np.median(eB, axis=1)
            d = mB - mA
            wins_b, wins_a = int((d < 0).sum()), int((d > 0).sum())
            result["variants"][name] = {
                **{s: {"median": float(np.median(res[s][0])), "p95": float(np.percentile(res[s][0], 95)),
                       "max": float(res[s][0].max()), "converged": int(sum(res[s][1])), "n": len(res[s][1]),
                       "mean_iterations": float(np.mean(res[s][2])), "J_mean": float(np.mean(res[s][3])),
                       "worst_node_median": float(np.median(res[s][0], axis=0).max())} for s in res},
                "B_minus_A_per_point_mean": float(d.mean()), "relative": float(d.mean() / mA.mean()),
                "B_better_points": wins_b, "A_better_points": wins_a,
                "sign_test_p": float(binomtest(wins_b, wins_b + wins_a, 0.5).pvalue) if wins_a + wins_b else None,
            }
            OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["variants"], indent=2))


if __name__ == "__main__":
    main()
