"""
Paired A-vs-B accuracy check on IDENTICAL rows.

validation/measure_meter_set.py reproduces the reference scorecard
protocol, which stratifies rows by the summed telemetry of the set's
own *_P columns. Sets A and B meter different loads, so that protocol
scores them on different rows (A_all91 vs B_all91: 19 of 60 rows
shared), even though their true physics is identical (all-91 corpora,
x_true arrays equal). This script scores both sets on the SAME rows,
stratified by a meter-independent quantity -- the true feeder-head P
from {split}_ground_truth.csv -- with the same seed and the same
A_flat / B_warm pairing, and reports per-point paired differences.

Run: python -m validation.paired_ab --a-registry ... --a-corpus ...
                                    --b-registry ... --b-corpus ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from observability.estimator_model import build_estimator_model
from estimation.state_estimator import StateEstimator

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "reports" / "measurements" / "paired_A_all91_vs_B_all91.json"
SEED = 777
N_PER_STRATUM = 10
SCORECARD_ROWS = 1440


def neutral_picks(corpus):
    gt = pd.read_csv(Path(corpus) / "test_ground_truth.csv", nrows=SCORECARD_ROWS)
    p = gt["feeder_head_p_kw"]
    q1, q2 = p.quantile([1 / 3, 2 / 3])
    rng = np.random.default_rng(SEED)
    out = []
    for idx in (gt.index[p <= q1], gt.index[(p > q1) & (p <= q2)], gt.index[p > q2]):
        out += sorted(rng.choice(idx, size=N_PER_STRATUM, replace=False).tolist())
    return out


def score(registry, corpus, rows):
    corpus = Path(corpus)
    cols = [c for c in pd.read_csv(corpus / "test_telemetry.csv", nrows=0).columns
            if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(corpus / "test_telemetry.csv", nrows=SCORECARD_ROWS + 1)
    x_v = np.load(corpus / "test_x_true.npz", allow_pickle=True)["x_v"]
    model = build_estimator_model("multiphase", registry_path=registry, corpus_dir=corpus)
    st, est = model.state, StateEstimator(model)
    v_base = np.array([st.v_base[n] for n in st.all_node_names])
    flat_v, flat_d = list(st.x0_v), list(st.x0_delta)
    res = []
    for i in rows:
        st.x0_v, st.x0_delta = list(flat_v), list(flat_d)
        r_a = est.estimate({c: float(df.iloc[i][c]) for c in cols}, str(df.iloc[i]["real_timestamp"]))
        e_a = np.abs(r_a.x_v - x_v[i]) / v_base
        st.x0_v, st.x0_delta = list(r_a.x_v), list(r_a.x_delta)
        r_b = est.estimate({c: float(df.iloc[i + 1][c]) for c in cols}, str(df.iloc[i + 1]["real_timestamp"]))
        e_b = np.abs(r_b.x_v - x_v[i + 1]) / v_base
        res.append((e_a, r_a.converged))
        res.append((e_b, r_b.converged))
    return res


def main(a_reg, a_corpus, b_reg, b_corpus):
    xa = np.load(Path(a_corpus) / "test_x_true.npz", allow_pickle=True)["x_v"]
    xb = np.load(Path(b_corpus) / "test_x_true.npz", allow_pickle=True)["x_v"]
    if not np.array_equal(xa, xb):
        raise ValueError("A and B corpora do not share the same physics; a paired comparison is meaningless")
    rows = neutral_picks(a_corpus)
    A, B = score(a_reg, a_corpus, rows), score(b_reg, b_corpus, rows)
    ea, eb = np.array([x[0] for x in A]), np.array([x[0] for x in B])
    med_a, med_b = np.median(ea, axis=1), np.median(eb, axis=1)  # per-point median over nodes
    d = med_b - med_a  # > 0 means B worse
    pooled_a, pooled_b = ea.ravel(), eb.ravel()
    out = {
        "source": "validation/paired_ab.py (same 60 points for both sets, stratified by true feeder-head P)",
        "rows": rows, "n_points": len(d),
        "converged": {"A": int(sum(x[1] for x in A)), "B": int(sum(x[1] for x in B))},
        "pooled": {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
                   for k, v in (("A", pooled_a), ("B", pooled_b))},
        "per_point_median_err": {
            "B_minus_A_mean": float(d.mean()), "B_minus_A_median": float(np.median(d)),
            "B_better_points": int((d < 0).sum()), "A_better_points": int((d > 0).sum()),
            "relative_mean_change": float(d.mean() / med_a.mean()),
        },
        "worst_node_median_err": {"A": float(np.median(ea, axis=0).max()), "B": float(np.median(eb, axis=0).max())},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("a-registry", "a-corpus", "b-registry", "b-corpus"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    r = lambda x: Path(x).resolve()
    main(r(a.a_registry), r(a.a_corpus), r(a.b_registry), r(a.b_corpus))
