"""
RES.md D10 Phase 4 sub-study 4A, TIER 2 -- observability/security
metrics now that H exists (Tier 1 was placement-quality only, no
Jacobian). Reuses the Tier-1 grid shape but RESCOPES THE AXIS to
instrumentation FRACTION of the 85 candidate buses:

    10/20/30/40/50/60/80/100% -> 9/17/26/34/43/51/68/85

Rationale (RES.md): the Tier-1 grid (15/30/45/49/60/70/80/85) put 5 of
8 points above 50% and only one below 30%, under-sampling the sparse
regime this project studies. 40% is retained because Ald25 fixes ~40%
smart-meter coverage as its given constraint -- the only stated
coverage fraction in the 32-paper corpus, hence a citable anchor.

Per cell (strategy x count, INDEPENDENT construction only -- Tier 1
already showed independent-vs-nested divergence is a placement-quality
question; Tier 2's new metrics are properties of the selected SET at a
given count, and doubling every metric here for both constructions was
judged not worth the added report length for this phase):
  fraction, count, m, m/n, rank(H)/dim(x), cond(H^T W H) [meters-only
  H -- ALWAYS singular here, m never reaches 553 from load metering
  alone at any count up to 85 (Part A's own headline finding); reported
  as inf, not silently omitted], critical-measurement count, and the
  security index alpha_k per meter (Sayghe eq. 38):

      alpha_k = min ||Hc||_0  s.t.  H(k,:) c = 1

  NP-hard in general -- relaxed here to an L1 LINEAR PROGRAM (min
  ||Hc||_1 s.t. H(k,:)c=1, standard basis-pursuit-style relaxation,
  solved via scipy.optimize.linprog/HiGHS). Limitation: L1 relaxation
  is a convex proxy for L0 and can under- or over-estimate the true
  sparsest solution when H's columns are not sufficiently incoherent;
  it is used here because exact L0 minimisation over a 553-dimensional
  c is combinatorial and RES.md explicitly authorises the relaxation.
  One alpha_k per METERED NODE (its P_injection row is the
  representative k -- Q_injection typically yields a closely related
  value for the same physical meter, and computing both would double
  cost for little extra insight, stated here as a scoping choice).

RandomPlacement seed policy (RES.md): 30 seeds below 30% instrumentation
(counts 9, 17), 10 seeds at/above (26, 34, 43, 51, 68, 85).

Output:
  artifacts/<run_id>/sweep_4a_tier2/cells.csv
  artifacts/<run_id>/sweep_4a_tier2/alpha_k.csv          -- per-meter detail
  artifacts/<run_id>/sweep_4a_tier2/hybrid_alpha_correlation.json
  reports/phase4_observability_report.txt (Part B section)

Run: python -m validation.sweep_4a_tier2
"""

import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.optimize import linprog

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import (
    HybridPlacement, RandomPlacement, DegreePlacement, CommunityDetectionPlacement,
)
from observability.state_vector import build_state_vector
from observability import jacobian as J
from observability.sweep import find_critical_measurements
from placement.electrical_distance import get_system_y_index

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ARTIFACTS_DIR / RUN_ID / "sweep_4a_tier2"

CEILING = 85
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
# RES.md D10 Phase 4 sub-study 4A Tier 2 states these counts explicitly
# -- taken verbatim, NOT recomputed via round(CEILING * fraction):
# Python's round() is round-half-to-even, which silently turns
# 85*0.10=8.5 into 8 (not 9) and 85*0.50=42.5 into 42 (not 43).
# Discovered via a first run that produced exactly those two wrong
# counts; fixed by using the task's own integers directly.
COUNTS = [9, 17, 26, 34, 43, 51, 68, 85]
assert len(COUNTS) == len(FRACTIONS)

SEEDS_BELOW_30PCT = 30
SEEDS_AT_OR_ABOVE_30PCT = 10


def seeds_for_count(fraction):
    n = SEEDS_BELOW_30PCT if fraction < 0.30 else SEEDS_AT_OR_ABOVE_30PCT
    return list(range(n))


CELL_FIELDS = [
    "strategy", "fraction", "count", "seed",
    "m", "n", "m_over_n", "rank", "rank_frac", "cond_htwh", "n_critical",
    "alpha_k_mean", "alpha_k_median", "alpha_k_min",
]

ALPHA_FIELDS = ["strategy", "fraction", "count", "seed", "load_name", "bus", "hybrid_score", "alpha_k"]


def _entries_for_selection(selected):
    """One P_injection entry per selected load, node/phase resolved via
    the same primary-phase convention the live registry uses."""

    import opendssdirect as dss

    entries = []
    for load in selected:
        dss.Circuit.SetActiveElement(f"Load.{load['name']}")
        node_order = dss.CktElement.NodeOrder()
        phase = {1: "A", 2: "B", 3: "C"}[node_order[0]]
        bus = str(load["bus"])
        entries.append({
            "id": f"{load['name']}_P", "node": bus, "phase": phase,
            "quantity": "P_injection", "load_name": load["name"],
        })
        entries.append({
            "id": f"{load['name']}_Q", "node": bus, "phase": phase,
            "quantity": "Q_injection", "load_name": load["name"],
        })
    return entries


def _alpha_k_lp(H, k_row):
    """min ||Hc||_1 s.t. H(k,:)c = 1 -- L1 relaxation of the security
    index (Sayghe eq. 38). Returns the optimal objective value (the
    relaxed alpha_k), or None if infeasible/unbounded."""

    m, n = H.shape
    c_obj = np.concatenate([np.zeros(n), np.ones(m)])
    A_ub = np.vstack([
        np.hstack([H, -np.eye(m)]),
        np.hstack([-H, -np.eye(m)]),
    ])
    b_ub = np.zeros(2 * m)
    A_eq = np.zeros((1, n + m))
    A_eq[0, :n] = H[k_row]
    b_eq = [1.0]
    bounds = [(None, None)] * n + [(0, None)] * m

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(res.fun) if res.success else None


def compute_cell(strategy_name, selected, state, Y, node_index, seed=None, hybrid_scores=None):
    entries = _entries_for_selection(selected)
    H, h0 = J.build_jacobian(entries, Y, node_index, state)

    m, n = H.shape
    rank = int(np.linalg.matrix_rank(H, tol=None))

    sigmas = np.array([max(abs(v) * 0.01, 1e-9) for v in h0])
    HtWH = H.T @ (H * (1.0 / sigmas ** 2)[:, None])
    try:
        cond = float(np.linalg.cond(HtWH))
    except np.linalg.LinAlgError:
        cond = float("inf")

    crit_idx, _ = find_critical_measurements(H)
    n_critical = len(crit_idx)

    alpha_values = []
    alpha_rows = []
    for i, load in enumerate(selected):
        k_row = 2 * i  # P_injection row for this meter
        alpha = _alpha_k_lp(H, k_row)
        if alpha is not None:
            alpha_values.append(alpha)
        alpha_rows.append({
            "strategy": strategy_name, "fraction": len(selected) / CEILING,
            "count": len(selected), "seed": seed,
            "load_name": load["name"], "bus": load["bus"],
            "hybrid_score": load.get("score") if hybrid_scores is None else hybrid_scores.get(load["name"]),
            "alpha_k": alpha,
        })

    cell = {
        "strategy": strategy_name, "fraction": len(selected) / CEILING, "count": len(selected),
        "seed": seed, "m": m, "n": n, "m_over_n": m / n,
        "rank": rank, "rank_frac": rank / n, "cond_htwh": cond, "n_critical": n_critical,
        "alpha_k_mean": mean(alpha_values) if alpha_values else None,
        "alpha_k_median": sorted(alpha_values)[len(alpha_values) // 2] if alpha_values else None,
        "alpha_k_min": min(alpha_values) if alpha_values else None,
    }
    return cell, alpha_rows


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    state = build_state_vector()
    Y, node_index = get_system_y_index()

    CommunityDetectionPlacement.clear_cache()

    cells = []
    alpha_rows_all = []

    t0 = time.time()

    print("Hybrid...")
    for count in COUNTS:
        strategy = HybridPlacement(target_meters=count, verbose=False)
        selected = strategy.select([dict(l) for l in loads], context)
        cell, arows = compute_cell("Hybrid", selected, state, Y, node_index)
        cells.append(cell)
        alpha_rows_all.extend(arows)

    print("Degree...")
    for count in COUNTS:
        strategy = DegreePlacement(target_meters=count)
        selected = strategy.select([dict(l) for l in loads], context)
        cell, arows = compute_cell("Degree", selected, state, Y, node_index)
        cells.append(cell)
        alpha_rows_all.extend(arows)

    print("Community12 (raw)...")
    for count in COUNTS:
        strategy = CommunityDetectionPlacement(target_meters=count, num_communities=None)
        selected = strategy.select([dict(l) for l in loads], context)
        cell, arows = compute_cell("Community12", selected, state, Y, node_index)
        cells.append(cell)
        alpha_rows_all.extend(arows)

    print("Community4 (merged)...")
    for count in COUNTS:
        strategy = CommunityDetectionPlacement(target_meters=count, num_communities=4)
        selected = strategy.select([dict(l) for l in loads], context)
        cell, arows = compute_cell("Community4", selected, state, Y, node_index)
        cells.append(cell)
        alpha_rows_all.extend(arows)

    print("Random...")
    for count, fraction in zip(COUNTS, FRACTIONS):
        for seed in seeds_for_count(fraction):
            strategy = RandomPlacement(target_meters=count, seed=seed)
            selected = strategy.select([dict(l) for l in loads], context)
            cell, arows = compute_cell("Random", selected, state, Y, node_index, seed=seed)
            cells.append(cell)
            alpha_rows_all.extend(arows)

    print(f"Total cells: {len(cells)}  elapsed: {time.time()-t0:.1f}s")

    with open(RUN_DIR / "cells.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_FIELDS)
        writer.writeheader()
        writer.writerows(cells)

    with open(RUN_DIR / "alpha_k.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALPHA_FIELDS)
        writer.writeheader()
        writer.writerows(alpha_rows_all)

    # Correlation: hybrid placement score vs. alpha_k, pooled across all
    # 8 Hybrid counts (each meter contributes one (score, alpha_k) pair).
    hybrid_pairs = [
        (r["hybrid_score"], r["alpha_k"]) for r in alpha_rows_all
        if r["strategy"] == "Hybrid" and r["hybrid_score"] is not None and r["alpha_k"] is not None
    ]
    scores = np.array([p[0] for p in hybrid_pairs])
    alphas = np.array([p[1] for p in hybrid_pairs])
    if len(scores) > 1 and scores.std() > 0 and alphas.std() > 0:
        pearson_r = float(np.corrcoef(scores, alphas)[0, 1])
    else:
        pearson_r = None

    correlation = {
        "n_pairs": len(hybrid_pairs),
        "pearson_r": pearson_r,
        "scatter_sample": [
            {"hybrid_score": float(s), "alpha_k": float(a)}
            for s, a in list(zip(scores, alphas))[:30]
        ],
    }
    with open(RUN_DIR / "hybrid_alpha_correlation.json", "w") as f:
        json.dump(correlation, f, indent=2)

    print(f"Hybrid score vs alpha_k: n={correlation['n_pairs']} pearson_r={pearson_r}")
    print(f"Wrote {len(cells)} cells to {RUN_DIR / 'cells.csv'}")

    return cells, alpha_rows_all, correlation, RUN_DIR


if __name__ == "__main__":
    main()
