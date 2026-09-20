"""
RES.md D10 Phase 4 sub-study 4A, TIER 1 ONLY (placement-quality metrics
computable from FeederContext alone -- no state estimator / Jacobian;
that's Tier 2, deferred). Pre-phase groundwork, not a numbered phase.

Grid: 8 meter counts x 5 strategies x 2 constructions (independent,
nested), RandomPlacement run over >=10 seeds per cell.

Output:
  artifacts/<run_id>/sweep_4a_tier1/cells.csv       -- one row per
      (strategy, construction, count[, seed])
  artifacts/<run_id>/sweep_4a_tier1/community_variants.json -- Part A
      cross-reference (sizes, boundary nodes, both variants)
  reports/sweep_4a_tier1_report.txt

Run: python -m validation.sweep_4a_tier1
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import (
    HybridPlacement,
    RandomPlacement,
    DegreePlacement,
    CommunityDetectionPlacement,
)
from placement.sweep_metrics import compute_cell_metrics, PHASE_KW_BASELINE, TOTAL_FEEDER_KW

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ARTIFACTS_DIR / RUN_ID / "sweep_4a_tier1"

COUNTS = [15, 30, 45, 49, 60, 70, 80, 85]
CEILING = 85
NUM_SEEDS = 10
SEEDS = list(range(NUM_SEEDS))

CSV_FIELDS = [
    "strategy", "construction", "count", "seed",
    "n_selected", "downstream_coverage_kw", "downstream_coverage_frac",
    "phase_kw_A", "phase_kw_B", "phase_kw_C",
    "depth_min", "depth_max", "depth_mean",
    "backbone_hits", "branch_diversity",
]


def _row(strategy, construction, count, seed, metrics):
    row = {"strategy": strategy, "construction": construction, "count": count, "seed": seed}
    row.update(metrics)
    return row


def run_deterministic(name, make_strategy, loads, context, rows):
    # independent
    for count in COUNTS:
        strategy = make_strategy(count)
        selected = strategy.select([dict(l) for l in loads], context)
        metrics = compute_cell_metrics(selected, context)
        rows.append(_row(name, "independent", count, None, metrics))

    # nested: one 85-run, preserve append order, slice prefixes
    strategy85 = make_strategy(CEILING)
    order = strategy85.select([dict(l) for l in loads], context)
    for count in COUNTS:
        prefix = order[:count]
        metrics = compute_cell_metrics(prefix, context)
        rows.append(_row(name, "nested", count, None, metrics))


def run_random(loads, context, rows):
    # independent: fresh draw per (count, seed)
    for count in COUNTS:
        for seed in SEEDS:
            strategy = RandomPlacement(target_meters=count, seed=seed)
            selected = strategy.select([dict(l) for l in loads], context)
            metrics = compute_cell_metrics(selected, context)
            rows.append(_row("Random", "independent", count, seed, metrics))

    # nested: one 85-draw per seed (== a full shuffle of all 85 distinct
    # load buses, since target_meters == population size), slice prefixes
    for seed in SEEDS:
        strategy85 = RandomPlacement(target_meters=CEILING, seed=seed)
        order = strategy85.select([dict(l) for l in loads], context)
        for count in COUNTS:
            prefix = order[:count]
            metrics = compute_cell_metrics(prefix, context)
            rows.append(_row("Random", "nested", count, seed, metrics))


def aggregate_random(rows):
    """mean/std across seeds, per (construction, count), for each metric."""

    metric_names = [f for f in CSV_FIELDS if f not in ("strategy", "construction", "count", "seed")]
    by_cell = defaultdict(list)

    for r in rows:
        if r["strategy"] != "Random":
            continue
        by_cell[(r["construction"], r["count"])].append(r)

    summary = []
    for (construction, count), cell_rows in sorted(by_cell.items()):
        entry = {"construction": construction, "count": count, "n_seeds": len(cell_rows)}
        for m in metric_names:
            values = [r[m] for r in cell_rows]
            entry[f"{m}_mean"] = mean(values)
            entry[f"{m}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(entry)

    return summary


def community_variant_summary(context):
    from placement.community_detection import merge_communities, relabel_communities

    CommunityDetectionPlacement.clear_cache()
    raw_comm, adjacency = CommunityDetectionPlacement._get_raw_communities(context)
    merged_comm = relabel_communities(merge_communities(adjacency, raw_comm, 4))

    def is_boundary(node, comm):
        own = comm[node]
        return any(comm.get(n) != own for n in adjacency.get(node, {}))

    def summarize(comm):
        by_c = defaultdict(list)
        for node, c in comm.items():
            by_c[c].append(node)
        sizes = sorted((len(v) for v in by_c.values()), reverse=True)
        boundary = {n: is_boundary(n, comm) for n in ("18", "67", "76")}
        return {"num_communities": len(by_c), "sizes": sizes, "boundary_18_67_76": boundary}

    return {
        "raw_12": summarize(raw_comm),
        "merged_4": summarize(merged_comm),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    CommunityDetectionPlacement.clear_cache()

    rows = []

    print("Hybrid...")
    run_deterministic("Hybrid", lambda t: HybridPlacement(target_meters=t, verbose=False),
                       loads, context, rows)

    print("Degree...")
    run_deterministic("Degree", lambda t: DegreePlacement(target_meters=t),
                       loads, context, rows)

    print("Community12...")
    run_deterministic("Community12", lambda t: CommunityDetectionPlacement(target_meters=t, num_communities=None),
                       loads, context, rows)

    print("Community4...")
    run_deterministic("Community4", lambda t: CommunityDetectionPlacement(target_meters=t, num_communities=4),
                       loads, context, rows)

    print(f"Random ({NUM_SEEDS} seeds)...")
    run_random(loads, context, rows)

    print(f"Total cells: {len(rows)}")

    with open(RUN_DIR / "cells.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    random_summary = aggregate_random(rows)
    with open(RUN_DIR / "random_summary.json", "w") as f:
        json.dump(random_summary, f, indent=2)

    community_variants = community_variant_summary(context)
    with open(RUN_DIR / "community_variants.json", "w") as f:
        json.dump(community_variants, f, indent=2)

    print(f"Wrote {len(rows)} rows to {RUN_DIR / 'cells.csv'}")
    print(f"Wrote random seed summary to {RUN_DIR / 'random_summary.json'}")
    print(f"Wrote community variant summary to {RUN_DIR / 'community_variants.json'}")

    return rows, random_summary, community_variants, RUN_DIR


if __name__ == "__main__":
    main()
