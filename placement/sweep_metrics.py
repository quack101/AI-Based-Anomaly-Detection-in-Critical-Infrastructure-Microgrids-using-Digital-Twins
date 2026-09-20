"""
Tier-1 (placement-quality) per-cell metrics for RES.md D10 Phase 4
sub-study 4A. Tier 1 = computable from FeederContext alone (no state
estimator / Jacobian -- that's Tier 2, deferred to Phase 4 itself).

All five metrics assume `selected` is a list of load dicts already
annotated by FeederContext.annotate_loads() (depth, branch, on_backbone,
downstream_kw all present).
"""

TOTAL_FEEDER_KW = 3490.0  # matches PHASE_KW_BASELINE's sum, used
                          # throughout this codebase since model
                          # verification (config/load_metadata.json
                          # nominal total)
PHASE_KW_BASELINE = {"A": 1400.0, "B": 952.5, "C": 1137.5}


def _load_phases(load):
    phase = load.get("phase", "")
    return [p for p in phase if p in ("A", "B", "C")] or ["A", "B", "C"]


def downstream_coverage_kw(selected, context):
    """Union, across all selected meters, of each meter's downstream
    subtree kW (context.downstream_load), de-duplicated so an ancestor-
    descendant pair among the selected buses is not double-counted:
    only "topmost" selected buses (no selected ancestor) contribute
    their full downstream_load; a selected descendant of another
    selected bus contributes nothing extra, since its downstream region
    is a strict subset of its selected ancestor's."""

    selected_buses = {str(l["bus"]) for l in selected}
    covering = []

    for bus in selected_buses:
        node = context.parent.get(bus)
        is_topmost = True
        while node is not None:
            if node in selected_buses:
                is_topmost = False
                break
            node = context.parent.get(node)
        if is_topmost:
            covering.append(bus)

    return sum(context.downstream_load.get(b, 0.0) for b in covering)


def phase_kw(selected):
    totals = {"A": 0.0, "B": 0.0, "C": 0.0}
    for load in selected:
        phases = _load_phases(load)
        share = load["base_kw"] / len(phases)
        for p in phases:
            totals[p] += share
    return totals


def compute_cell_metrics(selected, context):
    downstream_kw = downstream_coverage_kw(selected, context)
    phases = phase_kw(selected)
    depths = [l["depth"] for l in selected]

    return {
        "n_selected": len(selected),
        "downstream_coverage_kw": downstream_kw,
        "downstream_coverage_frac": downstream_kw / TOTAL_FEEDER_KW,
        "phase_kw_A": phases["A"],
        "phase_kw_B": phases["B"],
        "phase_kw_C": phases["C"],
        "depth_min": min(depths),
        "depth_max": max(depths),
        "depth_mean": sum(depths) / len(depths),
        "backbone_hits": sum(1 for l in selected if l["on_backbone"]),
        "branch_diversity": len({l["branch"] for l in selected}),
    }
