"""
Structural placement metrics for Hybrid v1 / v2 / v3 (all three
connection priorities) and Degree at several meter budgets, for
reports/hybrid_v3.md. No estimator runs; everything is from the
placement and the measurement Jacobian at a flat start:

  SORI, coverage, unseen   BOI/SORI as in validation/literature_metrics.py
                           boi (same registry rules, same Jacobian test)
  dead ends                selected buses with topology degree 1
  branches                 distinct branch ids metered
  phase dev (pp)           placement.sweep_metrics.compute_phase_dev_pp;
                           tests/test_placement_regression.py allows 5 pp

Output: reports/literature_metrics/structure_metrics.json (not committed)

Run: python -m validation.placement_structure_metrics [--targets 20 30 49 60 70]
"""

import argparse
import contextlib
import io
import json
import tempfile

import numpy as np

from validation.literature_metrics import OUT, _meters_from_selection, _registry_for

TARGETS = (20, 30, 49, 60, 70)


def _strategies(n):
    from placement.placement_strategy import HybridPlacement, DegreePlacement
    from placement.hybrid_v2 import HybridV2Placement
    from placement.hybrid_v3 import HybridV3Placement
    return {
        "Hybrid v1": HybridPlacement(n),
        "HybridV2": HybridV2Placement(n),
        "V3 within_tier": HybridV3Placement(n, connection_priority="within_tier"),
        "V3 first": HybridV3Placement(n, connection_priority="first"),
        "V3 first_balanced (default)": HybridV3Placement(n),
        "Degree": DegreePlacement(n),
    }


def main(targets=TARGETS):
    from core.ieee123_model import extract_all
    from placement.feeder_context import FeederContext
    from placement.feeder_adapter import network_and_loads_from_core_extraction
    from placement.placement_strategy import HybridPlacement
    from placement.sweep_metrics import compute_phase_dev_pp
    from observability.state_vector import build_state_vector, complex_v_from_state
    from observability import jacobian as J
    from placement.electrical_distance import get_system_y_index

    with contextlib.redirect_stdout(io.StringIO()):
        net, loads = network_and_loads_from_core_extraction(extract_all())
    ctx = FeederContext(net, loads)
    ctx.annotate_loads(loads)
    phase_helper = HybridPlacement.__new__(HybridPlacement)  # pure helper methods only

    sels = {n: {k: s.select([dict(l) for l in loads], ctx) for k, s in _strategies(n).items()} for n in targets}
    # registries first (generate_registry recompiles the circuit), then the Jacobian
    with tempfile.TemporaryDirectory() as d:
        regs = {n: {k: _registry_for(_meters_from_selection(sel), d) for k, sel in by.items()}
                for n, by in sels.items()}
    state = build_state_vector()
    Y, node_index = get_system_y_index()
    flat = complex_v_from_state(state, state.x0_v, state.x0_delta)
    n_v = state.n_phase_nodes

    results = {"_meta": {"source": "validation/placement_structure_metrics.py", "n_phase_nodes": n_v,
                         "evaluated_at": "flat start"}}
    for n, by in sels.items():
        results[str(n)] = {}
        print(f"\n{n} meters\n{'method':28s}  SORI  cover  unseen  dead_ends  branches  phase_dev_pp")
        for k, sel in by.items():
            H, _ = J.build_jacobian(regs[n][k], Y, node_index, state, at_v=flat)
            Hv = np.abs(H[:, :n_v])
            boi = (Hv > np.maximum(1e-9 * Hv.max(axis=1, keepdims=True), 1e-12)).sum(axis=0)
            pk = phase_helper._selected_phase_kw(sel)
            r = {"SORI": int(boi.sum()), "coverage": float((boi >= 1).mean()), "unseen": int((boi == 0).sum()),
                 "dead_ends": int(sum(ctx.bus_degree(l["bus"]) < 2 for l in sel)),
                 "branches": len({l["branch"] for l in sel}),
                 "phase_dev_pp": float(compute_phase_dev_pp(pk["A"], pk["B"], pk["C"]))}
            results[str(n)][k] = r
            print(f"{k:28s}  {r['SORI']:4d}  {r['coverage']:.1%}  {r['unseen']:5d}  {r['dead_ends']:8d}  "
                  f"{r['branches']:8d}  {r['phase_dev_pp']:11.1f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "structure_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=int, nargs="+", default=list(TARGETS))
    main(p.parse_args().targets)
