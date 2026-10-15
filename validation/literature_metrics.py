"""
Placement metrics from the published literature that the A-vs-B work had
not computed yet, for reports/placement_literature_metrics.md.

  boi       BOI / SORI redundancy (PMU-placement literature, e.g. Ahmed
            et al. Energies 2024; Hyacinth & Gomathi Energies 2021).
            For a PMU a bus is "observed" by every PMU at or next to it.
            Our sensors are not PMUs, so the measurement analogue is used:
            BOI(phase-node k) = number of enabled measurement rows whose
            Jacobian row depends on |V_k| (an injection meter depends on
            its own bus and its neighbours, like a PMU). SORI = sum of BOI.
            Coverage = share of phase-nodes with BOI >= 1. Evaluated at a
            flat start for every placement method at 49 meters, each
            method's registry built with the same rules as
            sensors/build_registry_for_set.py's noise stage (meter P,Q,|V|
            + substation head), in memory.
  accuracy  From reports/measurements/paired_A_all91_vs_B_all91.npz
            (validation/paired_ab.py: same 60 points, same physics):
            - Singh, Pal & Vinter (IEEE TPWRS 2009) style per-bus
              probability index: per phase-node, the share of points where
              the relative |V| error and the angle error are both within a
              limit. Singh et al. bound this with a Chebyshev inequality on
              the error covariance; here it is the EMPIRICAL share.
            - angle error (deg): median / p95 / max.
            - Picallo et al. (IEEE TPWRS 2019) optimal-design criteria on
              the |V| error, EMPIRICAL versions from the 60 points:
              A = trace of the error second-moment matrix (sum of per-node
              mean squared error), E = its largest eigenvalue. The analytic
              covariance is not usable on this feeder (see
              validation/measure_meter_set.py docstring), and D (log det)
              is not computable from 60 samples of 278 nodes (rank <= 60).

Outputs: reports/literature_metrics/*.json and *.png

Run: python -m validation.literature_metrics boi
     python -m validation.literature_metrics accuracy
"""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "reports" / "literature_metrics"
COLORS = {"Set A (frozen)": "#eba53a", "Hybrid v1 (current code)": "#f6d29b", "HybridV2 (set B)": "#2a78d6",
          "Random": "#9aa9b8", "Degree": "#7b6fd6", "Community4": "#3a8a55", "Community12": "#b5403a"}
N_METERS = 49
RANDOM_SEEDS = range(10)
# (relative |V| error limit, angle error limit in degrees) pairs for the probability
# index. Measured angle errors are ~1.8 deg median on this feeder, so a 1 deg limit is
# met by almost no bus for either set; the pairs below are the ones that separate buses.
LIMITS = ((0.02, 2.0), (0.05, 3.0), (0.05, 5.0))
PLOT_LIMIT = (0.05, 3.0)


def _registry_for(meters, workdir):
    from sensors import generate_registry
    from sensors.build_registry_for_set import apply_deployment_flips
    path = Path(workdir) / "meters.json"
    path.write_text(json.dumps(meters))
    with contextlib.redirect_stdout(io.StringIO()):
        entries = generate_registry.main(multiphase=True, write=False, meters_path=path)
    apply_deployment_flips(entries)
    return [e for e in entries if e["enabled"]]


def _meters_from_selection(selected):
    """Same meter_id / load_name / bus layout as placement.representative_meters,
    for any strategy (non-Hybrid strategies do not tag size_class)."""
    return [{"meter_id": f"M{i:03d}", "load_name": l["name"], "bus": l["bus"],
             "base_kw": l["base_kw"], "branch": l.get("branch"), "depth": l.get("depth"),
             "downstream_kw": l.get("downstream_kw"), "size_class": l.get("size_class"),
             "on_backbone": l.get("on_backbone")} for i, l in enumerate(selected, start=1)]


def boi():
    from core.ieee123_model import extract_all
    from placement.feeder_context import FeederContext
    from placement.feeder_adapter import network_and_loads_from_core_extraction
    from placement.placement_strategy import (HybridPlacement, RandomPlacement, DegreePlacement,
                                              CommunityDetectionPlacement)
    from observability.state_vector import build_state_vector, complex_v_from_state
    from observability import jacobian as J
    from placement.electrical_distance import get_system_y_index

    with contextlib.redirect_stdout(io.StringIO()):
        net, loads = network_and_loads_from_core_extraction(extract_all())
    ctx = FeederContext(net, loads)
    ctx.annotate_loads(loads)

    sets = {
        "Set A (frozen)": [json.loads((BASE_DIR / "simulation" / "representative_meters.json").read_text())],
        "HybridV2 (set B)": [json.loads((BASE_DIR / "simulation" / "representative_meters_B.json").read_text())],
    }
    pick = lambda s: _meters_from_selection(s.select([dict(l) for l in loads], ctx))
    sets["Hybrid v1 (current code)"] = [pick(HybridPlacement(N_METERS))]
    sets["Degree"] = [pick(DegreePlacement(N_METERS))]
    CommunityDetectionPlacement.clear_cache()
    sets["Community4"] = [pick(CommunityDetectionPlacement(N_METERS, num_communities=4))]
    sets["Community12"] = [pick(CommunityDetectionPlacement(N_METERS, num_communities=None))]
    sets["Random"] = [pick(RandomPlacement(N_METERS, seed=s)) for s in RANDOM_SEEDS]

    # registries first (generate_registry recompiles the circuit), then the Jacobian
    with tempfile.TemporaryDirectory() as d:
        regs = {name: [_registry_for(m, d) for m in ms] for name, ms in sets.items()}
    state = build_state_vector()
    Y, node_index = get_system_y_index()
    flat = complex_v_from_state(state, state.x0_v, state.x0_delta)
    n_v = state.n_phase_nodes

    results = {}
    for name, rlist in regs.items():
        per = []
        for entries in rlist:
            H, _ = J.build_jacobian(entries, Y, node_index, state, at_v=flat)
            Hv = np.abs(H[:, :n_v])
            tol = 1e-9 * Hv.max(axis=1, keepdims=True)
            boi_k = (Hv > np.maximum(tol, 1e-12)).sum(axis=0)
            per.append({"m": len(entries), "SORI": int(boi_k.sum()), "coverage": float((boi_k >= 1).mean()),
                        "nodes_unseen": int((boi_k == 0).sum()), "BOI_mean": float(boi_k.mean()),
                        "BOI_max": int(boi_k.max())})
        agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        if len(per) > 1:
            agg.update({f"{k}_std": float(np.std([p[k] for p in per])) for k in ("SORI", "coverage", "nodes_unseen")})
        agg["n_runs"] = len(per)
        results[name] = agg
    results["_meta"] = {"source": "validation/literature_metrics.py boi", "n_meters": N_METERS,
                        "n_phase_nodes": n_v, "evaluated_at": "flat start",
                        "measurements": "multiphase registry rules: meter P,Q,|V| + substation head P,Q,|V|"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "boi_sori.json").write_text(json.dumps(results, indent=2))
    _plot_boi(results)
    print(json.dumps(results, indent=2))


def _plot_boi(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in COLORS if n in res]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, key, title, fmt in ((axes[0], "SORI", "SORI (total redundancy) — higher is better", "{:.0f}"),
                                (axes[1], "coverage", "Phase-nodes seen by ≥1 measurement — higher is better", "{:.1%}")):
        vals = [res[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names], width=0.65)
        if "Random" in names and f"{key}_std" in res["Random"]:
            i = names.index("Random")
            ax.errorbar([i], [res["Random"][key]], yerr=[res["Random"][f"{key}_std"]], color="#333", capsize=5)
        for i, (b, v) in enumerate(zip(bars, vals)):
            top = v + (res["Random"].get(f"{key}_std", 0) if names[i] == "Random" else 0)  # clear the error bar
            ax.text(b.get_x() + b.get_width() / 2, top, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_axisbelow(True)
        if key == "coverage":
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"BOI / SORI at {res['_meta']['n_meters']} meters (all methods)", fontweight="bold")
    fig.savefig(OUT / "boi_sori.png", dpi=160)


def accuracy():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(BASE_DIR / "reports" / "measurements" / "paired_A_all91_vs_B_all91.npz", allow_pickle=True)
    vA, vB = d["v_err_A"], d["v_err_B"]                    # (points, v-nodes), relative |V| error
    aA, aB = np.degrees(d["ang_err_A"]), np.degrees(d["ang_err_B"])  # (points, delta-nodes)
    v_nodes, d_nodes = list(d["v_nodes"]), list(d["delta_nodes"])
    d_idx = np.array([v_nodes.index(n) for n in d_nodes])

    def prob_index(v, a, lim, ang):
        both = (v[:, d_idx] <= lim) & (a <= ang)
        return both.mean(axis=0)  # per delta-node share of points within both limits

    res = {"source": "validation/literature_metrics.py accuracy (from validation/paired_ab.py arrays)",
           "n_points": int(vA.shape[0]), "limits": [list(x) for x in LIMITS]}
    for lab, v, a in (("A", vA, aA), ("B", vB, aB)):
        e2 = (v ** 2)
        M = (v.T @ v) / v.shape[0]
        r = {"angle_err_deg": {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())},
             "A_opt_trace_mse": float(e2.mean(axis=0).sum()),
             "E_opt_max_eig": float(np.linalg.eigvalsh(M).max())}
        for lim in (0.01, 0.02, 0.05):
            r[f"share_within_V{lim:.0%}_only"] = float((v <= lim).mean())
        for lim, ang in LIMITS:
            p = prob_index(v, a, lim, ang)
            r[f"prob_within_V{lim:.0%}_ang{ang:g}deg"] = {
                "median_over_nodes": float(np.median(p)), "worst_node": float(p.min()),
                "share_nodes_ge_95pct": float((p >= 0.95).mean()), "mean_over_nodes": float(p.mean())}
        res[lab] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accuracy_extras.json").write_text(json.dumps(res, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    c = {"A": "#eba53a", "B": "#2a78d6"}
    lab = {"A": "Set A (v1)", "B": "Set B (HybridV2)"}
    for k, v in (("A", vA), ("B", vB)):
        x = np.sort(v.ravel()); axes[0].plot(x, np.linspace(0, 1, len(x)), color=c[k], lw=2, label=lab[k])
    axes[0].set_xlabel("relative |V| error"); axes[0].set_title("|V| error, all nodes × 60 points (CDF)", fontweight="bold")
    for k, a in (("A", aA), ("B", aB)):
        x = np.sort(a.ravel()); axes[1].plot(x, np.linspace(0, 1, len(x)), color=c[k], lw=2, label=lab[k])
    axes[1].set_xlabel("angle error (degrees)"); axes[1].set_title("Angle error (CDF)", fontweight="bold")
    lim, ang = PLOT_LIMIT
    for k, v, a in (("A", vA, aA), ("B", vB, aB)):
        p = np.sort(prob_index(v, a, lim, ang)); axes[2].plot(p, np.linspace(0, 1, len(p)), color=c[k], lw=2, label=lab[k])
    axes[2].set_xlabel(f"share of points with |V| err ≤ {lim:.0%} and angle err ≤ {ang:g}°")
    axes[2].set_title("Singh-style per-bus index (CDF over buses)", fontweight="bold")
    for ax in axes:
        ax.set_ylabel("fraction"); ax.legend(frameon=False); ax.grid(color="#e4e3df"); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "accuracy_extras.png", dpi=160)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["boi", "accuracy"])
    a = p.parse_args()
    boi() if a.part == "boi" else accuracy()
