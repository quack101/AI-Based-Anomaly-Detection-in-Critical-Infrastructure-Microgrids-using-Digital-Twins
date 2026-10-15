"""
What changes between the frozen 49-meter set A and the 45-meter set, and
what the measurements say about 30 / 45 / 49.

  python -m visualization.set_comparison

Left: the feeder with buses kept, dropped (in A only) and added (in the
45-meter set only). Right: bars read from reports/measurements/*.json
(written by validation/measure_meter_set.py). If those files are absent
the panel says so; no value is hardcoded.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.feeder_data import (
    FIGURES_DIR, load_feeder, load_measurement, load_meter_set, metered_buses,
)

SETS = [("Aall91", "49 (set A)", "#34495e"), ("H45all91", "45", "#2e9e5b"), ("H30all91", "30", "#c0392b")]


def _metrics(label):
    d = load_measurement(label)
    if d is None:
        return None
    sc, bd, ns = d["scorecard"], d["bdd_clean"], d["nullspace"]["points"][0]
    return {
        "scorecard converged (of 60)": sc["converged"],
        "clean points converged (of 90)": bd["converged"],
        "mean iterations": sc["mean_iterations"],
        "null-space dimension": ns["nullity"],
        "measured dof (median)": sc["dof_measured_median"],
    }


def main():
    feeder = load_feeder()
    a = metered_buses(load_meter_set("A"))
    b = metered_buses(load_meter_set("H45"))
    kept, dropped, added = a & b, a - b, b - a

    fig = plt.figure(figsize=(17, 8.5))
    ax = fig.add_axes([0.01, 0.06, 0.52, 0.86])
    ax.set_aspect("equal")
    ax.axis("off")
    for u, v in feeder.edges:
        ax.plot([feeder.pos[u][0], feeder.pos[v][0]], [feeder.pos[u][1], feeder.pos[v][1]],
                "-", color="#ccc", lw=1, zorder=1)
    for bus in feeder.buses:
        ax.scatter(*feeder.pos[bus], s=8, color="#bbb", zorder=2)
    for bus in kept:
        ax.scatter(*feeder.pos[bus], s=70, color="#2e9e5b", edgecolors="white", zorder=3)
    for bus in dropped:
        ax.scatter(*feeder.pos[bus], s=150, marker="X", color="#c0392b", edgecolors="black", zorder=4)
    for bus in added:
        ax.scatter(*feeder.pos[bus], s=150, marker="P", color="#1f77b4", edgecolors="black", zorder=4)
    ax.scatter(*feeder.pos[feeder.source], marker="*", s=380, color="#f1c40f", edgecolors="black", zorder=5)
    ax.set_title(f"Set A (49) vs 45-meter set: {len(kept)} buses kept, {len(dropped)} dropped, "
                 f"{len(added)} added", fontsize=13, loc="left")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color="#2e9e5b", label=f"in both ({len(kept)})"),
        plt.Line2D([], [], marker="X", ls="", color="#c0392b", markeredgecolor="black",
                   label=f"set A only ({len(dropped)})"),
        plt.Line2D([], [], marker="P", ls="", color="#1f77b4", markeredgecolor="black",
                   label=f"45-meter set only ({len(added)})"),
    ], loc="lower left", fontsize=10)

    data = {lab: _metrics(lab) for lab, _, _ in SETS}
    names = ["scorecard converged (of 60)", "clean points converged (of 90)", "mean iterations",
             "null-space dimension", "measured dof (median)"]
    if all(v is None for v in data.values()):
        fig.text(0.62, 0.5, "No measurements found.\nRun validation/measure_meter_set.py\n"
                 "(reports/measurements/ is git-ignored).", fontsize=12)
    else:
        for i, name in enumerate(names):
            axb = fig.add_axes([0.58 + (i % 2) * 0.21, 0.66 - (i // 2) * 0.28, 0.17, 0.20])
            vals = [(data[lab] or {}).get(name) for lab, _, _ in SETS]
            xs = list(range(len(SETS)))
            axb.bar(xs, [v if v is not None else 0 for v in vals], color=[c for _, _, c in SETS])
            for x, v in zip(xs, vals):
                axb.text(x, (v or 0), "n/a" if v is None else f"{v:.1f}".rstrip("0").rstrip("."), ha="center", va="bottom", fontsize=9)
            axb.set_xticks(xs, [t for _, t, _ in SETS], fontsize=9)
            axb.set_title(name, fontsize=10)
            axb.spines[["top", "right"]].set_visible(False)
        fig.text(0.58, 0.005, "Same physics (all 91 loads driven), frozen estimator. 30 meters: "
                 "5 of 60 scorecard points hit the 40-iteration cap.\nSource: "
                 "reports/measurements/*.json; see reports/meter_count_30_45_49_measured.md.",
                 fontsize=9)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "set_comparison_A_vs_H45.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    print("dropped buses:", sorted(dropped, key=lambda x: (len(x), x)))
    print("added buses:  ", sorted(added, key=lambda x: (len(x), x)))


if __name__ == "__main__":
    main()
