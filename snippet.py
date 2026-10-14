"""
Placement-quality comparison chart: Hybrid vs baselines, by meter count.

Reads directly from the real sweep artifact -- nothing here is hardcoded.
Produces two panels (independent construction, nested construction) side
by side, because Phase 4 Tier 1 found the placement-quality knee and the
Community12 coverage behaviour both depend on which construction is used --
collapsing them into one panel would hide that.

Usage:
    python plot_placement_comparison.py --metric branch_diversity
    python plot_placement_comparison.py --metric phase_dev_pp
    python plot_placement_comparison.py --metric downstream_coverage_frac
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- fixed knowns, not fitted from data ----
CSV_PATH = "artifacts/20260822T092205Z/sweep_4a_tier1/cells.csv"
PHASE_BASELINE = {"A": 1400.0, "B": 952.5, "C": 1137.5}
PHASE_TOTAL = sum(PHASE_BASELINE.values())
PHASE_TARGET_SHARE = {p: v / PHASE_TOTAL for p, v in PHASE_BASELINE.items()}

DETERMINISTIC_STRATEGIES = ["Hybrid", "Degree", "Community12", "Community4"]
COLORS = {
    "Hybrid": "#E8A33D",
    "Random": "#9FB0C0",
    "Degree": "#7F77DD",
    "Community12": "#3F8F5E",
    "Community4": "#B5453C",
}


def compute_phase_dev_pp(row):
    """Max absolute deviation, in percentage points, from the feeder's own
    per-phase kW baseline. This is the formula independently re-derived and
    verified against the sweep report's own cited figures (5.52 / 4.69 at
    n=15; 0.9732 at n=85) before use -- do not change without re-verifying."""
    total = row["phase_kw_A"] + row["phase_kw_B"] + row["phase_kw_C"]
    if total == 0:
        return np.nan
    shares = {
        "A": row["phase_kw_A"] / total,
        "B": row["phase_kw_B"] / total,
        "C": row["phase_kw_C"] / total,
    }
    return max(abs(shares[p] - PHASE_TARGET_SHARE[p]) * 100 for p in shares)


METRIC_CONFIG = {
    "branch_diversity": {
        "column": "branch_diversity",
        "ylabel": "Branch diversity (raw count of distinct branches)",
        "title": "Branch diversity vs meter count",
        "derive": None,
    },
    "downstream_coverage_frac": {
        "column": "downstream_coverage_frac",
        "ylabel": "Downstream load coverage (fraction of 3490 kW)",
        "title": "Downstream coverage vs meter count",
        "derive": None,
    },
    "phase_dev_pp": {
        "column": "phase_dev_pp",
        "ylabel": "Max phase-balance deviation (percentage points, lower is better)",
        "title": "Phase-balance deviation vs meter count",
        "derive": compute_phase_dev_pp,
    },
}


def load_data(csv_path, metric_key):
    df = pd.read_csv(csv_path)

    cfg = METRIC_CONFIG[metric_key]
    if cfg["derive"] is not None:
        df[cfg["column"]] = df.apply(cfg["derive"], axis=1)

    return df, cfg


def build_panel(ax, df, construction, metric_col, ylabel_lower_is_better=False):
    sub = df[df["construction"] == construction]

    counts = sorted(sub["n_selected"].unique())

    # Random: mean +/- std across seeds, per count -- never a single draw
    random_rows = sub[sub["strategy"] == "Random"]
    random_mean = random_rows.groupby("n_selected")[metric_col].mean().reindex(counts)
    random_std = random_rows.groupby("n_selected")[metric_col].std().reindex(counts)

    ax.plot(counts, random_mean.values, color=COLORS["Random"], marker="o",
             linewidth=2, label=f"Random (mean of {random_rows['seed'].nunique()} seeds)")
    ax.fill_between(counts,
                     (random_mean - random_std).values,
                     (random_mean + random_std).values,
                     color=COLORS["Random"], alpha=0.15, linewidth=0)

    for strat in DETERMINISTIC_STRATEGIES:
        strat_rows = sub[sub["strategy"] == strat]
        if strat_rows.empty:
            continue
        vals = strat_rows.groupby("n_selected")[metric_col].mean().reindex(counts)
        label = "Hybrid (ours)" if strat == "Hybrid" else strat
        lw = 2.5 if strat == "Hybrid" else 1.5
        z = 5 if strat == "Hybrid" else 3
        ax.plot(counts, vals.values, color=COLORS[strat], marker="o",
                 linewidth=lw, label=label, zorder=z)

    ax.set_xlabel("Meters placed")
    ax.set_title(f"{construction.capitalize()} construction", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=list(METRIC_CONFIG.keys()),
                         default="branch_diversity")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--out", default=None,
                         help="output path; if omitted, shows interactively")
    args = parser.parse_args()

    df, cfg = load_data(args.csv, args.metric)

    constructions = sorted(df["construction"].unique())
    fig, axes = plt.subplots(1, len(constructions), figsize=(6.5 * len(constructions), 5),
                              sharey=True)
    if len(constructions) == 1:
        axes = [axes]

    for ax, construction in zip(axes, constructions):
        build_panel(ax, df, construction, cfg["column"])

    axes[0].set_ylabel(cfg["ylabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06),
               frameon=False, fontsize=9)
    fig.suptitle(cfg["title"], y=1.14, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02,
              f"Source: {args.csv}  --  every value read directly, none hardcoded",
              ha="center", fontsize=8, color="#5B6B7C")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"written: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()