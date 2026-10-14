"""
Print the side-by-side table for reports/placement_A_vs_B_measured.md
from reports/measurements/<label>.json. Every number is read from those
files (written by validation/measure_meter_set.py); nothing is typed in.

Run: python -m validation.ab_table
"""

import json
from pathlib import Path

M = Path(__file__).resolve().parent.parent / "reports" / "measurements"
COLS = ["A_exact", "A_rebuilt", "A_all91", "B_all91"]

ROWS = [
    # (label, section, key, format)
    ("|V| error median (pu)", "scorecard", "v_err_pu_median", "{:.5f}"),
    ("|V| error p95 (pu)", "scorecard", "v_err_pu_p95", "{:.5f}"),
    ("|V| error max (pu)", "scorecard", "v_err_pu_max", "{:.5f}"),
    ("converged (of 60)", "scorecard", "converged", "{}"),
    ("mean iterations", "scorecard", "mean_iterations", "{:.2f}"),
    ("weakest node", "weakest_nodes", None, None),
    ("weakest node median error (pu)", "weakest_nodes", "worst_node_median_err_pu", "{:.5f}"),
    ("median over nodes of median error (pu)", "weakest_nodes", "median_over_nodes_of_median_err_pu", "{:.5f}"),
    ("null-space dim (unattackable rows)", "nullspace", None, None),
    ("J mean [not for verdict]", "scorecard", "J_mean", "{:.4g}"),
    ("J mean / 57 [not for verdict]", "scorecard", "J_over_ref_dof57", "{:.1f}"),
    ("measured dof (median)", "scorecard", "dof_measured_median", "{}"),
    ("J99 clean, n=90 [not for verdict]", "bdd_clean", "J99", "{:.2f}"),
    ("LNR99 clean, n=90", "bdd_clean", "LNR99", "{:.2f}"),
    ("HEAD_P_A mean signed NR", "head_p", "HEAD_P_A_mean_signed_nr", "{:+.1f}"),
    ("HEAD_P_B mean signed NR", "head_p", "HEAD_P_B_mean_signed_nr", "{:+.1f}"),
    ("HEAD_P_C mean signed NR", "head_p", "HEAD_P_C_mean_signed_nr", "{:+.1f}"),
    ("share of J from HEAD_P rows (mean)", "head_p", "J_share_head_p_mean", "{:.1%}"),
]


def cell(d, sec, key, fmt, label):
    if d is None or sec not in d:
        return "n/a"
    if label == "weakest node":
        return d[sec]["top10"][0]["node"]
    if label.startswith("null-space"):
        dims = sorted({p["nullity"] for p in d[sec]["points"]})
        rows = sorted({p["n_unattackable_rows"] for p in d[sec]["points"]})
        return f"{'/'.join(map(str, dims))} ({'/'.join(map(str, rows))})"
    v = d[sec].get(key)
    return "n/a" if v is None else fmt.format(v)


def main():
    data = {c: (json.loads((M / f"{c}.json").read_text()) if (M / f"{c}.json").exists() else None) for c in COLS}
    print("| metric | " + " | ".join(COLS) + " | source |")
    print("|---|" + "---|" * len(COLS) + "---|")
    for label, sec, key, fmt in ROWS:
        src = next((d[sec]["source"] for d in data.values() if d and sec in d and "source" in d[sec]), "")
        src = src.split(" (")[0] if sec != "nullspace" else src
        print(f"| {label} | " + " | ".join(cell(data[c], sec, key, fmt, label) for c in COLS) + f" | `{src}` |")
    print()
    print("Inputs: " + "; ".join(f"{c}: registry `{Path(d['registry']).name}`, corpus `{Path(d['corpus']).name}`"
                                 for c, d in data.items() if d))


if __name__ == "__main__":
    main()
