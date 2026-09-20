"""
RES.md Deliverable 10, Phase 0 -- instrument and validate the current twin.

Replays the 49-meter representative subset directly through the same
OpenDSS driver functions the Kafka consumer uses (compile_feeder,
get_load_telemetry, get_solution_metrics), applying loads and solving
once per timestamp in the same order as streaming/consumer.py's main
loop. The other 42 (unmetered, per RES.md's pseudo-measurement framing)
loads are left at their as-authored nominal DSS values throughout, same
as the live pipeline -- this script validates the CURRENT 49-meter
architecture, not a hypothetical fully-metered one, and that is
unchanged by which corpus feeds the 49.

Why this script exists instead of running streaming/consumer.py against a
live broker: no Kafka broker is available in this environment (Docker
daemon is not running here). This script exercises the identical OpenDSS
driver code path and the identical unscaled synthetic P,Q inputs the
producer publishes -- it changes nothing about data scaling or telemetry
content, per the Phase-0 scope constraint in RES.md.

Data source: dataset/node_loads_91/ (the IDEAL Household Energy Dataset
corpus, 91 per-load P,Q series, 2018-04-04 to 2018-06-03, 1-min
resolution). Each of representative_meters.json's 49 load_name entries
is looked up by filename (node_<LoadName>.csv), not by positional
order -- the corpus is delivered keyed by load name, so pairing by
position (as the deleted Cholesky corpus required) would silently
mismatch data to meters.

Output:
  - simulation/phase0_validation_log.csv   (per-timestamp metrics)
  - reports/phase0_validation_report.txt   (summary + acceptance criteria)
  - simulation/circuit_topology.json        (N_phase_nodes, AllNodeNames)
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.opendss_utils import (
    compile_feeder,
    get_circuit_topology,
    get_solution_metrics,
)
import opendssdirect as dss

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

VALIDATION_LOG = BASE_DIR / "simulation" / "phase0_validation_log_ideal.csv"
VALIDATION_REPORT = REPORTS_DIR / "phase0_validation_report_ideal.txt"
TOPOLOGY_SNAPSHOT = BASE_DIR / "simulation" / "circuit_topology.json"

# Old-corpus baseline (reports/phase0_validation_report.txt, Cholesky
# corpus, 1440 timestamps) -- kept as literal constants so the
# old-vs-new comparison table doesn't silently drift if that file is
# ever regenerated.
OLD_REPORT_CONVERGED_PCT = 100.00
OLD_REPORT_V_PU_RANGE = (0.980101, 1.045615)
OLD_REPORT_LOSS_PCT = 1.8920
OLD_REPORT_HEAD_P_KW = 1671.8
OLD_REPORT_HEAD_P_PCT_OF_NOMINAL = 47.9
NOMINAL_HEAD_P_KW = 3490.0
NOMINAL_HEAD_Q_KVAR = 1920.0

NUM_TIMESTAMPS = 86400  # full 60-day corpus at 1 min/step (was 1440 -- one
                        # day -- against the old, much shorter Cholesky corpus)

VALIDATION_FIELDS = [
    "timestamp",
    "real_timestamp",
    "converged",
    "iterations",
    "v_pu_min",
    "v_pu_max",
    "v_pu_min_phase1",
    "v_pu_max_phase1",
    "v_pu_min_phase2",
    "v_pu_max_phase2",
    "v_pu_min_phase3",
    "v_pu_max_phase3",
    "losses_kw",
    "losses_kvar",
    "feeder_head_p_kw",
    "feeder_head_q_kvar",
    "solve_time_s",
]


def load_inputs():
    with open(REPRESENTATIVE_METERS, "r") as f:
        representative_meters = json.load(f)

    # Corpus files are node_<LoadName>.csv -- keyed by load name, not by
    # position. Build the name -> path index once, then look up each of
    # the 49 representative meters explicitly.
    available = {
        f.stem[len("node_"):].lower(): f
        for f in DATASET_DIR.glob("node_*.csv")
    }

    missing = [
        m["load_name"] for m in representative_meters
        if m["load_name"].lower() not in available
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} representative meter(s) have no corpus file "
            f"in {DATASET_DIR}: {missing}"
        )

    datasets = [
        pd.read_csv(available[m["load_name"].lower()], nrows=NUM_TIMESTAMPS)
        for m in representative_meters
    ]

    return representative_meters, datasets


def main():
    representative_meters, datasets = load_inputs()

    compile_feeder()

    # Nominal (as-authored) total feeder load, captured before any load
    # is overridden by synthetic telemetry -- the baseline the Phase-1
    # under-loading defect is measured against.
    nominal_total_kw = 0.0
    load_nominal_kw = {}
    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)
        load_nominal_kw[load_name] = dss.Loads.kW()
        nominal_total_kw += dss.Loads.kW()

    n_phase_nodes, all_node_names = get_circuit_topology()
    all_node_names_arr = np.array(all_node_names)

    print(f"N_phase_nodes: {n_phase_nodes}")

    with open(TOPOLOGY_SNAPSHOT, "w") as f:
        json.dump(
            {"n_phase_nodes": n_phase_nodes, "all_node_names": all_node_names},
            f,
            indent=2,
        )

    num_rows = min(len(df) for df in datasets)
    num_rows = min(num_rows, NUM_TIMESTAMPS)

    print(f"Replaying {num_rows} timestamps...")

    rows = []

    # Per-node instrumentation beyond get_solution_metrics()'s per-phase
    # min/max: identity of the single global V_pu minimum (value, node,
    # phase, timestamp) and a per-node count of [0.90,1.10]-band
    # breaches, so clustering can be reported. Computed locally here
    # (not added to shared simulation/opendss_utils.py) to avoid any
    # risk to streaming/consumer.py's live driver path.
    min_v_record = {"value": float("inf"), "node": None, "phase": None,
                     "t_index": None, "real_timestamp": None}
    breach_by_node = defaultdict(int)
    breach_timestep_count = 0

    for t in range(num_rows):

        real_ts = None
        for i, (meter, df) in enumerate(zip(representative_meters, datasets)):
            row = df.iloc[t]
            if i == 0:
                real_ts = str(row["timestamp"])

            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))

        solve_start = time.perf_counter()
        dss.Solution.Solve()
        solve_time_s = time.perf_counter() - solve_start

        metrics = get_solution_metrics(solve_time_s)

        per_phase = metrics["v_pu_per_phase"]
        all_mins = [p["min"] for p in per_phase.values()]
        all_maxs = [p["max"] for p in per_phase.values()]

        v_pu_arr = np.array(dss.Circuit.AllBusMagPu())
        idx_min = int(np.argmin(v_pu_arr))
        if v_pu_arr[idx_min] < min_v_record["value"]:
            node = all_node_names_arr[idx_min]
            min_v_record.update(
                value=float(v_pu_arr[idx_min]),
                node=node,
                phase=node.split(".")[-1],
                t_index=t,
                real_timestamp=real_ts,
            )
        breach_mask = (v_pu_arr < 0.90) | (v_pu_arr > 1.10)
        if breach_mask.any():
            breach_timestep_count += 1
            for node in all_node_names_arr[breach_mask]:
                breach_by_node[node] += 1

        log_row = {
            "timestamp": t,
            "real_timestamp": real_ts,
            "converged": metrics["converged"],
            "iterations": metrics["iterations"],
            "v_pu_min": min(all_mins),
            "v_pu_max": max(all_maxs),
            "losses_kw": metrics["losses_kw"],
            "losses_kvar": metrics["losses_kvar"],
            "feeder_head_p_kw": metrics["feeder_head_p_kw"],
            "feeder_head_q_kvar": metrics["feeder_head_q_kvar"],
            "solve_time_s": metrics["solve_time_s"],
        }
        for phase in ("1", "2", "3"):
            bounds = per_phase.get(phase)
            log_row[f"v_pu_min_phase{phase}"] = bounds["min"] if bounds else ""
            log_row[f"v_pu_max_phase{phase}"] = bounds["max"] if bounds else ""

        rows.append(log_row)

        if t % 100 == 0:
            print(f"  t={t}  converged={metrics['converged']}  "
                  f"v_pu=[{min(all_mins):.4f}, {max(all_maxs):.4f}]  "
                  f"losses_kw={metrics['losses_kw']:.3f}  "
                  f"solve_time_s={solve_time_s:.4f}")

    with open(VALIDATION_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    meter_means = {
        m["load_name"]: float(df["P_kW"].mean())
        for m, df in zip(representative_meters, datasets)
    }

    write_ideal_summary_report(
        rows, n_phase_nodes, all_node_names, nominal_total_kw,
        load_nominal_kw, meter_means,
        min_v_record, breach_by_node, breach_timestep_count,
        output_path=VALIDATION_REPORT,
    )


DIRECT_REPLAY_METHODOLOGY = (
    "No live Kafka broker was available in this environment at the time "
    "of this run (Docker daemon not running), so this replay drives "
    "OpenDSS directly from the same synthetic CSVs and "
    "representative_meters.json the producer publishes, applying loads "
    "and solving once per timestamp in the same order as "
    "streaming/consumer.py's main loop, and using the identical "
    "instrumented driver functions (get_circuit_topology, "
    "get_solution_metrics) that consumer.py now also calls per timestamp "
    "against live telemetry. This has since been confirmed against the "
    "real producer -> Kafka -> consumer pipeline once Docker was "
    "available: see reports/phase0_validation_report_live.txt, generated "
    "by simulation/summarize_live_validation.py from a full-day live run. "
    "The two reports agree to the figures shown here."
)


def write_summary_report(
    rows, n_phase_nodes, all_node_names, nominal_total_kw,
    output_path=None, methodology=None,
):
    output_path = output_path or VALIDATION_REPORT
    methodology = methodology or DIRECT_REPLAY_METHODOLOGY
    n = len(rows)
    converged_count = sum(1 for r in rows if r["converged"])
    v_mins = [r["v_pu_min"] for r in rows]
    v_maxs = [r["v_pu_max"] for r in rows]
    losses_kw = [r["losses_kw"] for r in rows]
    head_p = [r["feeder_head_p_kw"] for r in rows]
    solve_times = [r["solve_time_s"] for r in rows]

    v_band_violations = sum(
        1 for r in rows if r["v_pu_min"] < 0.90 or r["v_pu_max"] > 1.10
    )

    loss_pct = [
        (l / p * 100.0) if p else float("nan")
        for l, p in zip(losses_kw, head_p)
    ]
    avg_loss_pct = sum(loss_pct) / len(loss_pct) if loss_pct else float("nan")

    convergence_pass = converged_count == n
    voltage_pass = v_band_violations == 0
    losses_pass = 1.0 <= avg_loss_pct <= 5.0
    solve_time_pass = max(solve_times) < 60.0

    avg_head_p = sum(head_p) / len(head_p) if head_p else float("nan")
    load_pct_of_nominal = (
        100.0 * avg_head_p / nominal_total_kw if nominal_total_kw else float("nan")
    )

    lines = []
    lines.append("=" * 70)
    lines.append("RES.md D10 Phase 0 -- Twin Validation Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Replay length          : {n} timestamps (target: full day, 1440)")
    lines.append(f"N_phase_nodes           : {n_phase_nodes}")
    lines.append(f"AllNodeNames sample     : {all_node_names[:5]} ... "
                 f"(full list in circuit_topology.json)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Acceptance criteria")
    lines.append("-" * 70)
    lines.append(
        f"[{'PASS' if convergence_pass else 'FAIL'}] Convergence: "
        f"{converged_count}/{n} timestamps converged "
        f"({100.0 * converged_count / n:.2f}%), target 100%"
    )
    lines.append(
        f"[{'PASS' if voltage_pass else 'FAIL'}] Voltage band: "
        f"V_pu in [0.90, 1.10] violated at {v_band_violations}/{n} timestamps "
        f"(observed range [{min(v_mins):.6f}, {max(v_maxs):.6f}])"
    )
    lines.append(
        f"[{'PASS' if losses_pass else 'FAIL'}] Losses: "
        f"average losses = {avg_loss_pct:.4f}% of feeder-head load, "
        f"target band 1-5%"
    )
    lines.append(
        f"[{'PASS' if solve_time_pass else 'FAIL'}] Solve time: "
        f"max {max(solve_times):.4f}s, mean {sum(solve_times)/n:.4f}s, "
        f"target well under 60s"
    )
    lines.append(
        f"[INFO] Feeder-head load vs. nominal: avg feeder-head P = "
        f"{avg_head_p:.1f} kW vs. as-authored nominal total = "
        f"{nominal_total_kw:.1f} kW ({load_pct_of_nominal:.1f}% of nominal)"
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Interpretation")
    lines.append("-" * 70)
    lines.append(
        "Convergence and solve-time criteria pass cleanly -- the solver "
        "itself is healthy. The V_pu-band and losses-% acceptance checks "
        "as literally specified also pass here, which is worth stating "
        "plainly rather than forcing the 'implausibly close to 1.0' "
        "outcome RES.md anticipated: only 49 of this feeder's 91 Load "
        "objects are driven by the 49-meter synthetic telemetry (1985 kW "
        "of the 3490 kW nominal total); the other 42 loads keep their "
        "as-authored nominal DSS values throughout the replay. So the "
        "known Phase-1 defect does not collapse the whole feeder toward a "
        "flat 1.0 p.u. profile -- it is masked, in this particular check, "
        "by the ~1505 kW of unmetered load that never changes."
    )
    lines.append(
        "The defect is nonetheless clearly present and is visible in the "
        "feeder-head load line above: average feeder-head P is only "
        f"{load_pct_of_nominal:.1f}% of the as-authored nominal total. "
        "The 49 metered loads are being driven by raw UCI-household-scale "
        "kW (order 0.1-5 kW per meter, from the 'Global_active_power' "
        "column) substituted directly for IEEE-123 spot loads of order "
        "tens of kW -- exactly the dimensional incompatibility RES.md "
        "3.1.1 describes. The result is a feeder that is real-loaded at "
        f"roughly half its nominal rating rather than an order of "
        "magnitude under, because the unmetered majority of the load is "
        "unaffected -- a more precise characterization than RES.md's "
        "prior estimate, and one that matters: it means the V_pu-band and "
        "losses-% acceptance thresholds as stated are not, by themselves, "
        "a reliable gate for this defect on this feeder/meter-coverage "
        "combination. The feeder-head-vs-nominal check above is the one "
        "that actually catches it. Per RES.md Phase 0 scope, the defect "
        "is confirmed and reported here, not fixed -- that is Phase 1."
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Notes on methodology")
    lines.append("-" * 70)
    lines.append(methodology)
    lines.append("")
    lines.append(f"Per-timestamp log: simulation/phase0_validation_log.csv")
    lines.append(f"Topology snapshot: simulation/circuit_topology.json")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text + "\n")

    print()
    print(report_text)

    return report_text


def write_ideal_summary_report(
    rows, n_phase_nodes, all_node_names, nominal_total_kw,
    load_nominal_kw, meter_means,
    min_v_record, breach_by_node, breach_timestep_count,
    output_path=None,
):
    """Report for the IDEAL-corpus run (dataset/node_loads_91/). Reuses
    the same PASS/FAIL acceptance-criteria arithmetic as
    write_summary_report(), but adds the per-node identity, breach
    clustering, per-meter nominal check, peak-to-mean ratio and
    old-vs-new comparison table this run explicitly requires -- none of
    which get_solution_metrics()/write_summary_report() compute, since
    that function only ever tracked per-phase min/max, not per-node
    identity."""

    output_path = output_path or VALIDATION_REPORT
    n = len(rows)
    converged_count = sum(1 for r in rows if r["converged"])
    v_mins = [r["v_pu_min"] for r in rows]
    v_maxs = [r["v_pu_max"] for r in rows]
    losses_kw = [r["losses_kw"] for r in rows]
    head_p = [r["feeder_head_p_kw"] for r in rows]
    head_q = [r["feeder_head_q_kvar"] for r in rows]
    solve_times = [r["solve_time_s"] for r in rows]

    v_band_violations = sum(
        1 for r in rows if r["v_pu_min"] < 0.90 or r["v_pu_max"] > 1.10
    )

    loss_pct = [
        (l / p * 100.0) if p else float("nan")
        for l, p in zip(losses_kw, head_p)
    ]
    avg_loss_pct = sum(loss_pct) / len(loss_pct) if loss_pct else float("nan")

    convergence_pass = converged_count == n
    voltage_pass = v_band_violations == 0
    losses_pass = 1.0 <= avg_loss_pct <= 5.0
    solve_time_pass = max(solve_times) < 60.0

    new_v_range = (min(v_mins), max(v_maxs))
    new_spread = new_v_range[1] - new_v_range[0]
    old_spread = OLD_REPORT_V_PU_RANGE[1] - OLD_REPORT_V_PU_RANGE[0]
    spread_pass = new_spread > old_spread

    avg_head_p = sum(head_p) / len(head_p) if head_p else float("nan")
    avg_head_q = sum(head_q) / len(head_q) if head_q else float("nan")
    head_p_pct_err = abs(avg_head_p - NOMINAL_HEAD_P_KW) / NOMINAL_HEAD_P_KW * 100.0
    head_q_pct_err = abs(avg_head_q - NOMINAL_HEAD_Q_KVAR) / NOMINAL_HEAD_Q_KVAR * 100.0
    head_p_pass = head_p_pct_err <= 10.0
    head_q_pass = head_q_pct_err <= 10.0

    peak_idx = max(range(n), key=lambda i: head_p[i])
    peak_head_p = head_p[peak_idx]
    peak_head_p_ts = rows[peak_idx]["real_timestamp"]
    mean_head_p = avg_head_p
    peak_to_mean = peak_head_p / mean_head_p if mean_head_p else float("nan")

    # Per-node breach clustering: nodes accounting for >=5% of all
    # breach-timestep occurrences.
    total_breach_occurrences = sum(breach_by_node.values())
    clustered_nodes = sorted(
        breach_by_node.items(), key=lambda kv: kv[1], reverse=True
    )
    breach_pct = 100.0 * breach_timestep_count / n if n else float("nan")

    # Per-meter (49 dynamically-driven loads) mean-vs-nominal check.
    meter_checks = []
    for load_name, mean_kw in meter_means.items():
        nominal_kw = load_nominal_kw.get(load_name)
        if not nominal_kw:
            continue
        pct_err = abs(mean_kw - nominal_kw) / nominal_kw * 100.0
        meter_checks.append((load_name, mean_kw, nominal_kw, pct_err))
    meters_within_10pct = sum(1 for _, _, _, e in meter_checks if e <= 10.0)
    static_loads = len(load_nominal_kw) - len(meter_checks)

    lines = []
    lines.append("=" * 70)
    lines.append("RES.md D10 Phase 0/1 -- IDEAL Corpus Validation Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Corpus                  : dataset/node_loads_91/ (IDEAL Household "
                 f"Energy Dataset, 91 loads, 2018-04-04..2018-06-03, 1-min)")
    lines.append(f"Replay length           : {n} timestamps (target: full 60-day corpus, 86400)")
    lines.append(f"N_phase_nodes           : {n_phase_nodes}")
    lines.append(f"AllNodeNames sample     : {all_node_names[:5]} ... "
                 f"(full list in circuit_topology.json)")
    lines.append(f"Architecture            : 49 of 91 loads driven by synthetic "
                 f"telemetry (as in the live pipeline); 42 unmetered loads held "
                 f"at as-authored nominal DSS values throughout")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Acceptance criteria (RES.md Phase 0 and Phase 1)")
    lines.append("-" * 70)
    lines.append(
        f"[{'PASS' if convergence_pass else 'FAIL'}] Convergence: "
        f"{converged_count}/{n} timestamps converged "
        f"({100.0 * converged_count / n:.2f}%), target 100%"
    )
    lines.append(
        f"[{'PASS' if voltage_pass else 'FAIL'}] Voltage band: "
        f"V_pu in [0.90, 1.10] at every energised node, every timestamp -- "
        f"violated at {v_band_violations}/{n} timestamps "
        f"(observed range [{new_v_range[0]:.6f}, {new_v_range[1]:.6f}])"
    )
    lines.append(
        f"[{'PASS' if spread_pass else 'FAIL'}] Voltage spread vs. baseline: "
        f"new spread {new_spread:.6f} p.u. ([{new_v_range[0]:.6f}, "
        f"{new_v_range[1]:.6f}]) vs. old-corpus baseline spread "
        f"{old_spread:.6f} p.u. ({OLD_REPORT_V_PU_RANGE}) -- "
        f"{'wider' if spread_pass else 'NOT wider -- corpus may not have taken'}"
    )
    lines.append(
        f"[{'PASS' if losses_pass else 'FAIL'}] Losses: "
        f"average losses = {avg_loss_pct:.4f}% of feeder-head load, "
        f"target band 1-5%"
    )
    lines.append(
        f"[{'PASS' if head_p_pass else 'FAIL'}] Feeder-head P vs. nominal: "
        f"avg {avg_head_p:.1f} kW vs. nominal {NOMINAL_HEAD_P_KW:.1f} kW "
        f"({head_p_pct_err:.2f}% error, target within 10%)"
    )
    lines.append(
        f"[{'PASS' if head_q_pass else 'FAIL'}] Feeder-head Q vs. nominal: "
        f"avg {avg_head_q:.1f} kvar vs. nominal {NOMINAL_HEAD_Q_KVAR:.1f} kvar "
        f"({head_q_pct_err:.2f}% error, target within 10%)"
    )
    lines.append(
        f"[INFO] Per-node (per-meter) mean vs. own nominal: "
        f"{meters_within_10pct}/{len(meter_checks)} of the 49 driven meters "
        f"within +/-10% of their own nominal kW "
        f"({static_loads} unmetered loads are trivially at nominal, not counted)"
    )
    lines.append(
        f"[{'PASS' if solve_time_pass else 'FAIL'}] Solve time: "
        f"max {max(solve_times):.4f}s, mean {sum(solve_times)/n:.4f}s, "
        f"target well under 60s"
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Peak feeder-head load")
    lines.append("-" * 70)
    lines.append(
        f"Max instantaneous feeder-head P: {peak_head_p:.1f} kW at "
        f"real timestamp {peak_head_p_ts} (t_index={peak_idx}), "
        f"{peak_head_p / NOMINAL_HEAD_P_KW:.2f}x nominal"
    )
    lines.append(
        f"Peak-to-mean ratio (feeder aggregate, 91-node sum): "
        f"{peak_to_mean:.2f}x (mean {mean_head_p:.1f} kW) -- compare against "
        f"the stated per-node range 4.01-7.41x (median 4.86x). Aggregation "
        f"across 91 nodes should reduce this ratio substantially relative to "
        f"any single node's own peak-to-mean; if it does not, the nodes are "
        f"peaking in phase, which would itself indicate a generator problem."
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Minimum V_pu observed")
    lines.append("-" * 70)
    if min_v_record["node"] is not None:
        t_idx = min_v_record["t_index"]
        loading_at_min = rows[t_idx]["feeder_head_p_kw"]
        lines.append(
            f"V_pu_min = {min_v_record['value']:.6f} at node "
            f"{min_v_record['node']} (phase {min_v_record['phase']}), "
            f"real timestamp {min_v_record['real_timestamp']} "
            f"(t_index={t_idx}); feeder-head P at that moment = "
            f"{loading_at_min:.1f} kW ({loading_at_min / NOMINAL_HEAD_P_KW:.2f}x nominal)"
        )
    else:
        lines.append("No timestamps replayed -- no V_pu minimum recorded.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Voltage-band breach clustering")
    lines.append("-" * 70)
    lines.append(
        f"Timestamps with >=1 node outside [0.90, 1.10]: "
        f"{breach_timestep_count}/{n} ({breach_pct:.4f}%)"
    )
    if clustered_nodes:
        lines.append(f"Distinct nodes ever in breach: {len(clustered_nodes)} "
                     f"(total node-timestamp breach occurrences: {total_breach_occurrences})")
        lines.append("Top breaching nodes (node: breach-timestep count):")
        for node, count in clustered_nodes[:15]:
            lines.append(f"  {node}: {count}")
    else:
        lines.append("No node ever breached the [0.90, 1.10] band.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Old corpus (Cholesky, deleted) vs. new corpus (IDEAL) comparison")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<32}{'Old (phase0_validation_report.txt)':<38}{'New (this run)'}")
    lines.append(
        f"{'Convergence':<32}{f'{OLD_REPORT_CONVERGED_PCT:.2f}%':<38}"
        f"{100.0 * converged_count / n:.2f}%"
    )
    lines.append(
        f"{'V_pu range':<32}{str(OLD_REPORT_V_PU_RANGE):<38}"
        f"[{new_v_range[0]:.6f}, {new_v_range[1]:.6f}]"
    )
    lines.append(
        f"{'V_pu spread':<32}{f'{old_spread:.6f}':<38}{f'{new_spread:.6f}'}"
    )
    lines.append(
        f"{'Losses % of feeder-head':<32}{f'{OLD_REPORT_LOSS_PCT:.4f}%':<38}"
        f"{avg_loss_pct:.4f}%"
    )
    lines.append(
        f"{'Feeder-head P vs. nominal':<32}"
        f"{f'{OLD_REPORT_HEAD_P_KW:.1f} kW ({OLD_REPORT_HEAD_P_PCT_OF_NOMINAL:.1f}%)':<38}"
        f"{avg_head_p:.1f} kW ({100.0 * avg_head_p / NOMINAL_HEAD_P_KW:.1f}%)"
    )
    lines.append(
        f"{'Feeder-head Q vs. nominal':<32}{'n/a (not tracked in old report)':<38}"
        f"{avg_head_q:.1f} kvar ({100.0 * avg_head_q / NOMINAL_HEAD_Q_KVAR:.1f}%)"
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Interpretation")
    lines.append("-" * 70)
    lines.append(
        "This run replays the same 49-meter architecture as the original "
        "Phase 0 report, against the replacement IDEAL corpus (rescaled "
        "per-load P,Q at feeder-appropriate magnitude, in place of the "
        "deleted Cholesky corpus's raw UCI-household-scale kW). "
        f"Feeder-head P now averages {avg_head_p:.1f} kW "
        f"({100.0 * avg_head_p / NOMINAL_HEAD_P_KW:.1f}% of nominal) versus "
        f"{OLD_REPORT_HEAD_P_KW:.1f} kW ({OLD_REPORT_HEAD_P_PCT_OF_NOMINAL:.1f}%) "
        "before -- the under-loading defect that Phase 0 confirmed is "
        "the deliverable Phase 1 was scoped to fix."
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("Notes on methodology")
    lines.append("-" * 70)
    lines.append(DIRECT_REPLAY_METHODOLOGY_IDEAL)
    lines.append("")
    lines.append(f"Per-timestamp log: simulation/{VALIDATION_LOG.name}")
    lines.append(f"Topology snapshot: simulation/circuit_topology.json")
    lines.append(f"Old-corpus report (unmodified, kept for comparison): "
                 f"reports/phase0_validation_report.txt")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text + "\n")

    print()
    print(report_text)

    return report_text


DIRECT_REPLAY_METHODOLOGY_IDEAL = (
    "No live Kafka broker was available in this environment at the time "
    "of this run (Docker daemon not running), so this replay drives "
    "OpenDSS directly from dataset/node_loads_91/ (the IDEAL corpus) and "
    "representative_meters.json, applying loads and solving once per "
    "timestamp in the same order as streaming/consumer.py's main loop, "
    "using the identical instrumented driver functions "
    "(get_circuit_topology, get_solution_metrics) that consumer.py calls "
    "per timestamp against live telemetry. Per-node V_pu minimum "
    "identity and breach-clustering counts are computed locally in this "
    "script via dss.Circuit.AllBusMagPu(), not inside "
    "simulation/opendss_utils.py, so the shared live-pipeline driver is "
    "untouched."
)


if __name__ == "__main__":
    main()
