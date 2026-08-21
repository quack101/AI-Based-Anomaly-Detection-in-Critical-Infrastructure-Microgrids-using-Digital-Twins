"""
RES.md D10 Phase 0 -- summarize the live producer -> Kafka -> consumer
replay into the same acceptance-criteria report format as
validate_phase0.py's direct-replay report.

Run after streaming/producer.py and streaming/consumer.py have completed
a full-day (1440-timestamp) live run against a running Kafka broker.
Reads simulation/phase0_validation_log.csv (written live by
streaming/consumer.py's log_validation_row) and simulation/circuit_topology.json
(written live by streaming/consumer.py at startup) -- it does not re-run
OpenDSS itself.
"""

import csv
import json
from pathlib import Path

import opendssdirect as dss

from opendss_utils import compile_feeder
from validate_phase0 import write_summary_report, REPORTS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

VALIDATION_LOG = BASE_DIR / "simulation" / "phase0_validation_log.csv"
TOPOLOGY_SNAPSHOT = BASE_DIR / "simulation" / "circuit_topology.json"
LIVE_REPORT = REPORTS_DIR / "phase0_validation_report_live.txt"

LIVE_METHODOLOGY = (
    "This report is computed from simulation/phase0_validation_log.csv as "
    "written LIVE by streaming/consumer.py while consuming real messages "
    "from Kafka topic smartgrid.telemetry, published by streaming/producer.py "
    "against a running Docker Kafka broker (streaming/docker-compose.yml). "
    "It replaces the direct-replay workaround used earlier when no broker "
    "was available in this environment, and validates the full live "
    "pipeline end-to-end: producer -> Kafka -> consumer buffering -> "
    "per-timestamp OpenDSS solve -> instrumentation. producer.py was "
    "bounded to NUM_TIMESTAMPS = 1440 (one simulated day at 1 sec = 1 "
    "simulated minute) for this run, since the underlying synthetic "
    "datasets are far longer than a day."
)


def load_rows():
    rows = []
    with open(VALIDATION_LOG, "r", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append({
                "timestamp": int(raw["timestamp"]),
                "converged": raw["converged"] == "True",
                "iterations": int(raw["iterations"]),
                "v_pu_min": float(raw["v_pu_min"]),
                "v_pu_max": float(raw["v_pu_max"]),
                "losses_kw": float(raw["losses_kw"]),
                "losses_kvar": float(raw["losses_kvar"]),
                "feeder_head_p_kw": float(raw["feeder_head_p_kw"]),
                "feeder_head_q_kvar": float(raw["feeder_head_q_kvar"]),
                "solve_time_s": float(raw["solve_time_s"]),
            })
    return rows


def main():
    rows = load_rows()

    with open(TOPOLOGY_SNAPSHOT, "r") as f:
        topology = json.load(f)
    n_phase_nodes = topology["n_phase_nodes"]
    all_node_names = topology["all_node_names"]

    # Nominal total load is a property of the as-authored DSS model, not
    # of the run -- recompute it the same way validate_phase0.py does, on
    # a fresh compile, so it isn't polluted by whatever load values the
    # live consumer process left behind in memory.
    compile_feeder()
    nominal_total_kw = 0.0
    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)
        nominal_total_kw += dss.Loads.kW()

    write_summary_report(
        rows, n_phase_nodes, all_node_names, nominal_total_kw,
        output_path=LIVE_REPORT,
        methodology=LIVE_METHODOLOGY,
    )


if __name__ == "__main__":
    main()
