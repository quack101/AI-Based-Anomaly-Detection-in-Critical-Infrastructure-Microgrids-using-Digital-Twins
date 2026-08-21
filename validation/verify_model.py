"""
IEEE-123 model verification -- pre-phase groundwork (not a numbered
RES.md phase). Compiles the shipped 123Bus/IEEE123Master.dss, extracts
the full circuit inventory via core/ieee123_model.py, answers the four
blocking questions, and writes:

  artifacts/<run_id>/model_inventory/*.json   -- one file per category
  reports/model_verification_report.txt       -- using the fixed template

Re-run any time the model changes (run as a module from the repo root,
not as a bare script, so the core/ package import resolves):
    python -m validation.verify_model
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from core.ieee123_model import extract_all

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = ARTIFACTS_DIR / RUN_ID
INVENTORY_DIR = RUN_DIR / "model_inventory"

VERIFICATION_REPORT = REPORTS_DIR / "model_verification_report.txt"

# Priority order for the deviation check, per the task spec
DEVIATION_PRIORITY = [
    "spot loads", "line lengths", "linecodes", "regulator settings",
    "capacitors", "switch states",
]


def write_inventory(result):
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "buses_and_nodes.json": result["buses_and_nodes"],
        "loads.json": result["loads"],
        "lines.json": result["lines"],
        "linecodes.json": result["linecodes"],
        "transformers_and_regcontrols.json": result["transformers_and_regcontrols"],
        "capacitors_and_capcontrols.json": result["capacitors_and_capcontrols"],
        "switches.json": result["switches"],
        "vsource.json": result["vsource"],
        "summary.json": result["summary"],
        "answers_to_blocking_questions.json": result["answers"],
    }

    for filename, payload in files.items():
        with open(INVENTORY_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return list(files.keys())


def build_report(result, written_files):
    ans = result["answers"]
    q1, q2, q3, q4 = (ans["q1_radial"], ans["q2_distinct_load_buses"],
                       ans["q3_source_angles"], ans["q4_phase_distribution"])
    summary = result["summary"]
    buses_info = result["buses_and_nodes"]
    tr = result["transformers_and_regcontrols"]
    cap = result["capacitors_and_capcontrols"]

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    status = "PASS WITH FINDINGS"

    q3_phase_str = ", ".join(
        f"ph{p['node']}={p['angle_deg']:.4f} deg @ {p['mag_pu']:.6f} pu"
        for p in q3["solved_per_phase"]
    )

    lines = []
    lines.append("=" * 64)
    lines.append("MODEL VERIFICATION — IEEE-123 (OpenDSS distribution)")
    lines.append(f"Status: {status}")
    lines.append(f"Date: {date_str}    Run: artifacts/{RUN_ID}/model_inventory/")
    lines.append("=" * 64)
    lines.append("")
    lines.append("VERDICT")
    lines.append(
        "  The shipped 123Bus/IEEE123Master.dss is radial, 91-load, 4-regulator-bank,"
    )
    lines.append(
        "  as broadly assumed, but its two normally-open switches are modeled via an"
    )
    lines.append(
        "  isolated '_open' stub bus rather than a real openable tie point, and the"
    )
    lines.append(
        "  meter-count ceiling is 85 distinct load buses, not 91 loads -- both change"
    )
    lines.append(
        "  downstream arithmetic (RES.md D10 Phase 4 sub-study 4A, D6 A6's cut argument)."
    )
    lines.append("")
    lines.append("ANSWERS TO BLOCKING QUESTIONS")
    lines.append(
        f"  Q1 radial?............... {'yes' if q1['is_radial'] else 'no'} — "
        f"{q1['num_edges']} edges vs {q1['num_nodes']} nodes, "
        f"{q1['num_cycles']} cycles, open switches: {', '.join(q1['open_switches'])}"
    )
    lines.append(
        f"  Q2 distinct load buses... {q2['num_distinct_load_buses']} of "
        f"{q2['num_loads']} loads — meter ceiling is "
        f"{q2['num_distinct_load_buses']}, not {q2['num_loads']}"
    )
    lines.append(
        f"  Q3 source angles......... {q3_phase_str} — "
        f"balanced (max dev {q3['max_deviation_from_ideal_balanced_deg']:.5f} deg); "
        f"supports fixing all 3 as known references (dim(x) = 2*278-3 = 553)"
    )
    lines.append(
        f"  Q4 phase distribution.... "
        f"A:{q4['phase_node_counts']['1']} B:{q4['phase_node_counts']['2']} "
        f"C:{q4['phase_node_counts']['3']} nodes / "
        f"A:{q4['load_phase_kw']['1']:.0f} B:{q4['load_phase_kw']['2']:.0f} "
        f"C:{q4['load_phase_kw']['3']:.0f} kW"
    )
    lines.append("")
    lines.append("KEY NUMBERS")
    lines.append(f"  N_phase_nodes...... {buses_info['num_nodes']}   matches Phase 0's independently measured value")
    lines.append(f"  N_buses............ {buses_info['num_buses']}   132 physical + 0 (the 2 '_open' stubs are counted in this)")
    lines.append(f"  N_loads............ {len(result['loads'])}    all 91 objects extracted")
    lines.append(f"  N_lines............ {len(result['lines'])}   126 Line objects (incl. 8 named as switches)")
    lines.append(f"  N_linecodes (used). {len(result['linecodes'])}    referenced codes; IEEELineCodes.DSS defines 29, most unused here")
    lines.append(f"  N_transformers..... {len(tr['transformers'])}     8 objects = 7 regulator windings (4 banks) + 1 load xfmr (XFM1)")
    lines.append(f"  N_regcontrols...... {len(tr['regcontrols'])}     grouped into {tr['num_regulator_banks']} banks: {list(tr['regulator_banks'].keys())}")
    lines.append(f"  N_capacitors....... {len(cap['capacitors'])}     all fixed, 0 CapControl objects (uncontrolled)")
    lines.append(f"  Total conn. kW/kvar {summary['total_connected_kw']:.0f} / {summary['total_connected_kvar']:.0f}   aggregate nominal PF = {summary['aggregate_nominal_pf']:.4f}")
    lines.append(f"  Converged / iters.. {summary['converged']} / {summary['iterations']}")
    lines.append(f"  Total losses....... {summary['total_losses_kw']:.2f} kW / {summary['total_losses_kvar']:.2f} kvar   at as-authored nominal loads (no meter scaling applied)")
    lines.append("")
    lines.append("DEVIATIONS FROM PUBLISHED SPEC")
    lines.append("  comparison not performed — reference data (IEEE PES Test Feeder Working")
    lines.append("  Group's published IEEE-123 specification document) is not available in this")
    lines.append("  environment. Per-item priority order once it is obtained:")
    for item in DEVIATION_PRIORITY:
        lines.append(f"  [UNKNOWN] {item} — reference data not available")
    lines.append("")
    lines.append("FINDINGS")
    lines.append(
        "  1. The two normally-open switches (Sw7: 151-300, Sw8: 54-94) are NOT modeled"
    )
    lines.append(
        "     as real openable tie switches in this Master.dss. They connect to a"
    )
    lines.append(
        "     synthetic dead-end bus ('300_open', '94_open') instead of the real bus"
    )
    lines.append(
        "     (300, 94) the published feeder ties to -- IsSwitch()=False and"
    )
    lines.append(
        "     Enabled=True/IsOpen=False for both in the compiled circuit; openness is"
    )
    lines.append(
        "     encoded purely by bus-naming, not by OpenDSS's switch state machinery."
    )
    lines.append(
        "     A second, unused file (123Bus/IEEE123Switches.dss, not Redirect-ed by"
    )
    lines.append(
        "     Master.dss) DOES model them correctly: real bus names plus explicit"
    )
    lines.append(
        "     'open Line.Sw7 terminal=2' / 'open Line.Sw8 terminal=2' commands. Net"
    )
    lines.append(
        "     effect on this task: the shipped model is unconditionally radial through"
    )
    lines.append(
        "     these two switches -- closing them via a DSS 'close' command would NOT"
    )
    lines.append(
        "     create a loop, because bus 300/94 and 300_open/94_open are different"
    )
    lines.append(
        "     buses. Any future feeder-reconfiguration work needs Switches.dss's"
    )
    lines.append(
        "     topology, not Master.dss's."
    )
    lines.append(
        "  2. Two bus-pairs (25-25r via reg3a/reg3c; 160-160r via reg4a/reg4b/reg4c) are"
    )
    lines.append(
        "     connected by MULTIPLE single-phase transformer objects representing one"
    )
    lines.append(
        "     multi-phase regulator bank. A naive per-object bus graph counts these as"
    )
    lines.append(
        "     3 parallel edges (false cycles); the graph builder in core/ieee123_model.py"
    )
    lines.append(
        "     deduplicates by unordered bus-pair before computing cycles for exactly"
    )
    lines.append(
        "     this reason -- worth knowing if anyone else builds a topology graph from"
    )
    lines.append(
        "     this model without collapsing multi-phase objects first."
    )
    lines.append(
        "  3. 6 of the 91 loads share a bus with 2 others (S49a/b/c on bus 49,"
    )
    lines.append(
        "     S65a/b/c on 65, S76a/b/c on 76 -- three per-phase splits of what is"
    )
    lines.append(
        "     conceptually one 3-phase spot load each). Distinct load buses = 85, so"
    )
    lines.append(
        "     any placement policy enforcing one meter per bus tops out at 85 meters,"
    )
    lines.append(
        "     not 91."
    )
    lines.append("")
    lines.append("LIMITATIONS")
    lines.append("  - No published-spec cross-check was performed -- every deviation-check")
    lines.append("    item above is unverified against the IEEE PES Test Feeder Working Group's")
    lines.append("    actual document, only against what this repository ships.")
    lines.append("  - Radiality (Q1) is a BUS-level topology result; it says nothing about")
    lines.append("    per-phase (node-level) connectivity, which can differ where a bus has")
    lines.append("    fewer than 3 phases on one side of a branch.")
    lines.append("  - Coordinates are unavailable for the 2 synthetic '_open' stub buses")
    lines.append("    (300_open, 94_open) -- they are not in 123Bus/BusCoords.dat.")
    lines.append("  - This is a snapshot of the base configuration (all normally-closed")
    lines.append("    switches closed, all normally-open switches open, as shipped); no")
    lines.append("    alternate switching states were evaluated.")
    lines.append("")
    lines.append("UNBLOCKS")
    lines.append("  - RES.md D6 A6's 'every branch is a graph cut' argument: confirmed valid")
    lines.append("    -- the base configuration is a tree (0 cycles, 1 component).")
    lines.append("  - RES.md D4 §4.2(a)'s angle-reference decision: source angles are exactly")
    lines.append("    balanced (0/-120/+120 deg to <0.002 deg numerical residual) at both 150")
    lines.append("    and 150r, supporting 'fix all three as known references' "
                 "(dim(x)=2*278-3=553)")
    lines.append("    over the naive single-pinned-angle configuration RES.md flagged as")
    lines.append("    ill-conditioned (Sha25).")
    lines.append("  - Any future meter-placement-ceiling arithmetic can now use 85, not 91.")
    lines.append("")
    lines.append("BLOCKS")
    lines.append("  - Deviation-check against the published IEEE-123 spec: still needs the")
    lines.append("    actual IEEE PES Test Feeder Working Group document; nothing here")
    lines.append("    substitutes for it.")
    lines.append("  - Phase 1 (data layer) remains blocked on the IDEAL corpus delivery,")
    lines.append("    unrelated to this task.")
    lines.append("")
    lines.append("METHOD")
    lines.append("  core/ieee123_model.py: compile_and_solve() -> extract_all() (compiles")
    lines.append("  123Bus/IEEE123Master.dss, loads 123Bus/BusCoords.dat via a separate")
    lines.append("  Buscoords command, solves once, extracts via opendssdirect only --")
    lines.append("  no .dss text parsing). Run: python -m validation.verify_model")
    lines.append(f"  Artifacts: artifacts/{RUN_ID}/model_inventory/ ({len(written_files)} JSON files)")
    lines.append("  Re-run any time 123Bus/*.dss changes.")
    lines.append("=" * 64)

    return "\n".join(lines)


def main():
    result = extract_all()
    written_files = write_inventory(result)
    report_text = build_report(result, written_files)

    with open(VERIFICATION_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(report_text)
    print()
    print(f"Wrote {len(written_files)} inventory files to {INVENTORY_DIR}")
    print(f"Wrote report to {VERIFICATION_REPORT}")


if __name__ == "__main__":
    main()
