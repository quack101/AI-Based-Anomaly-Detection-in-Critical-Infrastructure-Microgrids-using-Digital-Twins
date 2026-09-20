"""
Builds config/sensor_registry.json (declarative, not code) from:
  - simulation/representative_meters.json    -- the 49 metered nodes
  - core.ieee123_model.extract_all()         -- verified compiled-circuit
                                                 topology and lines
  - placement.feeder_context.FeederContext   -- the SAME backbone
                                                 (main_feeder) computation
                                                 the placement work uses,
                                                 per RES.md D4 4.3 row 3's
                                                 explicit instruction to
                                                 reuse the backbone-
                                                 coverage criterion

Registry contents, per RES.md D10 Phase 2's explicit scope for this
phase (do not add sensors this phase does not justify):
  - 49 x P_injection, 49 x Q_injection   -- ENABLED (current config, m=98)
  - 49 x V_magnitude at the same meters  -- disabled (D4 4.3 row 4)
  - substation head P,Q,|V|,|I| x 3 phases = 12 entries -- disabled (row 2)
  - branch P_flow,Q_flow,|I| at each backbone (main-feeder) Line segment,
    per phase actually present -- disabled (row 3)
  Row 0 (zero-injection) and row 1 (pseudo-measurements at unmetered
  Loads) are NOT sensor-registry entries -- they are not things a sensor
  reads off a solved circuit (row 0 is an exact algebraic fact about the
  network; row 1 is an estimator input derived from historical data, not
  a live measurement). Both belong to Phase 4/5's estimator input, not
  Phase 2's sensor layer. Transformer sensors (row 6) are explicitly not
  justified by any cited paper -- none are added. PMUs (row 5) are
  explicitly optional and deferred -- none are added.

sigma (noise model): RES.md does not specify sensor accuracy numbers
for these live sensors (only sigma_pseudo, for the DIFFERENT row-1
pseudo-measurements, explicitly deferred to Phase 5). In their absence,
this generator uses documented, conventional instrumentation-accuracy
assumptions, computed per sensor from its OWN solved-circuit reading
(not a single global constant, per the Phase 2 acceptance criterion):
  - P/Q injection, branch flow, head P/Q: 1% of the sensor's own
    noise-free reading magnitude at generation time (AMI/SCADA-grade
    accuracy order of magnitude; ANSI C12.20 revenue meters are
    tighter, ~0.2-0.5%, general SCADA analog channels are often looser,
    1-3% -- 1% is a defensible mid-point, not a literature-cited figure).
  - Current magnitude (branch, head): 1%, same reasoning.
  - Voltage magnitude: a FLAT 0.5% of NOMINAL (1.0 p.u.), not of the
    reading -- voltage-instrument accuracy is conventionally expressed
    against nominal, not against the (near-1.0) reading itself.
These are flagged here, in this docstring, and in the generated file's
own top-level "_sigma_assumptions" note, as a documented assumption
pending more specific guidance -- not asserted as literature fact.

Run: python -m sensors.generate_registry
"""

import json
from pathlib import Path

import opendssdirect as dss

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from sensors.measurement_functions import measure

BASE_DIR = Path(__file__).resolve().parent.parent
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
OUTPUT_PATH = BASE_DIR / "config" / "sensor_registry.json"

RATE_SECONDS = 60  # 1 sim-minute, matching the existing producer cadence

SIGMA_FRAC_POWER = 0.01     # P, Q, |I| -- 1% of own reading
SIGMA_FRAC_VOLTAGE = 0.005  # |V| -- 0.5% of NOMINAL (1.0 p.u.), flat


def _sigma_from_reading(value, frac):
    return max(abs(value) * frac, 1e-9)  # floor avoids sigma==0 at a near-zero reading


def build_meter_entries(representative_meters, multiphase=False):
    """One P_injection + one Q_injection per meter -- m = 98 for the 49
    meters, exactly as RES.md D10 Phase 2 states for 'the current
    configuration'. Multi-phase loads are metered on their primary phase
    only (node_order[0]) by default, matching the existing pipeline's
    one-P/Q-per-meter convention.

    Phase-2 Finding #1 correction: of the 5 loads originally flagged as
    "multi-phase" (s35a, s47, s48, s76a, s65c), direct inspection of
    123Bus/IEEE123Loads.DSS shows only s47 and s48 are genuinely 3-phase
    (Phases=3 Conn=Wye -- three independent P/Q injections). s35a, s65c
    and s76a are Phases=1 Conn=Delta: single-phase loads connected
    line-to-line, so node_order lists two bus/phase references (the
    delta's two terminals) but CktElement.Powers() only ever returns ONE
    (P,Q) pair -- there is no second phase's worth of power to meter on
    these three; the registry's existing primary-phase entry already
    captures 100% of what they draw. Confirmed via NumPhases() == 1 for
    all three (see reports/phase3_streaming_report.txt). Only s47/s48
    are genuinely under-covered by the single-phase-primary registry.

    multiphase=True adds P_injection + Q_injection (+ a disabled
    V_magnitude, matching the per-meter triplet convention) for s47 and
    s48's remaining phases B and C -- the second registry variant RES.md
    D10 Phase 3 asks for, selectable via sensors.registry.load_registry
    (variant='multiphase'). The single-phase-primary entries below are
    IDENTICAL in both variants (same ids, same values) -- multiphase
    only appends."""

    entries = []

    for meter in representative_meters:
        load_name = meter["load_name"]
        bus = str(meter["bus"])
        phase = _resolve_meter_phase(load_name)

        base_id = f"{meter['meter_id']}"

        p_entry = {
            "id": f"{base_id}_P",
            "node": bus,
            "phase": phase,
            "quantity": "P_injection",
            "load_name": load_name,
            "sigma": None,  # filled in after a solve, below
            "rate": RATE_SECONDS,
            "enabled": True,
            "note": "AMI meter P injection -- current configuration",
        }
        q_entry = {
            "id": f"{base_id}_Q",
            "node": bus,
            "phase": phase,
            "quantity": "Q_injection",
            "load_name": load_name,
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": True,
            "note": "AMI meter Q injection -- current configuration",
        }
        v_entry = {
            "id": f"{base_id}_V",
            "node": bus,
            "phase": phase,
            "quantity": "V_magnitude",
            "load_name": load_name,
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 4 -- free at existing meters, disabled by default",
        }

        entries.extend([p_entry, q_entry, v_entry])

        if multiphase:
            for extra_phase in _resolve_secondary_phases(load_name, phase):
                entries.extend(_extra_phase_entries(
                    base_id, bus, extra_phase, load_name,
                ))

    return entries


def _extra_phase_entries(base_id, bus, phase, load_name):
    return [
        {
            "id": f"{base_id}_P_{phase}",
            "node": bus,
            "phase": phase,
            "quantity": "P_injection",
            "load_name": load_name,
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": True,
            "note": "AMI meter P injection -- multiphase variant, additional phase",
        },
        {
            "id": f"{base_id}_Q_{phase}",
            "node": bus,
            "phase": phase,
            "quantity": "Q_injection",
            "load_name": load_name,
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": True,
            "note": "AMI meter Q injection -- multiphase variant, additional phase",
        },
        {
            "id": f"{base_id}_V_{phase}",
            "node": bus,
            "phase": phase,
            "quantity": "V_magnitude",
            "load_name": load_name,
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 4 -- multiphase variant, additional phase, disabled by default",
        },
    ]


def _resolve_meter_phase(load_name):
    """The load's primary phase -- node_order[0]. For single-phase loads
    (including the delta-connected s35a/s65c/s76a, see build_meter_entries)
    this is their only phase; for the two genuinely 3-phase loads (s47,
    s48) this is a documented under-coverage in the DEFAULT variant only."""

    dss.Circuit.SetActiveElement(f"Load.{load_name}")
    node_order = dss.CktElement.NodeOrder()
    phase_num = node_order[0]
    return {1: "A", 2: "B", 3: "C"}[phase_num]


def _resolve_secondary_phases(load_name, primary_phase):
    """Phases beyond the primary that this load ACTUALLY draws
    independent power on -- gated on NumPhases(), not on node_order
    length, precisely to avoid the delta-connection miscount corrected
    above (node_order for a 1-phase delta load lists 2 entries; only
    NumPhases() distinguishes a real 3-phase Wye load from a 1-phase
    load spanning two conductors)."""

    dss.Circuit.SetActiveElement(f"Load.{load_name}")
    num_phases = dss.CktElement.NumPhases()
    if num_phases <= 1:
        return []

    node_order = dss.CktElement.NodeOrder()[:num_phases]
    all_phases = sorted({1: "A", 2: "B", 3: "C"}[n] for n in node_order)
    return [p for p in all_phases if p != primary_phase]


def build_head_entries():
    entries = []

    for phase in ("A", "B", "C"):
        entries.append({
            "id": f"HEAD_P_{phase}",
            "node": "150",
            "phase": phase,
            "quantity": "P_head",
            "element_name": "source",
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 2 -- substation head",
        })
        entries.append({
            "id": f"HEAD_Q_{phase}",
            "node": "150",
            "phase": phase,
            "quantity": "Q_head",
            "element_name": "source",
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 2 -- substation head",
        })
        entries.append({
            "id": f"HEAD_V_{phase}",
            "node": "150",
            "phase": phase,
            "quantity": "V_magnitude",
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 2 -- substation head",
        })
        entries.append({
            "id": f"HEAD_I_{phase}",
            "node": "150",
            "phase": phase,
            "quantity": "I_magnitude_head",
            "element_name": "source",
            "sigma": None,
            "rate": RATE_SECONDS,
            "enabled": False,
            "note": "D4 4.3 row 2 -- substation head",
        })

    return entries


def _find_line_for_segment(bus_a, bus_b, lines_by_pair):
    return lines_by_pair.get(frozenset((bus_a, bus_b)))


def build_branch_entries(main_feeder, lines):
    """One P_flow/Q_flow/I_magnitude_branch entry per phase actually
    present, for each backbone (main-feeder) segment that is a Line
    (regulator/transformer hops on the backbone are skipped -- 'branch
    flow sensor' means a conductor, per RES.md D4 4.3 row 3's framing;
    the regulator's own head-of-feeder role is separately covered by
    the substation-head entries)."""

    lines_by_pair = {
        frozenset((line["bus1_root"], line["bus2_root"])): line
        for line in lines
    }

    entries = []

    for bus_a, bus_b in zip(main_feeder[:-1], main_feeder[1:]):
        line = _find_line_for_segment(bus_a, bus_b, lines_by_pair)
        if line is None:
            continue  # not a Line (e.g. a regulator transformer hop)

        dss.Circuit.SetActiveElement(f"Line.{line['name']}")
        num_phases = dss.CktElement.NumPhases()
        node_order = dss.CktElement.NodeOrder()[:num_phases]
        phases_present = sorted({1: "A", 2: "B", 3: "C"}[n] for n in node_order)

        for phase in phases_present:
            common = {
                "node": line["bus1_root"],
                "phase": phase,
                "element_name": line["name"],
                "element_kind": "Line",
                "rate": RATE_SECONDS,
                "enabled": False,
                "note": f"D4 4.3 row 3 -- backbone segment {bus_a}-{bus_b}",
            }
            entries.append({**common, "id": f"BR_{line['name']}_P_{phase}",
                             "quantity": "P_flow", "sigma": None})
            entries.append({**common, "id": f"BR_{line['name']}_Q_{phase}",
                             "quantity": "Q_flow", "sigma": None})
            entries.append({**common, "id": f"BR_{line['name']}_I_{phase}",
                             "quantity": "I_magnitude_branch", "sigma": None})

    return entries


def fill_sigmas(entries):
    for entry in entries:
        reading = measure(entry)
        frac = SIGMA_FRAC_VOLTAGE if entry["quantity"] == "V_magnitude" else SIGMA_FRAC_POWER
        reference = 1.0 if entry["quantity"] == "V_magnitude" else reading
        entry["sigma"] = _sigma_from_reading(reference, frac)
    return entries


def main(multiphase=False, output_path=None, write=True):
    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)

    extraction = extract_all()  # compiles + solves; leaves circuit active
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)

    meter_entries = build_meter_entries(representative_meters, multiphase=multiphase)
    head_entries = build_head_entries()
    branch_entries = build_branch_entries(context.main_feeder, extraction["lines"])

    entries = meter_entries + head_entries + branch_entries
    entries = fill_sigmas(entries)

    output_path = output_path or OUTPUT_PATH
    if write:
        with open(output_path, "w") as f:
            json.dump(entries, f, indent=2)

    num_segments = len({e["note"] for e in branch_entries})
    total_backbone_hops = len(context.main_feeder) - 1

    enabled = sum(1 for e in entries if e["enabled"])
    label = "multiphase" if multiphase else "default"
    print(f"[{label}] Wrote {len(entries)} registry entries ({enabled} enabled, m={enabled}) to {output_path}")
    print(f"  meters: {len(meter_entries)} ({sum(1 for e in meter_entries if e['enabled'])} enabled)")
    print(f"  head:   {len(head_entries)} ({sum(1 for e in head_entries if e['enabled'])} enabled)")
    print(f"  branch: {len(branch_entries)} ({sum(1 for e in branch_entries if e['enabled'])} enabled, "
          f"{num_segments} of {total_backbone_hops} backbone hops matched a Line "
          f"(the rest are regulator-collapsed hops with no single Line), "
          f"up to 3 phases x 3 quantities each)")

    return entries


if __name__ == "__main__":
    # Default variant (m=98) is left untouched by this run -- it is
    # already generated, validated (tests/test_sensors.py), and consumed
    # by Phase 3's producer; regenerating it here risks perturbing sigma
    # values with no benefit. Only the new multiphase variant is written.
    MULTIPHASE_OUTPUT = BASE_DIR / "config" / "sensor_registry_multiphase.json"
    main(multiphase=True, output_path=MULTIPHASE_OUTPUT)
