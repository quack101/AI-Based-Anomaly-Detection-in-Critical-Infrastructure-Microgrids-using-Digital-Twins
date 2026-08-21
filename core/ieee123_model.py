"""
IEEE-123 model extraction and verification -- reusable library code.

Extracts the full circuit inventory (buses, nodes, loads, lines,
linecodes, transformers/regcontrols, capacitors/capcontrols, switches,
Vsource, summary) from the COMPILED OpenDSS circuit -- never by reading
or parsing the .dss text directly -- and answers the four blocking
questions in the model-verification task (radiality, distinct load
buses, source phase angles, phase distribution).

Re-run via validation/verify_model.py whenever the model changes.
"""

import math
from pathlib import Path

import opendssdirect as dss

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DSS = BASE_DIR / "123Bus" / "IEEE123Master.dss"
BUS_COORDS = BASE_DIR / "123Bus" / "BusCoords.dat"

# Standard OpenDSS line-length unit enum -> label
_LENGTH_UNITS = {
    0: "none", 1: "mi", 2: "kft", 3: "km", 4: "m", 5: "ft", 6: "in", 7: "cm",
}


def compile_and_solve(master_path=MASTER_DSS, bus_coords_path=BUS_COORDS,
                       load_bus_coords=True):
    """Compile the circuit and solve once. Coordinates are not loaded by
    IEEE123Master.dss's own compile chain (no Buscoords command in it),
    so they are loaded here as a separate, explicit, read-only metadata
    step -- it does not alter the electrical model."""

    dss.Text.Command(f'Compile "{master_path}"')

    if load_bus_coords and bus_coords_path.exists():
        dss.Text.Command(f'Buscoords "{bus_coords_path}"')

    dss.Solution.Solve()

    return {
        "converged": dss.Solution.Converged(),
        "iterations": dss.Solution.Iterations(),
    }


def _bus_root(bus_spec):
    """'150r.1.2.3' -> '150r'"""
    return bus_spec.split(".")[0]


def extract_buses_and_nodes():
    all_bus_names = dss.Circuit.AllBusNames()
    all_node_names = dss.Circuit.AllNodeNames()
    num_nodes = dss.Circuit.NumNodes()

    buses = []
    for bus in all_bus_names:
        dss.Circuit.SetActiveBus(bus)
        buses.append({
            "name": bus,
            "nodes": dss.Bus.Nodes(),
            "num_phases": len(dss.Bus.Nodes()),
            "base_kv": dss.Bus.kVBase(),
            "x": dss.Bus.X(),
            "y": dss.Bus.Y(),
            "distance_from_source_km": dss.Bus.Distance(),
        })

    return {
        "num_buses": dss.Circuit.NumBuses(),
        "num_nodes": num_nodes,
        "all_bus_names": all_bus_names,
        "all_node_names": all_node_names,
        "voltage_bases_kv": dss.Settings.VoltageBases(),
        "buses": buses,
    }


def extract_loads():
    loads = []
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        dss.Circuit.SetActiveElement(f"Load.{name}")

        kw = dss.Loads.kW()
        kvar = dss.Loads.kvar()
        kva = math.hypot(kw, kvar)
        pf = (kw / kva) if kva > 0 else None

        loads.append({
            "name": name,
            "bus": dss.CktElement.BusNames()[0],
            "bus_root": _bus_root(dss.CktElement.BusNames()[0]),
            "phases": dss.CktElement.NumPhases(),
            "connection": "delta" if dss.Loads.IsDelta() else "wye",
            "kw": kw,
            "kvar": kvar,
            "model": dss.Loads.Model(),
            "kv": dss.Loads.kV(),
            "nominal_pf": pf,
        })

    return loads


def extract_lines():
    lines = []
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        dss.Circuit.SetActiveElement(f"Line.{name}")

        n = dss.Lines.Phases()
        rmat = dss.Lines.RMatrix()
        xmat = dss.Lines.XMatrix()

        lines.append({
            "name": name,
            "bus1": dss.Lines.Bus1(),
            "bus2": dss.Lines.Bus2(),
            "bus1_root": _bus_root(dss.Lines.Bus1()),
            "bus2_root": _bus_root(dss.Lines.Bus2()),
            "phases": n,
            "length": dss.Lines.Length(),
            "units_code": dss.Lines.Units(),
            "units": _LENGTH_UNITS.get(dss.Lines.Units(), "unknown"),
            "linecode": dss.Lines.LineCode(),
            "rmatrix": [rmat[i * n:(i + 1) * n] for i in range(n)] if rmat else [],
            "xmatrix": [xmat[i * n:(i + 1) * n] for i in range(n)] if xmat else [],
            "is_switch_named": name.lower().startswith("sw"),
            "enabled": dss.CktElement.Enabled(),
        })

    return lines


def extract_linecodes(referenced_codes):
    """Only linecodes actually referenced by a Line in this compiled
    circuit -- IEEELineCodes.DSS defines more codes than this feeder
    uses (leftover definitions shared with other Kersting test feeders)."""

    referenced = {c.lower() for c in referenced_codes if c}
    codes = []

    for name in dss.LineCodes.AllNames():
        if name.lower() not in referenced:
            continue

        dss.LineCodes.Name(name)
        n = dss.LineCodes.Phases()
        rmat = dss.LineCodes.Rmatrix()
        xmat = dss.LineCodes.Xmatrix()
        cmat = dss.LineCodes.Cmatrix()

        codes.append({
            "name": name,
            "nphases": n,
            "units_code": dss.LineCodes.Units(),
            "units": _LENGTH_UNITS.get(dss.LineCodes.Units(), "unknown"),
            "rmatrix": [rmat[i * n:(i + 1) * n] for i in range(n)] if rmat else [],
            "xmatrix": [xmat[i * n:(i + 1) * n] for i in range(n)] if xmat else [],
            "cmatrix": [cmat[i * n:(i + 1) * n] for i in range(n)] if cmat else [],
        })

    return codes


def extract_transformers_and_regcontrols():
    transformers = []
    for name in dss.Transformers.AllNames():
        dss.Transformers.Name(name)
        dss.Circuit.SetActiveElement(f"Transformer.{name}")

        num_windings = dss.Transformers.NumWindings()
        windings = []
        for w in range(1, num_windings + 1):
            dss.Transformers.Wdg(w)
            windings.append({
                "winding": w,
                "bus": dss.CktElement.BusNames()[w - 1] if w - 1 < len(dss.CktElement.BusNames()) else None,
                "kv": dss.Transformers.kV(),
                "kva": dss.Transformers.kVA(),
                "tap": dss.Transformers.Tap(),
                "min_tap": dss.Transformers.MinTap(),
                "max_tap": dss.Transformers.MaxTap(),
                "num_taps": dss.Transformers.NumTaps(),
            })

        transformers.append({
            "name": name,
            "num_windings": num_windings,
            "phases": dss.CktElement.NumPhases(),
            "xhl": dss.Transformers.Xhl(),
            "is_regulator": name.lower().startswith("reg"),
            "windings": windings,
        })

    regcontrols = []
    for name in dss.RegControls.AllNames():
        dss.RegControls.Name(name)
        regcontrols.append({
            "name": name,
            "transformer": dss.RegControls.Transformer(),
            "winding": dss.RegControls.Winding(),
            "vreg": dss.RegControls.ForwardVreg(),
            "band": dss.RegControls.ForwardBand(),
            "ptratio": dss.RegControls.PTRatio(),
            "ct_primary": dss.RegControls.CTPrimary(),
            "r_compensation": dss.RegControls.ForwardR(),
            "x_compensation": dss.RegControls.ForwardX(),
        })

    # Group by regulator "bank" (transformer-name prefix minus trailing
    # phase letter, e.g. reg4a/reg4b/reg4c -> bank reg4) so the 7
    # transformer+regcontrol objects map onto the feeder's 4 regulators.
    banks = {}
    for rc in regcontrols:
        xfmr = next((t for t in transformers if t["name"] == rc["transformer"].lower()), None)
        bank = rc["transformer"].lower().rstrip("abc")
        banks.setdefault(bank, []).append(rc["name"])

    return {
        "transformers": transformers,
        "regcontrols": regcontrols,
        "regulator_banks": banks,
        "num_regulator_banks": len(banks),
    }


def extract_capacitors_and_capcontrols():
    capacitors = []
    for name in dss.Capacitors.AllNames():
        dss.Capacitors.Name(name)
        dss.Circuit.SetActiveElement(f"Capacitor.{name}")
        capacitors.append({
            "name": name,
            "bus": dss.CktElement.BusNames()[0],
            "bus_root": _bus_root(dss.CktElement.BusNames()[0]),
            "phases": dss.CktElement.NumPhases(),
            "kvar": dss.Capacitors.kvar(),
            "kv": dss.Capacitors.kV(),
        })

    capcontrols = []
    for name in dss.CapControls.AllNames():
        dss.CapControls.Name(name)
        capcontrols.append({
            "name": name,
            "capacitor": dss.CapControls.Capacitor(),
            "mode": dss.CapControls.Mode(),
            "on_setting": dss.CapControls.ONSetting(),
            "off_setting": dss.CapControls.OFFSetting(),
        })

    return {
        "capacitors": capacitors,
        "capcontrols": capcontrols,
        "note": (
            "No CapControl objects exist in this compiled circuit -- all "
            "4 capacitors are fixed (always-on), uncontrolled shunts."
            if not capcontrols else None
        ),
    }


def extract_switches(lines):
    """Switches, identified by the 'Sw<N>' naming convention this model
    uses (IsSwitch() is False for all of them -- see FINDINGS in the
    verification report: this model represents open switches by wiring
    to a topologically isolated '<bus>_open' stub bus rather than via
    OpenDSS's Enabled/IsOpen switch machinery)."""

    switches = []
    for line in lines:
        if not line["is_switch_named"]:
            continue

        is_open = line["bus2_root"].lower().endswith("_open") or \
                  line["bus1_root"].lower().endswith("_open")

        switches.append({
            "name": line["name"],
            "bus1": line["bus1"],
            "bus2": line["bus2"],
            "state": "open" if is_open else "closed",
        })

    return switches


def extract_vsource():
    dss.Vsources.Name("source")
    dss.Circuit.SetActiveElement("Vsource.source")

    prop_names = dss.CktElement.AllPropertyNames()
    props = {n: dss.Properties.Value(n) for n in prop_names}

    source_bus = dss.CktElement.BusNames()[0]
    bus_root = _bus_root(source_bus)

    dss.Circuit.SetActiveBus(bus_root)
    solved = {
        "bus": bus_root,
        "nodes": dss.Bus.Nodes(),
        "pu_vmag_angle": dss.Bus.puVmagAngle(),
    }

    return {
        "name": "source",
        "bus": source_bus,
        "base_kv": dss.Vsources.BasekV(),
        "configured_angle_deg": dss.Vsources.AngleDeg(),
        "pu": dss.Vsources.PU(),
        "phases": dss.Vsources.Phases(),
        "properties": props,
        "solved_at_source_bus": solved,
    }


def extract_summary(loads):
    total_kw = sum(l["kw"] for l in loads)
    total_kvar = sum(l["kvar"] for l in loads)
    total_kva = math.hypot(total_kw, total_kvar)
    aggregate_pf = (total_kw / total_kva) if total_kva > 0 else None

    losses_w, losses_var = dss.Circuit.Losses()

    return {
        "total_connected_kw": total_kw,
        "total_connected_kvar": total_kvar,
        "aggregate_nominal_pf": aggregate_pf,
        "converged": dss.Solution.Converged(),
        "iterations": dss.Solution.Iterations(),
        "total_losses_kw": losses_w / 1000.0,
        "total_losses_kvar": losses_var / 1000.0,
    }


# --------------------------------------------------------------------
# Graph analysis (Q1) and the other blocking questions
# --------------------------------------------------------------------

def build_bus_graph(lines, transformers):
    """Bus-level (not phase-node-level) graph: one edge per Line and per
    2-winding Transformer, using bus roots. Radiality is a topological
    (bus-connectivity) property; per-phase impedance coupling does not
    add graph edges.

    Multiple single-phase objects connecting the SAME bus pair (e.g.
    regulator banks reg3a/reg3c on 25-25r, reg4a/reg4b/reg4c on
    160-160r -- one object per phase of one multi-phase regulator) are
    one electrical connection between those two buses, not parallel
    paths -- they are collapsed into a single graph edge. Leaving them
    as separate parallel edges was checked and produced 3 false
    "cycles" that traced entirely to these two regulator banks, not to
    any real mesh in the feeder (see model_verification_report.txt)."""

    raw_edges = []
    for line in lines:
        raw_edges.append((line["bus1_root"], line["bus2_root"], line["name"], "line"))

    for xfmr in transformers:
        buses = [w["bus"] for w in xfmr["windings"] if w["bus"]]
        roots = [_bus_root(b) for b in buses]
        if len(roots) >= 2:
            raw_edges.append((roots[0], roots[1], xfmr["name"], "transformer"))

    dedup = {}
    for u, v, name, etype in raw_edges:
        key = tuple(sorted((u, v)))
        dedup.setdefault(key, []).append((name, etype))

    edges = []
    for (u, v), members in dedup.items():
        edges.append((u, v, [m[0] for m in members], members[0][1]))

    nodes = set()
    adjacency = {}
    for u, v, _, _ in edges:
        nodes.add(u)
        nodes.add(v)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    # Connected components via BFS
    visited = set()
    components = []
    for start in nodes:
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            visited.add(node)
            queue.extend(adjacency.get(node, []))
        components.append(component)

    num_nodes = len(nodes)
    num_edges = len(edges)
    num_components = len(components)

    # Cyclomatic number (first Betti number): independent cycles in an
    # undirected graph = E - N + C. Zero <=> every component is a tree.
    num_cycles = num_edges - num_nodes + num_components

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_components": num_components,
        "num_cycles": num_cycles,
        "is_radial": num_cycles == 0,
        "component_sizes": sorted((len(c) for c in components), reverse=True),
        "parallel_object_bus_pairs": {
            f"{u}-{v}": names for (u, v), names in
            [((e[0], e[1]), e[2]) for e in edges if len(e[2]) > 1]
        },
        "edges": [{"from": u, "to": v, "elements": names, "type": etype}
                   for u, v, names, etype in edges],
    }


def answer_q1_radial(lines, transformers, switches):
    graph = build_bus_graph(lines, transformers)
    open_switches = [s for s in switches if s["state"] == "open"]
    closed_switches = [s for s in switches if s["state"] == "closed"]

    return {
        "is_radial": graph["is_radial"],
        "num_nodes": graph["num_nodes"],
        "num_edges": graph["num_edges"],
        "num_components": graph["num_components"],
        "num_cycles": graph["num_cycles"],
        "open_switches": [s["name"] for s in open_switches],
        "open_switch_details": open_switches,
        "closed_switches": [s["name"] for s in closed_switches],
        "graph": graph,
    }


def answer_q2_distinct_load_buses(loads):
    bus_load_count = {}
    for l in loads:
        bus_load_count.setdefault(l["bus_root"], []).append(l["name"])

    multi_load_buses = {b: names for b, names in bus_load_count.items() if len(names) > 1}

    return {
        "num_loads": len(loads),
        "num_distinct_load_buses": len(bus_load_count),
        "multi_load_buses": multi_load_buses,
        "meter_ceiling_is_91": len(bus_load_count) == len(loads),
    }


def answer_q3_source_angles(vsource_info):
    pu = vsource_info["solved_at_source_bus"]["pu_vmag_angle"]
    nodes = vsource_info["solved_at_source_bus"]["nodes"]

    per_phase = []
    for i, n in enumerate(nodes):
        per_phase.append({
            "node": n,
            "mag_pu": pu[2 * i],
            "angle_deg": pu[2 * i + 1],
        })

    expected = {1: 0.0, 2: -120.0, 3: 120.0}
    max_dev = max(
        abs(p["angle_deg"] - expected.get(p["node"], p["angle_deg"]))
        for p in per_phase
    )

    return {
        "configured_angle_deg": vsource_info["configured_angle_deg"],
        "solved_per_phase": per_phase,
        "max_deviation_from_ideal_balanced_deg": max_dev,
        "balanced": max_dev < 0.01,
    }


def answer_q4_phase_distribution(buses_info, loads):
    phase_node_counts = {"1": 0, "2": 0, "3": 0}
    for bus in buses_info["buses"]:
        for n in bus["nodes"]:
            if n in (1, 2, 3):
                phase_node_counts[str(n)] += 1

    load_phase_counts = {"1": 0, "2": 0, "3": 0}
    load_phase_kw = {"1": 0.0, "2": 0.0, "3": 0.0}

    for l in loads:
        bus_spec = l["bus"]
        parts = bus_spec.split(".")
        phases_on_bus = [p for p in parts[1:] if p in ("1", "2", "3")]
        if not phases_on_bus:
            phases_on_bus = ["1", "2", "3"]  # unspecified => all 3

        n = len(phases_on_bus)
        for p in phases_on_bus:
            load_phase_counts[p] += 1
            load_phase_kw[p] += l["kw"] / n

    return {
        "phase_node_counts": phase_node_counts,
        "load_phase_counts": load_phase_counts,
        "load_phase_kw": load_phase_kw,
    }


# --------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------

def extract_all():
    """Runs compile_and_solve() first if you haven't already. Returns
    the full inventory dict plus the four blocking-question answers."""

    solve_info = compile_and_solve()

    buses_info = extract_buses_and_nodes()
    loads = extract_loads()
    lines = extract_lines()
    referenced_codes = {l["linecode"] for l in lines}
    linecodes = extract_linecodes(referenced_codes)
    xfmr_reg = extract_transformers_and_regcontrols()
    cap_info = extract_capacitors_and_capcontrols()
    switches = extract_switches(lines)
    vsource = extract_vsource()
    summary = extract_summary(loads)

    q1 = answer_q1_radial(lines, xfmr_reg["transformers"], switches)
    q2 = answer_q2_distinct_load_buses(loads)
    q3 = answer_q3_source_angles(vsource)
    q4 = answer_q4_phase_distribution(buses_info, loads)

    return {
        "solve_info": solve_info,
        "buses_and_nodes": buses_info,
        "loads": loads,
        "lines": lines,
        "linecodes": linecodes,
        "transformers_and_regcontrols": xfmr_reg,
        "capacitors_and_capcontrols": cap_info,
        "switches": switches,
        "vsource": vsource,
        "summary": summary,
        "answers": {
            "q1_radial": q1,
            "q2_distinct_load_buses": q2,
            "q3_source_angles": q3,
            "q4_phase_distribution": q4,
        },
    }
