"""
Everything the figures need about the feeder, as plain data.

Sources: the compiled-circuit extraction (core.ieee123_model, same one
placement uses), 123Bus/BusCoords.dat for the layout, a representative-
meter JSON from sensors.meter_set.METER_SETS, and
observability.zero_injection for regulator/switch buses. No result is
hardcoded.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COORDS = BASE_DIR / "123Bus" / "BusCoords.dat"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MEASUREMENTS_DIR = BASE_DIR / "reports" / "measurements"


@dataclass
class Feeder:
    source: str
    buses: list
    edges: list                   # (from, to)
    pos: dict                     # bus -> (x, y)
    loads: list                   # load dicts (name, bus, phase, phases, base_kw, ...)
    regulator_buses: set
    regulator_edges: set          # frozenset({a, b})
    switch_buses: set
    capacitor_buses: set
    backbone: list = field(default_factory=list)
    missing_coords: list = field(default_factory=list)


def _bus_of(name):
    return str(name).split(".")[0]


def load_feeder():
    from core.ieee123_model import extract_all          # compiles OpenDSS, changes cwd
    from placement.feeder_adapter import network_and_loads_from_core_extraction
    from placement.feeder_context import FeederContext
    from observability.zero_injection import _regulator_buses, _switch_buses

    cwd = os.getcwd()
    try:
        extraction = extract_all()
    finally:
        os.chdir(cwd)
    network, loads = network_and_loads_from_core_extraction(extraction)

    buses = [n["id"] for n in network["nodes"]]
    edges = [(e["from"], e["to"]) for e in network["edges"]]

    pos = {}
    for line in COORDS.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            pos[parts[0]] = (float(parts[1]), float(parts[2]))

    # a bus without a coordinate is drawn next to a neighbour that has one
    missing = [b for b in buses if b not in pos]
    nbrs = {}
    for a, b in edges:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    for i, b in enumerate(missing):
        for n in nbrs.get(b, []):
            if n in pos:
                pos[b] = (pos[n][0] + 25 * (i + 1), pos[n][1] + 25)
                break

    assets = network.get("assets", {})
    reg_edges = set()
    for t in assets.get("transformers", []):
        if t.get("is_regulator"):
            w = t["windings"]
            reg_edges.add(frozenset((_bus_of(w[0]["bus"]), _bus_of(w[1]["bus"]))))
    cap_buses = {_bus_of(c["bus_root"]) for c in assets.get("capacitors", [])}

    ctx = FeederContext(network, loads)
    ctx.annotate_loads(loads)

    return Feeder(
        source=network["source"]["bus"], buses=buses, edges=edges, pos=pos, loads=loads,
        regulator_buses=set(_regulator_buses()), regulator_edges=reg_edges,
        switch_buses=set(_switch_buses()), capacitor_buses=cap_buses,
        backbone=list(ctx.main_feeder), missing_coords=missing,
    )


def load_meter_set(name):
    """Meters (list of dicts) of a named set, straight from its JSON."""
    from sensors.meter_set import METER_SETS
    return json.loads(Path(METER_SETS[name][0]).read_text())


def metered_buses(meters):
    return {str(m["bus"]) for m in meters}


def phase_kw(loads, only_names=None):
    """kW per phase letter; a three-phase load is split equally."""
    tot = {"A": 0.0, "B": 0.0, "C": 0.0}
    for l in loads:
        if only_names is not None and l["name"] not in only_names:
            continue
        if l.get("phases", 1) == 3 or len(str(l["phase"])) > 1:
            for p in "ABC":
                tot[p] += l["base_kw"] / 3.0
        else:
            tot[str(l["phase"])] += l["base_kw"]
    return tot


def load_measurement(label):
    p = MEASUREMENTS_DIR / f"{label}.json"
    return json.loads(p.read_text()) if p.exists() else None
