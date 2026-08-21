"""
Adapts core/ieee123_model.py's compiled-circuit extraction into the
(network, loads) shape FeederContext expects -- the same shape
config/network_metadata.json + config/load_metadata.json (the regex
parser's output) already provided, so this is a drop-in replacement of
the DATA SOURCE only, not of FeederContext or any placement strategy.

Why this exists: parser/parse_network.py has a known, unresolved,
documented bug (its own commented-out debug output: "edges are 126
instead of 125") -- it does not deduplicate multiple single-phase
transformer objects (regulator bank phases) that connect the same bus
pair, so it overcounts edges. core/ieee123_model.py's graph builder
already deduplicates this correctly (see model_verification_report.txt
FINDINGS #2) and is independently verified: 131 edges, 132 nodes,
0 cycles, radial (reports/model_verification_report.txt). This adapter
reuses that verified graph directly -- it does not rebuild it.
"""


def _restore_open_tie_casing(bus_id):
    """OpenDSS lowercases every bus name it returns (e.g. '300_open'),
    but FeederContext.get_utility_nodes() -- preserved byte-for-byte
    from the original hybrid logic -- checks the literal substring
    '_OPEN' (uppercase, matching how the .dss text itself is written:
    'Bus2=300_OPEN'). Restore that casing at the data-source boundary
    rather than changing the preserved logic."""

    if bus_id.lower().endswith("_open"):
        return bus_id[: -len("_open")] + "_OPEN"
    return bus_id


def network_and_loads_from_core_extraction(extraction):
    """extraction: the dict returned by core.ieee123_model.extract_all().
    Returns (network, loads) in the shape FeederContext/HybridPlacement
    expect."""

    source_bus = extraction["vsource"]["solved_at_source_bus"]["bus"]

    nodes = [
        {"id": _restore_open_tie_casing(bus["name"])}
        for bus in extraction["buses_and_nodes"]["buses"]
    ]

    graph = extraction["answers"]["q1_radial"]["graph"]
    edges = [
        {
            "from": _restore_open_tie_casing(e["from"]),
            "to": _restore_open_tie_casing(e["to"]),
        }
        for e in graph["edges"]
    ]

    network = {
        "source": {"bus": _restore_open_tie_casing(source_bus)},
        "nodes": nodes,
        "edges": edges,
        "assets": {
            "transformers": extraction["transformers_and_regcontrols"]["transformers"],
            "capacitors": extraction["capacitors_and_capcontrols"]["capacitors"],
            "regulators": extraction["transformers_and_regcontrols"]["regcontrols"],
        },
    }

    loads = []
    for load in extraction["loads"]:
        bus_root = _restore_open_tie_casing(load["bus_root"])

        parts = load["bus"].split(".")
        phase_lookup = {"1": "A", "2": "B", "3": "C"}
        raw_phases = [phase_lookup[p] for p in parts[1:] if p in phase_lookup]
        phase = "".join(raw_phases) if raw_phases else "ABC"

        loads.append({
            "name": load["name"],
            "bus": bus_root,
            "bus_terminal": load["bus"],
            "phase": phase,
            "phases": load["phases"],
            "connection": load["connection"],
            "model": load["model"],
            "kv": load["kv"],
            "base_kw": load["kw"],
            "base_kvar": load["kvar"],
        })

    return network, loads
