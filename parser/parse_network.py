import json
import re
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_FILE = BASE_DIR / "123Bus" / "IEEE123Master.dss"
OUTPUT_FILE = BASE_DIR / "config" / "network_metadata.json"

# --------------------------------------------------
# Collect all DSS files referenced by the master file
# --------------------------------------------------

def get_dss_files(master_file):

    files = [master_file]

    with open(master_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line.lower().startswith("redirect"):
                continue

            filename = line.split(maxsplit=1)[1].strip()

            files.append(master_file.parent / filename)

    return files
# --------------------------------------------------
# Regex Patterns
# --------------------------------------------------

SOURCE_PATTERN = re.compile(
    r"New\s+object=circuit\.(\S+).*?"
    r"basekv=([\d.]+).*?"
    r"Bus1=(\S+)",
    re.IGNORECASE | re.DOTALL
)

LINE_PATTERN = re.compile(
    r"New\s+Line\.(\S+).*?"
    r"(?:Phases=(\d+).*?)?"
    r"Bus1=(\S+).*?"
    r"Bus2=(\S+).*?"
    r"(?:LineCode=(\S+))?.*?"
    r"Length=([\d.]+)",
    re.IGNORECASE,
)

SWITCH_PATTERN = re.compile(
    r"New\s+Line\.(Sw\d+).*?"
    r"phases=(\d+).*?"
    r"Bus1=(\S+).*?"
    r"Bus2=(\S+).*?"
    r"Length=([\d.]+)",
    re.IGNORECASE,
)

TRANSFORMER_PATTERN = re.compile(
    r"New\s+Transformer\.(\S+)",
    re.IGNORECASE,
)

CAPACITOR_PATTERN = re.compile(
    r"New\s+Capacitor\.(\S+).*?"
    r"Bus1=(\S+).*?"
    r"kVAR=([\d.]+).*?"
    r"kV=([\d.]+)",
    re.IGNORECASE,
)

REGCONTROL_PATTERN = re.compile(
    r"New\s+RegControl\.(\S+).*?"
    r"transformer=(\S+).*?"
    r"winding=(\d+)",
    re.IGNORECASE,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_bus(bus_string):
    """
    25r.1.3
    61s
    300_OPEN
    47.1.2.3
    """

    parts = bus_string.split(".")

    bus = parts[0]

    phase_lookup = {
        "1": "A",
        "2": "B",
        "3": "C"
    }

    phases = []

    for p in parts[1:]:
        if p in phase_lookup:
            phases.append(phase_lookup[p])

    if bus.endswith("_OPEN"):
        kind = "open_tie"

    elif bus.endswith("r"):
        kind = "regulator"

    elif bus.endswith("s"):
        kind = "switch"

    else:
        kind = "bus"

    digits = "".join(c for c in bus if c.isdigit())

    base_bus = int(digits) if digits else None

    return {
        "id": bus,
        "base_bus": base_bus,
        "kind": kind,
        "phases": phases
    }


# --------------------------------------------------
# Containers
# --------------------------------------------------

source = {}

nodes = {}

edges = []

transformers = []

capacitors = []

regulators = []

# --------------------------------------------------
# Parse
# --------------------------------------------------

files = get_dss_files(MASTER_FILE)

for file in files:

    print(f"Parsing {file.name}")

    with open(file, "r") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line or line.startswith("!"):
            i += 1
            continue

        # Merge continuation (~) lines into one command
        command = line

        j = i + 1

        while j < len(lines):

            nxt = lines[j].strip()

            if nxt.startswith("~"):
                command += " " + nxt[1:].strip()
                j += 1
            else:
                break

        i = j

        # ---------------- Source ----------------

        m = SOURCE_PATTERN.search(command)

        if m:

            source = {
                "name": m.group(1),
                "base_kv": float(m.group(2)),
                "bus": m.group(3)
            }

            continue


        # ---------------- Switch ----------------

        m = SWITCH_PATTERN.search(command)

        if m:

            name = m.group(1)

            phases = int(m.group(2))

            bus1 = parse_bus(m.group(3))
            bus2 = parse_bus(m.group(4))

            nodes[bus1["id"]] = bus1
            nodes[bus2["id"]] = bus2

            edges.append({
                "id": name,
                "from": bus1["id"],
                "to": bus2["id"],
                "device": "switch",
                "phases": phases,
                "length": float(m.group(5))
            })

            continue

    # ---------------- Transformer ----------------

        m = TRANSFORMER_PATTERN.search(command)

        if m:

            name = m.group(1)

            # Extract buses=[...]
            bus_match = re.search(r"buses=\[([^\]]+)\]", command, re.IGNORECASE)

            parsed_buses = []

            if bus_match:

                bus_list = bus_match.group(1).split()

                for b in bus_list:

                    info = parse_bus(b)

                    nodes[info["id"]] = info

                    parsed_buses.append(info["id"])

            # Extract all kVA values
            kva = re.findall(r"kvas?=([\d.]+)", command, re.IGNORECASE)

            transformers.append({

                "name": name,

                "buses": parsed_buses,

                "kva": [float(x) for x in kva]

            })

            if len(parsed_buses) == 2:

                edges.append({
                    "id": name,
                    "from": parsed_buses[0],
                    "to": parsed_buses[1],
                    "device": "transformer",
                    "phases": 1,
                    "length": 0
                })

            continue


    # ---------------- RegControl ----------------

        m = REGCONTROL_PATTERN.search(command)

        if m:

            regulators.append({

                "name": m.group(1),

                "transformer": m.group(2),

                "winding": int(m.group(3))

            })

            continue

        # ---------------- Capacitor ----------------

        m = CAPACITOR_PATTERN.search(command)

        if m:

            bus = parse_bus(m.group(2))

            nodes[bus["id"]] = bus

            capacitors.append({

                "name": m.group(1),

                "bus": bus["id"],

                "kvar": float(m.group(3)),

                "kv": float(m.group(4))
            })

            continue

        # ---------------- Line ----------------

        m = LINE_PATTERN.search(command)

        if m:

            name = m.group(1)

            # Prevent switches from being parsed twice
            if name.startswith("Sw"):
                continue

            phases = int(m.group(2)) if m.group(2) else 3

            bus1 = parse_bus(m.group(3))
            bus2 = parse_bus(m.group(4))

            nodes[bus1["id"]] = bus1
            nodes[bus2["id"]] = bus2

            edges.append({
                "id": name,
                "from": bus1["id"],
                "to": bus2["id"],
                "device": "line",
                "phases": phases,
                "linecode": m.group(5),
                "length": float(m.group(6))
            })



# #debug cus edges are 126 instead of 125
# print(f"\nTotal edges: {len(edges)}\n")

# for edge in edges:
#     print(edge["id"], edge["device"])
# #end of debug 

network = {

    "source": source,

    "nodes": sorted(
        nodes.values(),
        key=lambda x: (
            x["base_bus"] if x["base_bus"] is not None else 9999,
            x["id"]
        )
    ),

    "edges": edges,

    "assets": {
        "transformers": transformers,

        "capacitors": capacitors,

        "regulators": regulators
    }

}

OUTPUT_FILE.parent.mkdir(exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(network, f, indent=4)

print(f"Source Bus      : {source.get('bus')}")
print(f"Nodes Parsed    : {len(network['nodes'])}")
print(f"Edges Parsed    : {len(edges)}")
print(f"Saved to {OUTPUT_FILE}")