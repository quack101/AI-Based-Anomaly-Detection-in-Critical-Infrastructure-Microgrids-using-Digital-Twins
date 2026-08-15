import json
import re
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "123Bus" / "IEEE123Loads.DSS"
OUTPUT_FILE = BASE_DIR / "config" / "load_metadata.json"

# --------------------------------------------------
# Regex
# --------------------------------------------------

pattern = re.compile(
    r"New\s+Load\.(\S+)\s+"
    r"Bus1=(\S+)\s+"
    r"Phases=(\d+)\s+"
    r"Conn=(\w+)\s+"
    r"Model=(\d+)\s+"
    r"kV=([\d.]+)\s+"
    r"kW=([\d.]+)\s+"
    r"kVAR?=([\d.]+)",
    re.IGNORECASE,
)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def parse_bus(bus_string, phases):
    """
    Examples

    1.1       -> bus=1, phase=A
    2.2       -> bus=2, phase=B
    4.3       -> bus=4, phase=C
    35.1.2    -> bus=35, phase=AB
    65.2.3    -> bus=65, phase=BC
    65.3.1    -> bus=65, phase=CA
    47        -> bus=47, phase=ABC
    """

    parts = bus_string.split(".")

    bus = int(parts[0])

    if phases == 3:
        return bus, "ABC"

    phase_lookup = {
        "1": "A",
        "2": "B",
        "3": "C"
    }

    if len(parts) == 2:
        phase = phase_lookup[parts[1]]

    elif len(parts) == 3:
        phase = phase_lookup[parts[1]] + phase_lookup[parts[2]]

    else:
        phase = "Unknown"

    return bus, phase


# --------------------------------------------------
# Parse
# --------------------------------------------------

loads = []

with open(INPUT_FILE, "r") as file:

    for line in file:

        line = line.strip()

        if not line.startswith("New Load"):
            continue

        match = pattern.search(line)

        if not match:
            print("Skipped:", line)
            continue

        name = match.group(1)
        bus_terminal = match.group(2)
        phases = int(match.group(3))

        bus, phase = parse_bus(bus_terminal, phases)

        load = {
            "name": name,
            "bus": bus,
            "bus_terminal": bus_terminal,
            "phase": phase,
            "phases": phases,
            "connection": match.group(4),
            "model": int(match.group(5)),
            "kv": float(match.group(6)),
            "base_kw": float(match.group(7)),
            "base_kvar": float(match.group(8))
        }

        loads.append(load)

# --------------------------------------------------
# Sort by Bus Number
# --------------------------------------------------

loads.sort(key=lambda x: x["bus"])

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(exist_ok=True)

with open(OUTPUT_FILE, "w") as file:
    json.dump(loads, file, indent=4)

print(f"\nParsed {len(loads)} loads.")
print(f"Saved to {OUTPUT_FILE}")