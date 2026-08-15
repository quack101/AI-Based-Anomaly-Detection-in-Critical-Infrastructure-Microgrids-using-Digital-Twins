from pathlib import Path
import json
import opendssdirect as dss

from opendss_utils import (
    compile_feeder,
    get_load_telemetry,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

REPRESENTATIVE_METERS = (
    BASE_DIR /
    "simulation" /
    "representative_meters.json"
)

# --------------------------------------------------
# Load representative meters
# --------------------------------------------------

with open(REPRESENTATIVE_METERS, "r") as f:
    representative_meters = json.load(f)

print(f"Representative meters loaded: {len(representative_meters)}")

# --------------------------------------------------
# Compile feeder
# --------------------------------------------------

compile_feeder()

# --------------------------------------------------
# Verify OpenDSS
# --------------------------------------------------

print("\n===== OpenDSS =====")

print("Converged:", dss.Solution.Converged())
print("Buses:", dss.Circuit.NumBuses())
print("Loads:", dss.Loads.Count())
print("Lines:", dss.Lines.Count())

print("\nFirst five loads:")

for load in dss.Loads.AllNames()[:5]:
    print(load)

# --------------------------------------------------
# Verify representative meters
# --------------------------------------------------

print("\n===== VERIFY REPRESENTATIVE METERS =====")

dss_loads = {
    name.lower()
    for name in dss.Loads.AllNames()
}

missing = []

for meter in representative_meters:

    load_name = meter["load_name"].lower()

    if load_name in dss_loads:
        print(f"{meter['meter_id']:>4}  {meter['load_name']:<6} ✓")
    else:
        print(f"{meter['meter_id']:>4}  {meter['load_name']:<6} ✗")
        missing.append(meter["load_name"])

print(
    f"\nMatched: "
    f"{len(representative_meters)-len(missing)}/"
    f"{len(representative_meters)}"
)

if missing:
    print("\nMissing loads:")
    for load in missing:
        print(load)

# --------------------------------------------------
# Create lookup table
# --------------------------------------------------

meter_lookup = {
    meter["load_name"].lower(): meter
    for meter in representative_meters
}

# --------------------------------------------------
# Collect telemetry
# --------------------------------------------------

telemetry = []

for load_name in dss.Loads.AllNames():

    if load_name.lower() not in meter_lookup:
        continue

    meter = meter_lookup[load_name.lower()]

    dss.Loads.Name(load_name)

    # --------------------------------------------------
    # Debugging (keep for documentation)
    # --------------------------------------------------

    if load_name.lower() == "s6c":

        print("\n===== S6 LOAD PARAMETERS =====")

        print("kW      :", dss.Loads.kW())
        print("kvar    :", dss.Loads.kvar())
        print("Model   :", dss.Loads.Model())
        print("PF      :", dss.Loads.PF())
        print("kV      :", dss.Loads.kV())
        print("Daily   :", dss.Loads.Daily())
        print("Yearly  :", dss.Loads.Yearly())
        print("Status  :", dss.Loads.Status())

        print("Powers:", dss.CktElement.Powers())

    bus = dss.CktElement.BusNames()[0]

    values = get_load_telemetry(
        load_name,
        debug=True,
    )

    telemetry.append({

        "timestamp": 0,

        "meter_id": meter["meter_id"],

        "load_name": load_name,

        "bus": bus,

        "voltage_v": values["voltage_v"],

        "voltage_pu": values["voltage_pu"],

        "current_a": values["current_a"],

        "p_kw": values["p_kw"],

        "q_kvar": values["q_kvar"],

    })

# --------------------------------------------------
# Display sample
# --------------------------------------------------

print("\n===== TELEMETRY SAMPLE =====")

for row in telemetry[:5]:
    print(row)