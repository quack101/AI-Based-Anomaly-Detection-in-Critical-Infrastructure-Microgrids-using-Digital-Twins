import json
from pathlib import Path

from placement_strategy import HybridRepresentativePlacement
from metrics import evaluate
from representative_meters import save_representative_meters

BASE_DIR = Path(__file__).resolve().parent.parent

LOAD_INPUT = BASE_DIR / "config" / "load_metadata.json"
NETWORK_INPUT = BASE_DIR / "config" / "network_metadata.json"

REPRESENTATIVE_OUTPUT = (BASE_DIR /"simulation" /"representative_meters.json")

# ----------------------------------------------------
# Load metadata
# ----------------------------------------------------

with open(LOAD_INPUT) as f:
    loads = json.load(f)

with open(NETWORK_INPUT) as f:
    network = json.load(f)

# ----------------------------------------------------
# Run placement
# ----------------------------------------------------

strategy = HybridRepresentativePlacement(target_meters=49)

selected = strategy.select(loads, network)

save_representative_meters(
    selected,
    REPRESENTATIVE_OUTPUT
)

evaluate(selected, loads)

# ----------------------------------------------------
# Display Results
# ----------------------------------------------------

print(f"Selected {len(selected)} representative meters")

for load in selected:
    print(
        load["name"],
        load["bus"],
        load["base_kw"],
        load["size_class"],
        round(load["score"], 3)
    )