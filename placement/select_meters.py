"""
Orchestration entry point: run HybridPlacement against the verified
compiled-circuit model (core/ieee123_model.py) and save
simulation/representative_meters.json.

Data source note: this used to read config/network_metadata.json (the
regex .dss parser's output), which has a known, unfixed bug -- it does
not deduplicate multiple single-phase transformer objects (regulator
bank phases) on the same bus pair, and separately drops the XFM1
transformer's edge entirely (its bus=/wdg= syntax doesn't match the
parser's `buses=[...]` regex). See
reports/placement_refactor_report.txt for the full comparison. Now
sources from core.ieee123_model.extract_all() via feeder_adapter.py,
which reuses core's already-verified, deduplicated graph (131 edges,
132 nodes, 0 cycles -- reports/model_verification_report.txt).

Run: python -m placement.select_meters [--verbose] [--target N]
"""

import argparse
from pathlib import Path

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import HybridPlacement
from placement.metrics import evaluate
from placement.representative_meters import save_representative_meters

BASE_DIR = Path(__file__).resolve().parent.parent

REPRESENTATIVE_OUTPUT = BASE_DIR / "simulation" / "representative_meters.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=49, help="target_meters")
    parser.add_argument("--verbose", action="store_true",
                         help="print topology graph, bus loads, per-load "
                              "branch assignment (32-run sweeps should NOT "
                              "use this -- Step-3 defect #3)")
    args = parser.parse_args()

    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)

    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    strategy = HybridPlacement(target_meters=args.target, verbose=args.verbose)
    selected = strategy.select(loads, context)

    save_representative_meters(selected, REPRESENTATIVE_OUTPUT)

    evaluate(selected, loads)

    print(f"Selected {len(selected)} representative meters")

    for load in selected:
        print(
            load["name"],
            load["bus"],
            load["base_kw"],
            load["size_class"],
            round(load["score"], 3),
        )


if __name__ == "__main__":
    main()
