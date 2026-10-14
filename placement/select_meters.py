"""
Orchestration entry point: run a placement strategy against the verified
compiled-circuit model (core/ieee123_model.py) and save the selected
meters as a representative-meter JSON.

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

Set A protection: simulation/representative_meters.json is set A, the
frozen meter set the estimator, Phase-6 corpus and attack work are built
on (and it is NOT what the current v1 code reproduces -- see
reports/set_A_reproducibility.md). Any run that would write to it is
refused unless --overwrite-set-A is given, and the check runs before any
placement work, so a refused run writes nothing.

  --strategy v1 (default) | v2 | v3 HybridPlacement | HybridV2Placement | HybridV3Placement
  --out PATH                         where to save
  no --out, v2                       -> simulation/representative_meters_B.json
  no --out, v3                       -> simulation/representative_meters_C.json
  no --out, v1                       -> refused (would be set A) unless
                                        --overwrite-set-A

Run: python -m placement.select_meters [--strategy v1|v2|v3] [--out PATH]
                                       [--overwrite-set-A] [--target N] [--verbose]
"""

import argparse
import sys
from pathlib import Path

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import HybridPlacement
from placement.hybrid_v2 import HybridV2Placement
from placement.hybrid_v3 import HybridV3Placement
from placement.metrics import evaluate
from placement.representative_meters import save_representative_meters

BASE_DIR = Path(__file__).resolve().parent.parent

SET_A_OUTPUT = BASE_DIR / "simulation" / "representative_meters.json"
SET_B_OUTPUT = BASE_DIR / "simulation" / "representative_meters_B.json"
SET_C_OUTPUT = BASE_DIR / "simulation" / "representative_meters_C.json"
# Kept for any caller that imported the old name; it is set A's path.
REPRESENTATIVE_OUTPUT = SET_A_OUTPUT

STRATEGIES = {"v1": HybridPlacement, "v2": HybridV2Placement, "v3": HybridV3Placement}
DEFAULT_OUTPUTS = {"v2": SET_B_OUTPUT, "v3": SET_C_OUTPUT}


class SetAOverwriteRefused(Exception):
    """Raised when a run would write set A without --overwrite-set-A."""


def resolve_output(strategy, out, overwrite_set_a):
    """Pick the output path for a run, refusing to write set A unless
    explicitly allowed. Pure function (no I/O) so it is testable without
    running placement."""

    if out is None:
        path = DEFAULT_OUTPUTS.get(strategy, SET_A_OUTPUT)
    else:
        path = Path(out)
        if not path.is_absolute():
            path = (Path.cwd() / path)

    if path.resolve() == SET_A_OUTPUT.resolve() and not overwrite_set_a:
        hint = ("pass --out PATH to save elsewhere" if out is None
                else "choose a different --out")
        raise SetAOverwriteRefused(
            f"refusing to overwrite set A ({SET_A_OUTPUT}); {hint}, or add "
            f"--overwrite-set-A if replacing set A is really intended"
        )
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="v1",
                        help="v1 = HybridPlacement (default), v2 = HybridV2Placement, "
                             "v3 = HybridV3Placement")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: set B path for v2, set C path for v3; "
                             "v1 requires --out or --overwrite-set-A)")
    parser.add_argument("--overwrite-set-A", dest="overwrite_set_a", action="store_true",
                        help="allow writing simulation/representative_meters.json (set A)")
    parser.add_argument("--target", type=int, default=49, help="target_meters")
    parser.add_argument("--verbose", action="store_true",
                         help="print topology graph, bus loads, per-load "
                              "branch assignment (32-run sweeps should NOT "
                              "use this -- Step-3 defect #3)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        output = resolve_output(args.strategy, args.out, args.overwrite_set_a)
    except SetAOverwriteRefused as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)

    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    strategy = STRATEGIES[args.strategy](target_meters=args.target, verbose=args.verbose)
    selected = strategy.select(loads, context)

    save_representative_meters(selected, output)

    evaluate(selected, loads)

    print(f"Strategy: {args.strategy} ({type(strategy).__name__})")
    print(f"Selected {len(selected)} representative meters")

    for load in selected:
        print(
            load["name"],
            load["bus"],
            load["base_kw"],
            load["size_class"],
            round(load["score"], 3),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
