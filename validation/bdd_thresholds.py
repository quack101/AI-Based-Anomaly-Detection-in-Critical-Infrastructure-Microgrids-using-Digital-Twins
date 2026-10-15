"""
Single source of truth for which BDD decision-threshold file (J99,
LNR99) the Phase 8 validation scripts use. Every attack/scan script
loads its thresholds through load_active_thresholds(), and the
baseline script that measures them writes to ACTIVE_THRESHOLDS_PATH.

Active (branch fix-h-yload): config/bdd_thresholds_v5.json -- measured
by validation/phase8_bdd_baseline_pfstart.py with the estimator defaults
of estimator-v5-hfix (network-Y measurement model, S1 + structural
sigmas, power-flow start). The older bdd_thresholds_*.json files are
kept for reference only; nothing reads them.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ACTIVE_THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_v5.json"


def load_active_thresholds():
    """Return (J99, LNR99) as floats from ACTIVE_THRESHOLDS_PATH."""
    with open(ACTIVE_THRESHOLDS_PATH) as f:
        th = json.load(f)
    return float(th["J99"]), float(th["LNR99"])
