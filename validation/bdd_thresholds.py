"""
Single source of truth for which BDD decision-threshold file (J99,
LNR99) the Phase 8 validation scripts use. Every attack/scan script
loads its thresholds through load_active_thresholds(), and the
baseline script that measures them writes to ACTIVE_THRESHOLDS_PATH.

Active: config/bdd_thresholds_headpexcl.json -- measured by
validation/phase8_bdd_baseline_headpexcl.py on the HEAD_P-excluded
estimator (tag estimator-v2-headp-excluded). The older
bdd_thresholds_frozen.json and bdd_thresholds_headpfix.json are kept
for reference only; nothing reads them.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ACTIVE_THRESHOLDS_PATH = BASE_DIR / "config" / "bdd_thresholds_headpexcl.json"


def load_active_thresholds():
    """Return (J99, LNR99) as floats from ACTIVE_THRESHOLDS_PATH."""
    with open(ACTIVE_THRESHOLDS_PATH) as f:
        th = json.load(f)
    return float(th["J99"]), float(th["LNR99"])
