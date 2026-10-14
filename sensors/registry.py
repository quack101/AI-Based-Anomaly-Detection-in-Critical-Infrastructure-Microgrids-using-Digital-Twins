"""
Sensor registry loading and the ordering invariant.

RES.md D10 Phase 2 flags this explicitly as a silent-wrong-answer bug
class: the order of entries in z, the order of rows in H (Phase 4), and
the order of registry entries must be the SAME, derived from a SINGLE
iteration over the registry. enabled_entries() is that single source of
truth -- sensor_layer.py's z-builder and (in Phase 4) H's row-builder
must both consume its output directly, never re-filter the raw registry
independently.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE_DIR / "config" / "sensor_registry.json"

# RES.md D10 Phase 3 / Phase-2 Finding #1 follow-up: a second registry
# variant with all phases of genuinely multi-phase (3-phase Wye) loads
# metered, selectable by config rather than hardcoded. "default" is the
# untouched, already-validated m=98 registry; callers that want the
# fuller variant pass variant="multiphase" explicitly -- the default
# behaviour of load_registry() is unchanged so every existing caller
# (and the 75 pre-Phase-3 tests) keeps working exactly as before.
REGISTRY_PATHS = {
    "default": REGISTRY_PATH,
    "multiphase": BASE_DIR / "config" / "sensor_registry_multiphase.json",
}

REQUIRED_FIELDS = {"id", "node", "phase", "quantity", "sigma", "rate", "enabled"}
VALID_PHASES = {"A", "B", "C"}

# Meter sets (placement-v2 A-vs-B work). "A" is the frozen set the
# estimator, corpus and attack work are built on, and maps to the
# original file names above, unchanged. Any other set name X maps to
# config/sensor_registry_X.json / config/sensor_registry_multiphase_X.json
# (e.g. "B" from HybridV2, "Arebuilt" from the scripted pipeline).
DEFAULT_METER_SET = "A"


def registry_path(variant="default", meter_set=DEFAULT_METER_SET):
    if variant not in REGISTRY_PATHS:
        raise KeyError(f"unknown registry variant {variant!r}; expected one of {sorted(REGISTRY_PATHS)}")
    if meter_set == DEFAULT_METER_SET:
        return REGISTRY_PATHS[variant]
    if not meter_set or not meter_set.replace("_", "").isalnum():
        raise ValueError(f"invalid meter set name {meter_set!r}")
    base = REGISTRY_PATHS[variant]
    return base.with_name(f"{base.stem}_{meter_set}{base.suffix}")


def load_registry(path=None, variant="default", meter_set=DEFAULT_METER_SET):
    if path is None:
        path = registry_path(variant, meter_set)

    with open(path, "r") as f:
        registry = json.load(f)

    validate_registry(registry)
    return registry


def validate_registry(registry):
    ids_seen = set()

    for entry in registry:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Registry entry {entry.get('id')!r} missing fields: {missing}")

        if entry["id"] in ids_seen:
            raise ValueError(f"Duplicate registry id: {entry['id']!r}")
        ids_seen.add(entry["id"])

        if entry["phase"] not in VALID_PHASES:
            raise ValueError(f"Entry {entry['id']!r}: invalid phase {entry['phase']!r}")

        if entry["sigma"] < 0:
            raise ValueError(f"Entry {entry['id']!r}: sigma must be >= 0")

    return True


def enabled_entries(registry):
    """The SINGLE ordered, filtered list every downstream consumer (z,
    H, Kafka publish order) must be built from -- in FILE ORDER, i.e.
    the order the registry declares them, restricted to enabled=True.
    Call this ONCE per build; pass its result into both z- and
    H-builders. Never filter the raw registry a second time elsewhere."""

    return [entry for entry in registry if entry.get("enabled", False)]


def entry_ids(entries):
    return [entry["id"] for entry in entries]
