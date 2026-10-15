"""
Named meter sets for the live streaming path (producer, consumer, injector).

Each set is a representative-meter JSON plus the registry that describes
its channels (sensors.registry.registry_path's `meter_set` suffix). The
active set is chosen by the METER_SET environment variable and defaults
to "A", the frozen 49-meter set the estimator and corpus are built on, so
nothing changes unless it is set. All three streaming processes read it
through resolve_meter_set(), so they cannot disagree on the meter list.

This module only resolves paths; it never reads a circuit.

Non-default sets' registries are generated, not committed
(sensors/build_registry_for_set.py); selecting one that has not been
built raises FileNotFoundError that names the command to build it.
"""

import os
from pathlib import Path

from sensors.registry import DEFAULT_METER_SET, registry_path

BASE_DIR = Path(__file__).resolve().parent.parent
SIM = BASE_DIR / "simulation"

ENV_VAR = "METER_SET"
DEFAULT_NAME = "A"

# name -> (meter JSON, registry meter_set suffix). The "_noise" registry is
# the one each set's corpus noise was drawn from (the producer's registry).
METER_SETS = {
    "A": (SIM / "representative_meters.json", DEFAULT_METER_SET),
    "B": (SIM / "representative_meters_B.json", "Ball91_noise"),
    "H30": (SIM / "representative_meters_H30.json", "H30all91_noise"),
    "H45": (SIM / "representative_meters_H45.json", "H45all91_noise"),
}


def resolve_meter_set(name=None):
    """Return (meters_json_path, registry_meter_set) for `name`, or for
    $METER_SET, or for set A. Raises KeyError for an unknown name and
    FileNotFoundError if the set's multiphase registry has not been built."""

    name = name or os.environ.get(ENV_VAR) or DEFAULT_NAME
    if name not in METER_SETS:
        raise KeyError(f"unknown meter set {name!r}; expected one of {sorted(METER_SETS)}")

    meters_path, registry_set = METER_SETS[name]
    if registry_set != DEFAULT_METER_SET:
        reg = registry_path("multiphase", registry_set)
        if not reg.exists():
            raise FileNotFoundError(
                f"meter set {name!r}: registry {reg.name} not built; run "
                f"python -m sensors.build_registry_for_set noise --set "
                f"{registry_set.removesuffix('_noise')} --meters {meters_path.relative_to(BASE_DIR)}"
            )
    return meters_path, registry_set
