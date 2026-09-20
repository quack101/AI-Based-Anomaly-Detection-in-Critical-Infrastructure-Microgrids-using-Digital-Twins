"""
RES.md D2: only the sensors module (plus the ground-truth generator
itself) may read a solved circuit and emit measurements. estimation/,
detection/ and evaluation/ consume z (Layer-2 telemetry, already
published) -- they must never import opendssdirect directly, nor any
module in this codebase that itself reaches a live/ground-truth circuit
instance.

estimation/ now exists (RES.md D10 Phase 7, the state estimator) --
detection/ and evaluation/ do not yet. test_no_ground_truth_circuit_
access_outside_sensors is no longer vacuous for estimation/: it caught
a real violation during Phase 7's own development (state_estimator.py
originally imported opendssdirect and simulation.opendss_utils
directly to compile the feeder and extract the Y-bus itself) and was
the reason that model-building work was moved to
observability/estimator_model.py instead -- estimation/state_
estimator.py now only ever receives an already-built model and a plain
z dict, never touching a circuit itself.
"""

import ast
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESTRICTED_DIRS = ["estimation", "detection", "evaluation"]

# Modules that reach a live/ground-truth OpenDSS circuit instance --
# forbidden inside RESTRICTED_DIRS. sensors.registry is deliberately
# NOT here: it only loads/validates the JSON registry (ids, sigma,
# rate, enabled flags) and never touches a circuit -- consuming sensor
# METADATA is fine; reaching the circuit itself is not.
FORBIDDEN_MODULES = {
    "opendssdirect",
    "core.ieee123_model",
    "sensors.measurement_functions",
    "sensors.sensor_layer",
    "sensors.generate_registry",
    "simulation.opendss_utils",
}


def _imported_module_names(py_file):
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    return modules


def _is_forbidden(module_name):
    return any(
        module_name == forbidden or module_name.startswith(forbidden + ".")
        for forbidden in FORBIDDEN_MODULES
    )


def test_no_ground_truth_circuit_access_outside_sensors():
    violations = []

    for dirname in RESTRICTED_DIRS:
        dir_path = BASE_DIR / dirname
        if not dir_path.exists():
            continue  # vacuous pass -- this phase doesn't exist yet

        for py_file in dir_path.rglob("*.py"):
            imported = _imported_module_names(py_file)
            forbidden_hits = {m for m in imported if _is_forbidden(m)}

            if forbidden_hits:
                violations.append((str(py_file.relative_to(BASE_DIR)), sorted(forbidden_hits)))

    assert not violations, (
        "Layer-1/Layer-2 boundary violated -- these modules under "
        "estimation/, detection/ or evaluation/ import something that "
        f"reaches a ground-truth circuit instance directly: {violations}"
    )


def test_estimation_exists_and_is_actively_checked_not_vacuous():
    """RES.md D10 Phase 7 created estimation/ -- confirm the boundary
    check above is genuinely exercising it (has .py files to scan),
    not silently passing because the directory is empty. detection/ and
    evaluation/ remain future phases; documented here so the next phase
    that creates one of them knows to update this list, not because
    their absence is itself asserted (that assertion is what USED to
    live here, before estimation/ existed)."""

    assert (BASE_DIR / "estimation").exists()
    py_files = list((BASE_DIR / "estimation").rglob("*.py"))
    assert len(py_files) > 0, "estimation/ exists but has no .py files -- the boundary check above would be vacuous"

    for dirname in ("detection", "evaluation"):
        assert not (BASE_DIR / dirname).exists(), (
            f"{dirname}/ now exists -- add it to this test's own awareness "
            "(it already participates in RESTRICTED_DIRS/the boundary check "
            "above automatically; this assertion is just tracking which "
            "phases have actually arrived)"
        )
