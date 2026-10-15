"""
The live streaming path reads its meter list and registry through
sensors.meter_set. Default must stay set A (49 meters, frozen files);
METER_SET / meter_set= selects another committed set. Non-default sets'
registries and the load corpus are generated, not committed, so those
checks skip with a reason when absent.
"""

import json
from pathlib import Path

import pytest

from sensors.meter_set import ENV_VAR, METER_SETS, resolve_meter_set
from sensors.registry import registry_path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset" / "node_loads_91"


def _n(path):
    return len(json.loads(Path(path).read_text()))


def test_default_is_set_a(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    meters, reg = resolve_meter_set()
    assert meters == BASE / "simulation" / "representative_meters.json"
    assert _n(meters) == 49
    assert registry_path("multiphase", reg) == registry_path("multiphase")


def test_env_var_selects_set(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "H45")
    if not registry_path("multiphase", METER_SETS["H45"][1]).exists():
        pytest.skip("H45 registry not built (sensors.build_registry_for_set)")
    meters, _ = resolve_meter_set()
    assert _n(meters) == 45


def test_explicit_name_beats_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "H45")
    meters, _ = resolve_meter_set("A")
    assert _n(meters) == 49


def test_unknown_set_fails_loudly():
    with pytest.raises(KeyError):
        resolve_meter_set("nope")


def test_unbuilt_registry_names_the_build_command(monkeypatch):
    monkeypatch.setitem(METER_SETS, "TMP", (METER_SETS["A"][0], "NotBuiltXyz_noise"))
    with pytest.raises(FileNotFoundError, match="build_registry_for_set"):
        resolve_meter_set("TMP")


def test_every_declared_meter_file_exists():
    for name, (path, _) in METER_SETS.items():
        assert path.exists(), name


def test_producer_and_consumer_agree_on_45_meters():
    if not registry_path("multiphase", METER_SETS["H45"][1]).exists():
        pytest.skip("H45 registry not built (sensors.build_registry_for_set)")
    if not DATASET.exists():
        pytest.skip("dataset/node_loads_91 not present")
    from streaming.consumer import Consumer
    from streaming.producer import Producer

    p = Producer(registry_variant="multiphase", add_noise=False, seed=0, meter_set="H45")
    c = Consumer(registry_variant="multiphase", meter_set="H45")
    assert len(p.representative_meters) == c.expected_meters == 45

    messages, _ = p.run_timestep(0)
    assert len(messages) == 45
    state = None
    for m in messages:
        state = c.ingest(m) or state
    assert state is not None, "window did not close with all 45 meters"
    assert state["dropout_count"] == 0
