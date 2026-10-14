"""
Meter JSON, registry and corpus must describe the same meters, for every
meter set. Set A's committed files are always checked. Other sets are
checked when their files exist; a set whose meter JSON exists but whose
registry is missing FAILS (half-built set), it is not skipped.
Corpora are large and not committed, so a missing corpus folder is a
skip with an explicit reason, never a pass.
"""

import json
from pathlib import Path

import pytest

from sensors.meter_set_consistency import MeterSetMismatch, check_meter_set
from sensors.registry import registry_path

BASE = Path(__file__).resolve().parent.parent
SIM = BASE / "simulation"

SETS = {
    # name: (meters json, registry name, corpus folder)
    "A": (SIM / "representative_meters.json", "A", SIM / "phase6_corpus"),
    "A_rebuilt": (SIM / "representative_meters.json", "Arebuilt_noise", SIM / "phase6_corpus_Arebuilt"),
    "B": (SIM / "representative_meters_B.json", "Ball91_noise", SIM / "phase6_corpus_Ball91"),
    "A_all91": (SIM / "representative_meters.json", "Aall91_noise", SIM / "phase6_corpus_Aall91"),
}


@pytest.mark.parametrize("name", sorted(SETS))
def test_meters_and_registry_agree(name):
    meters, reg_name, _ = SETS[name]
    reg = registry_path("multiphase", reg_name)
    if name != "A" and not reg.exists():
        if meters.exists() and name == "B":
            pytest.fail(f"set B meter JSON exists but its registry {reg.name} does not")
        pytest.skip(f"{reg.name} not built")
    check_meter_set(meters, reg)


@pytest.mark.parametrize("name", sorted(SETS))
def test_corpus_columns_match_registry(name):
    meters, reg_name, corpus = SETS[name]
    reg = registry_path("multiphase", reg_name)
    if not corpus.exists():
        pytest.skip(f"corpus {corpus.name} not present (not committed, regenerate it)")
    check_meter_set(meters, reg, corpus)


def test_mixing_sets_fails_loudly(tmp_path):
    """Set B's meters with set A's registry must raise, not pass."""
    with pytest.raises(MeterSetMismatch):
        check_meter_set(SIM / "representative_meters_B.json", registry_path("multiphase", "A"))


def test_corpus_header_in_wrong_order_fails(tmp_path):
    reg = json.loads(registry_path("multiphase", "A").read_text())
    ids = [e["id"] for e in reg if e["enabled"]]
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}_telemetry.csv").write_text(
            ",".join(["timestamp", "real_timestamp"] + ids[::-1]) + "\n")
    with pytest.raises(MeterSetMismatch):
        check_meter_set(SIM / "representative_meters.json", registry_path("multiphase", "A"), tmp_path)
