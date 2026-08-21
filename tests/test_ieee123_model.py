"""
Fast, assertive regression tests for core/ieee123_model.py. No file
output -- artifacts/ and reports/ are validation/'s job, not tests/.
"""

import pytest

from core import ieee123_model as m


@pytest.fixture(scope="module")
def extracted():
    return m.extract_all()


def test_compiles_and_converges(extracted):
    assert extracted["summary"]["converged"] is True


def test_phase_node_count_matches_phase0(extracted):
    # Independently measured by RES.md D10 Phase 0 (circuit_topology.json)
    assert extracted["buses_and_nodes"]["num_nodes"] == 278
    assert extracted["buses_and_nodes"]["num_buses"] == 132


def test_all_91_loads_extracted(extracted):
    assert len(extracted["loads"]) == 91
    for load in extracted["loads"]:
        assert load["kw"] > 0
        assert load["connection"] in ("wye", "delta")


def test_q1_base_configuration_is_radial(extracted):
    q1 = extracted["answers"]["q1_radial"]
    assert q1["is_radial"] is True
    assert q1["num_cycles"] == 0
    assert q1["num_components"] == 1
    assert q1["num_edges"] == q1["num_nodes"] - 1
    assert set(q1["open_switches"]) == {"sw7", "sw8"}
    assert set(q1["closed_switches"]) == {"sw1", "sw2", "sw3", "sw4", "sw5", "sw6"}


def test_regulator_bus_pairs_deduplicate_to_single_edges(extracted):
    # reg3a/reg3c (25-25r) and reg4a/reg4b/reg4c (160-160r) are per-phase
    # windings of ONE regulator bank each -- must not appear as parallel
    # graph edges (that was a real bug: it produced 3 false cycles).
    graph = extracted["answers"]["q1_radial"]["graph"]
    parallel_pairs = graph["parallel_object_bus_pairs"]
    assert "25-25r" in parallel_pairs
    assert "160-160r" in parallel_pairs
    assert len(parallel_pairs) == 2


def test_q2_distinct_load_buses(extracted):
    q2 = extracted["answers"]["q2_distinct_load_buses"]
    assert q2["num_loads"] == 91
    assert q2["num_distinct_load_buses"] == 85
    assert set(q2["multi_load_buses"].keys()) == {"49", "65", "76"}


def test_q3_source_angles_balanced(extracted):
    q3 = extracted["answers"]["q3_source_angles"]
    assert q3["balanced"] is True
    assert q3["max_deviation_from_ideal_balanced_deg"] < 0.01


def test_q4_phase_distribution_sums_correctly(extracted):
    q4 = extracted["answers"]["q4_phase_distribution"]
    assert sum(q4["load_phase_counts"].values()) == 102  # 91 loads incl. multi-phase splits...
    total_kw = sum(q4["load_phase_kw"].values())
    assert total_kw == pytest.approx(3490.0, abs=0.5)


def test_regulator_banks_group_correctly(extracted):
    tr = extracted["transformers_and_regcontrols"]
    assert tr["num_regulator_banks"] == 4
    assert set(tr["regulator_banks"].keys()) == {"reg1", "reg2", "reg3", "reg4"}
    assert len(tr["transformers"]) == 8  # 7 regulator windings + XFM1
    assert len(tr["regcontrols"]) == 7


def test_no_capcontrols_all_capacitors_fixed(extracted):
    cap = extracted["capacitors_and_capcontrols"]
    assert len(cap["capacitors"]) == 4
    assert len(cap["capcontrols"]) == 0


def test_extraction_is_json_serializable(extracted):
    import json
    json.dumps(extracted)  # raises TypeError if anything isn't serializable
