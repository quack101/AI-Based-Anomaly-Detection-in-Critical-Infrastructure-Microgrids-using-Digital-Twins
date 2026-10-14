"""
select_meters.py must never overwrite set A (simulation/representative_
meters.json) unless --overwrite-set-A is given, and must default to v1.
resolve_output() is pure, so most cases run without placement; the
end-to-end refusal case checks that set A's bytes are untouched.
"""

import hashlib
from pathlib import Path

import pytest

from placement.select_meters import (
    SET_A_OUTPUT, SET_B_OUTPUT, SET_C_OUTPUT, SetAOverwriteRefused, build_parser, main, resolve_output,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_default_strategy_is_v1():
    assert build_parser().parse_args([]).strategy == "v1"


def test_v1_without_out_is_refused():
    with pytest.raises(SetAOverwriteRefused):
        resolve_output("v1", None, overwrite_set_a=False)


def test_v1_without_out_allowed_with_flag():
    assert resolve_output("v1", None, overwrite_set_a=True) == SET_A_OUTPUT


def test_v2_without_out_goes_to_set_b():
    assert resolve_output("v2", None, overwrite_set_a=False) == SET_B_OUTPUT


def test_v3_without_out_goes_to_set_c():
    assert resolve_output("v3", None, overwrite_set_a=False) == SET_C_OUTPUT


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_explicit_out_pointing_at_set_a_is_refused(strategy):
    with pytest.raises(SetAOverwriteRefused):
        resolve_output(strategy, str(SET_A_OUTPUT), overwrite_set_a=False)


def test_explicit_relative_out_pointing_at_set_a_is_refused(monkeypatch):
    monkeypatch.chdir(SET_A_OUTPUT.parent.parent)
    with pytest.raises(SetAOverwriteRefused):
        resolve_output("v2", "simulation/representative_meters.json", overwrite_set_a=False)


def test_other_out_is_accepted(tmp_path):
    out = tmp_path / "meters.json"
    assert resolve_output("v1", str(out), overwrite_set_a=False) == out


def test_refused_run_leaves_set_a_untouched():
    before = _sha(SET_A_OUTPUT)
    assert main(["--strategy", "v1"]) == 2
    assert _sha(SET_A_OUTPUT) == before
