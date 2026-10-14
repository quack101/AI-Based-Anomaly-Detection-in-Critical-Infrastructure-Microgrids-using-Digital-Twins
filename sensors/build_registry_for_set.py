"""
One scripted pipeline that builds the sensor registry for ANY meter set,
reproducing -- as rules, with no hand-picked channels -- what was done
to set A's registry by hand (reports/registry_pipeline.md has the
commit-by-commit trace):

  stage "noise"  (the registry the corpus noise is drawn from)
    1. sensors/generate_registry.py main(): base entries + per-channel
       sigma (1% of reading for P/Q/|I|, 0.5% of nominal for |V|) for the
       given meter JSON. Default variant: meter P,Q enabled (98 for
       49 meters). Multiphase variant: all phases of 3-phase meters.
    2. Deployment flips on the MULTIPHASE variant only, as rules:
       - Phase 6 (generate_phase6_corpus.py docstring): enable the
         substation head P,Q channels (HEAD_P_*, HEAD_Q_*).
       - commit 0366af8 (2026-10-10): enable every V_magnitude channel
         (all meter |V| + HEAD_V_*). Set A: 56 channels, 112 -> 168.
       Branch-flow channels stay disabled (as in set A).
     For set A this reproduces the never-committed registry the frozen
     corpus was generated with (reports/set_A_corpus_rebuild.md).

  stage "recal"  (the registry the estimator weights with)
    3. Global sigma recalibration (commit 81ff01a,
       validation/phase7_global_sigma_recalibration.py), same formula
       and constants, evaluated on the meter set's OWN corpus:
         reference state: the estimate at test row TARGET_I = 46
         Q-type (Q_injection, Q_head):  amp = |dQ/dV| * 0.04 * v_base
         P-type (P_injection, P_head):  amp = hypot(dP/dV * 0.04 * v_base,
                                                     dP/ddelta * 0.03)
         sigma_new = hypot(sigma_old, amp); V_magnitude untouched.
       Exclusion RULE (decision Q3 = (ii)): channels whose bus is a
       regulator bus (observability.zero_injection._regulator_buses,
       e.g. 150/150r, 9/9r, 25/25r, 160/160r) keep sigma_old. The
       hand-picked M040_P exclusion of 81ff01a is NOT reproduced.
       Applied to both variants with one sigma per channel id, as
       81ff01a did.
     The REGZI_83 amplification (observability/estimator_model.py
     AMPLIFIED_SIGMA_REGZI_IDS) is built from the network layout at a
     flat start, not from the meter set, so it is the same for every
     set and needs no per-set step.

Outputs (never set A's files):
  config/sensor_registry{,_multiphase}_<SET>_noise.json   stage noise
  config/sensor_registry{,_multiphase}_<SET>.json         stage recal

Run:
  python -m sensors.build_registry_for_set noise --set B \
      --meters simulation/representative_meters_B.json
  python -m sensors.build_registry_for_set recal --set B \
      --corpus simulation/phase6_corpus_B
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensors import generate_registry
from sensors.registry import registry_path, validate_registry

BASE_DIR = Path(__file__).resolve().parent.parent
SET_A_FILES = {registry_path("default", "A").resolve(), registry_path("multiphase", "A").resolve()}

TARGET_I = 46
TYPICAL_V_ERR_FRAC = 0.04
TYPICAL_DELTA_ERR_RAD = 0.03
PHASE_NUM = {"A": 1, "B": 2, "C": 3}
P_TYPES = {"P_injection", "P_head"}
Q_TYPES = {"Q_injection", "Q_head"}


def _write(path, entries):
    path = Path(path).resolve()
    if path in SET_A_FILES:
        raise RuntimeError(f"refusing to write set A's registry {path}")
    validate_registry(entries)
    path.write_text(json.dumps(entries, indent=2))
    return path


def apply_deployment_flips(multiphase_entries):
    """Phase-6 head P,Q + 0366af8 all-|V|. Returns the ids switched on."""
    flipped = []
    for e in multiphase_entries:
        is_head_pq = e["id"].startswith(("HEAD_P_", "HEAD_Q_"))
        is_v = e["quantity"] == "V_magnitude"
        if (is_head_pq or is_v) and not e["enabled"]:
            e["enabled"] = True
            flipped.append(e["id"])
    return flipped


def build_noise(meters_path, set_name):
    out = {}
    for variant, mp in (("default", False), ("multiphase", True)):
        with contextlib.redirect_stdout(io.StringIO()):
            entries = generate_registry.main(multiphase=mp, write=False, meters_path=meters_path)
        flipped = apply_deployment_flips(entries) if mp else []
        path = _write(registry_path(variant, f"{set_name}_noise"), entries)
        out[variant] = {"path": str(path), "entries": len(entries),
                        "enabled": sum(e["enabled"] for e in entries), "flipped": len(flipped)}
    print(json.dumps(out, indent=2))
    return out


def build_recal(set_name, corpus_dir):
    from observability.estimator_model import build_estimator_model
    from observability.state_vector import complex_v_from_state
    from observability import jacobian as J
    from observability.zero_injection import _regulator_buses
    from estimation.state_estimator import StateEstimator

    corpus_dir = Path(corpus_dir).resolve()
    noise = {v: json.loads(registry_path(v, f"{set_name}_noise").read_text()) for v in ("default", "multiphase")}

    telem_cols = pd.read_csv(corpus_dir / "test_telemetry.csv", nrows=0).columns.tolist()
    z_cols = [c for c in telem_cols if c not in ("timestamp", "real_timestamp")]
    df = pd.read_csv(corpus_dir / "test_telemetry.csv", usecols=["timestamp", "real_timestamp"] + z_cols, nrows=1440)

    model = build_estimator_model("multiphase", registry_path=registry_path("multiphase", f"{set_name}_noise"),
                                  corpus_dir=corpus_dir)
    state = model.state
    est = StateEstimator(model)
    row = df.iloc[TARGET_I]
    result = est.estimate({c: float(row[c]) for c in z_cols}, str(row["real_timestamp"]))
    at_v = complex_v_from_state(state, result.x_v.copy(), result.x_delta.copy())

    reg_buses = {b.lower() for b in _regulator_buses()}
    computed, excluded = {}, []
    for entries in noise.values():
        for e in entries:
            if not e.get("enabled") or e["id"] in computed or e["id"] in excluded:
                continue
            q = e["quantity"]
            if q not in P_TYPES and q not in Q_TYPES:
                continue
            if str(e["node"]).lower() in reg_buses:
                excluded.append(e["id"])
                continue
            node_key = f"{e['node']}.{PHASE_NUM[e['phase']]}"
            v_idx, v_base = state.v_index[node_key], state.v_base[node_key]
            d_idx = state.delta_index.get(node_key)
            H, _ = J.build_jacobian([e], model.Y, model.node_index, state, at_v=at_v)
            if q in Q_TYPES:
                amp = abs(H[0, v_idx]) * TYPICAL_V_ERR_FRAC * v_base
            else:
                dPdd = H[0, d_idx] if d_idx is not None else 0.0
                amp = float(np.hypot(H[0, v_idx] * TYPICAL_V_ERR_FRAC * v_base, dPdd * TYPICAL_DELTA_ERR_RAD))
            computed[e["id"]] = float(np.hypot(e["sigma"], amp))

    out = {"reference_row": TARGET_I, "converged_at_reference": bool(result.converged),
           "recalibrated": len(computed), "excluded_regulator_bus": sorted(excluded)}
    for variant, entries in noise.items():
        for e in entries:
            if e["id"] in computed:
                old = e["sigma"]
                e["sigma"] = computed[e["id"]]
                e["note"] = (f"sigma recalibrated by sensors/build_registry_for_set.py "
                             f"(rule of 81ff01a, own corpus, row {TARGET_I}): old={old:.6f}, "
                             f"new={e['sigma']:.6f}")
            elif e["id"] in excluded:
                e["note"] = "regulator-bus channel: kept original sigma (recalibration exclusion rule)"
        out[variant] = str(_write(registry_path(variant, set_name), entries))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["noise", "recal"])
    p.add_argument("--set", required=True, dest="set_name")
    p.add_argument("--meters")
    p.add_argument("--corpus")
    a = p.parse_args()
    if a.set_name == "A":
        raise SystemExit("set name 'A' is reserved for the frozen set-A files; use e.g. 'Arebuilt'")
    if a.stage == "noise":
        build_noise(Path(a.meters).resolve(), a.set_name)
    else:
        build_recal(a.set_name, Path(a.corpus).resolve())
