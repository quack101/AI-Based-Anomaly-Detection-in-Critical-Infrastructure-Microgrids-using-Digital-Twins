"""
RES.md D10 Phase 4 -- the cumulative sensor-configuration sweep:
{49 P,Q} -> {+|V| at meters} -> {+substation head} -> {+branch units}
-> {+zero-injection} -> {+pseudo-measurements at the 42 unmetered
loads}, for both registry variants (default m=98, multiphase m=106).

Per RES.md D4 Sec 4.3's acceptance test: rank(H), rank(H)/dim(x),
cond(H^T W H), global redundancy (m-rank(H)), and the critical-
measurement set (rows whose removal decreases rank -- Liang Sec V-A /
Bobba et al.'s exact criterion, matching RES.md's own definition
verbatim: "measurements whose removal makes the system unobservable").

Zero-injection rows (Sha25 Sec II eq. 1-6, hard Lagrangian equality
constraints) are included in the RANK computation (observability is a
question about the full linear system: measurements + exact
constraints together), but EXCLUDED from H^T W H (which is specifically
the WEIGHTED NORMAL-EQUATIONS matrix used by the noisy-measurement WLS
estimator -- mixing a near-zero-sigma equality constraint into that
weighted sum is exactly the conditioning damage Sha25 motivates the
Lagrangian form to avoid). cond() is reported for the noisy-measurement
block only; a separate KKT/saddle-point conditioning analysis is Phase
7's estimator-design concern, not Phase 4's observability check.
"""

from dataclasses import dataclass, field

import numpy as np
import opendssdirect as dss

from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector
from observability import jacobian as J
from observability.zero_injection import build_zero_injection_entries
from observability.pseudo_measurements import build_pseudo_entries_with_estimated_sigma
from sensors.registry import load_registry

RANK_TOL = 1e-6  # relative to the largest singular value, numpy default scaling


def _unmetered_load_names(registry):
    metered = {e["load_name"] for e in registry if "load_name" in e}
    all_loads = set(dss.Loads.AllNames())
    return sorted(all_loads - metered, key=str.lower)


def _resolve_bus_and_phase(load_name):
    dss.Circuit.SetActiveElement(f"Load.{load_name}")
    bus = dss.CktElement.BusNames()[0].split(".")[0]
    node_order = dss.CktElement.NodeOrder()
    phase = {1: "A", 2: "B", 3: "C"}[node_order[0]]
    return bus, phase


def build_pseudo_entries(registry):
    """RES.md D10 Phase 5: sigma is ESTIMATED per node from held-out
    IDEAL-corpus variance (observability.pseudo_measurements), not a
    flat asserted fraction -- see that module's docstring for the
    method and the static/offline circularity-immunity rationale."""

    entries, _summaries = build_pseudo_entries_with_estimated_sigma(
        _unmetered_load_names(registry), _resolve_bus_and_phase,
    )
    return entries


def _sigma_for(entry, reading):
    if entry.get("sigma") is not None:
        return entry["sigma"]  # PSEUDO_ entries: pre-estimated, per-node (Phase 5)
    if entry["quantity"] == "V_magnitude":
        return max(abs(1.0) * 0.005, 1e-9)
    return max(abs(reading) * 0.01, 1e-9)  # AMI/SCADA-grade, matches sensors.generate_registry


@dataclass
class ConfigResult:
    variant: str
    config_name: str
    m_measurements: int      # noisy/weighted rows only (excludes zero-injection)
    m_total_rows: int        # including zero-injection, for the rank system
    dim_x: int
    rank: int
    rank_frac: float
    cond_htwh: float
    redundancy: int
    critical_ids: list = field(default_factory=list)
    entry_ids: list = field(default_factory=list)


def build_cumulative_groups(registry_variant="default"):
    """{group_name: [entries]}, in RES.md's cumulative order. Built from
    the RAW registry (ignoring its 'enabled' flags -- the sweep decides
    what's active at each step, independent of the live-pipeline's
    Phase-3 default config)."""

    registry = load_registry(variant=registry_variant)

    meters_pq = [e for e in registry if "load_name" in e and e["quantity"] in ("P_injection", "Q_injection")]
    meters_v = [e for e in registry if "load_name" in e and e["quantity"] == "V_magnitude"]
    head = [e for e in registry if e["id"].startswith("HEAD_")]
    branch = [e for e in registry if e["id"].startswith("BR_")]
    zero_injection = build_zero_injection_entries(dss.Circuit.AllNodeNames())
    pseudo = build_pseudo_entries(registry)

    return {
        "1_meters_pq": meters_pq,
        "2_plus_meters_v": meters_v,
        "3_plus_head": head,
        "4_plus_branch": branch,
        "5_plus_zero_injection": zero_injection,
        "6_plus_pseudo": pseudo,
    }


def _weighted_htwh(H, sigmas):
    w = 1.0 / (np.array(sigmas) ** 2)
    return H.T @ (H * w[:, None])


def find_critical_measurements(H):
    """Row i is critical iff removing it strictly decreases rank(H).
    Liang Sec V-A / Bobba et al.'s exact criterion, matching RES.md's
    own definition verbatim."""

    full_rank = np.linalg.matrix_rank(H, tol=None)
    critical = []
    for i in range(H.shape[0]):
        reduced = np.delete(H, i, axis=0)
        if np.linalg.matrix_rank(reduced, tol=None) < full_rank:
            critical.append(i)
    return critical, full_rank


def run_cumulative_sweep(registry_variant="default", find_critical=True):
    """Returns list[ConfigResult], one per cumulative step."""

    state = build_state_vector()
    Y, node_index = get_system_y_index()
    groups = build_cumulative_groups(registry_variant)

    results = []
    cumulative_entries = []          # noisy/weighted rows (all but zero-injection)
    cumulative_zi_entries = []       # zero-injection rows

    for step_name, group_entries in groups.items():
        if step_name == "5_plus_zero_injection":
            cumulative_zi_entries = list(group_entries)
        else:
            cumulative_entries = cumulative_entries + list(group_entries)

        meas_entries = cumulative_entries
        all_rank_entries = cumulative_entries + cumulative_zi_entries

        if not all_rank_entries:
            continue

        H_rank, h0_rank = J.build_jacobian(all_rank_entries, Y, node_index, state)
        rank = int(np.linalg.matrix_rank(H_rank, tol=None))

        if meas_entries:
            H_meas, h0_meas = J.build_jacobian(meas_entries, Y, node_index, state)
            sigmas = [_sigma_for(e, h0_meas[i]) for i, e in enumerate(meas_entries)]
            HtWH = _weighted_htwh(H_meas, sigmas)
            try:
                cond = float(np.linalg.cond(HtWH))
            except np.linalg.LinAlgError:
                cond = float("inf")
        else:
            cond = float("inf")

        critical_ids = []
        if find_critical and all_rank_entries:
            crit_idx, _ = find_critical_measurements(H_rank)
            critical_ids = [all_rank_entries[i]["id"] for i in crit_idx]

        results.append(ConfigResult(
            variant=registry_variant,
            config_name=step_name,
            m_measurements=len(meas_entries),
            m_total_rows=len(all_rank_entries),
            dim_x=state.dim,
            rank=rank,
            rank_frac=rank / state.dim,
            cond_htwh=cond,
            redundancy=len(all_rank_entries) - rank,
            critical_ids=critical_ids,
            entry_ids=[e["id"] for e in all_rank_entries],
        ))

    return results
