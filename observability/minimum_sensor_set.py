"""
RES.md D10 Phase 5 -- find the MINIMUM sensor configuration (fewest
enabled registry entries) that still achieves rank(H) = dim(x) with
ZERO critical measurements, given Phase 4's own finding that the
MAXIMAL multiphase configuration already achieves both (786 rows).
Phase 5's brief is explicit: "do not simply keep everything enabled."

MANDATORY base (never searched over -- these are not optional
"sensors" in the RES.md D4 4.3 sense, they are structural/necessary by
construction):
  - meters P,Q (106, multiphase variant -- Phase 4's own registry
    decision, not re-litigated here)
  - zero-injection constraints (366, free and exact -- Sha25 Sec II)
  - pseudo-measurements at the 42 unmetered loads (84 -- real,
    unmeasured consumption; omitting them would mean 42 loads'
    injections are simply unknown, not a sensor trade-off)

SEARCHED-OVER additions, in RES.md D4 4.3's cumulative priority order:
  |V| at meters (53) -> substation head (12, sub-searched down to its
  P/Q-only, V-only and I-only sub-groups) -> branch units (165).

Method: for each candidate ADDITION (a full category or one of its
quantity-type sub-groups), build BASE + addition and check whether
n_critical drops to 0 (rank is already 553 at BASE alone in this
feeder -- see MINIMUM_SET_REPORT.base_rank -- so the search question is
CRITICALITY-ELIMINATION, not rank-closing, once BASE is established).
The first (smallest, in RES.md's priority order) addition that reaches
n_critical=0 is reported as the minimum; every other candidate is
reported as redundant FOR THIS SPECIFIC GOAL (they may still carry
independent operational value -- Sec 4.3's own justification for head/
branch/V is about real-time bad-data cross-checking and free AMI
capability, not solely about closing this rank/criticality gap; this
module's job is only to answer the rank/criticality question precisely,
not to make the broader deployment call).
"""

from dataclasses import dataclass, field

import numpy as np
import opendssdirect as dss

from placement.electrical_distance import get_system_y_index
from observability.state_vector import build_state_vector
from observability import jacobian as J
from observability.sweep import build_cumulative_groups, find_critical_measurements


@dataclass
class MinimumSetResult:
    base_rows: int
    base_rank: int
    base_n_critical: int
    base_critical_ids: list
    minimum_addition_name: str
    minimum_addition_entries: list
    final_rows: int
    final_rank: int
    final_n_critical: int
    redundant_candidates: list = field(default_factory=list)  # [(name, n_entries, n_critical_after)]


def _rank_and_critical(entries, state, Y, node_index):
    H, h0 = J.build_jacobian(entries, Y, node_index, state)
    rank = int(np.linalg.matrix_rank(H))
    crit_idx, _ = find_critical_measurements(H)
    return rank, [entries[i]["id"] for i in crit_idx]


def find_minimum_sensor_set(registry_variant="multiphase"):
    state = build_state_vector()
    Y, node_index = get_system_y_index()
    groups = build_cumulative_groups(registry_variant)

    meters_pq = groups["1_meters_pq"]
    meters_v = groups["2_plus_meters_v"]
    head = groups["3_plus_head"]
    branch = groups["4_plus_branch"]
    zero_injection = groups["5_plus_zero_injection"]
    pseudo = groups["6_plus_pseudo"]

    base = meters_pq + pseudo + zero_injection
    base_rank, base_critical = _rank_and_critical(base, state, Y, node_index)

    head_pq = [e for e in head if e["quantity"] in ("P_head", "Q_head")]
    head_v = [e for e in head if e["quantity"] == "V_magnitude"]
    head_i = [e for e in head if e["quantity"] == "I_magnitude_head"]

    # RES.md D4 4.3's own cumulative priority order: |V| at meters,
    # THEN head (smallest sub-group first, since RES.md's Phase-5 "Out"
    # line asks for the head sensor as a whole but the search itself
    # should still report which of its parts is load-bearing), THEN
    # branch.
    candidates = [
        ("meters_V (53)", meters_v),
        ("head_P_Q (6)", head_pq),
        ("head_V (3)", head_v),
        ("head_I (3)", head_i),
        ("head_full (12)", head),
        ("branch_full (165)", branch),
    ]

    minimum_name, minimum_entries = None, None
    final_rank, final_critical = base_rank, base_critical
    redundant = []

    for name, entries in candidates:
        if minimum_name is not None:
            # already found the minimum -- classify every remaining
            # candidate as redundant FOR THIS GOAL without re-testing
            # combinations (RES.md asks for the smallest addition, not
            # an exhaustive Pareto frontier).
            rank_c, crit_c = _rank_and_critical(base + entries, state, Y, node_index)
            redundant.append((name, len(entries), len(crit_c)))
            continue

        rank_c, crit_c = _rank_and_critical(base + entries, state, Y, node_index)
        if len(crit_c) == 0 and rank_c == state.dim:
            minimum_name, minimum_entries = name, entries
            final_rank, final_critical = rank_c, crit_c
        else:
            redundant.append((name, len(entries), len(crit_c)))

    return MinimumSetResult(
        base_rows=len(base), base_rank=base_rank,
        base_n_critical=len(base_critical), base_critical_ids=base_critical,
        minimum_addition_name=minimum_name, minimum_addition_entries=minimum_entries,
        final_rows=len(base) + (len(minimum_entries) if minimum_entries else 0),
        final_rank=final_rank, final_n_critical=len(final_critical),
        redundant_candidates=redundant,
    )
