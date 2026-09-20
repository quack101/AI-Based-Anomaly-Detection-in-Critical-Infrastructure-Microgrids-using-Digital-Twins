"""
RES.md D4 Sec 4.3 row 0 -- zero-injection equality constraints at
load-free phase-nodes. "Most of the 278 phase-nodes carry no load; each
contributes two exact constraints at zero cost." Enumerated from the
COMPILED circuit (every Load object's bus), never from design-level
metadata, matching D4 Sec 4.2(a)'s instruction for N_phase_nodes itself.

Sha25 Sec II eq. (1)-(6): these are HARD Lagrangian equality
constraints, not high-weight pseudo-measurements -- kept structurally
distinct from row 1 (pseudo-measurements at unmetered LOAD nodes, which
DO have real, just-unmeasured consumption). This module only builds
row 0; row 1 (pseudo-measurements) reuses the ordinary bus-injection
registry-entry machinery in sweep.py, since those nodes are functionally
identical to a metered P/Q entry, just with sigma reflecting historical-
data uncertainty instead of AMI-instrument accuracy.
"""

import opendssdirect as dss

PHASE_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C"}


def _source_bus():
    dss.Circuit.SetActiveElement("Vsource.source")
    return dss.CktElement.BusNames()[0].split(".")[0].lower()


def _regulator_buses():
    """Every bus touching a voltage-regulator transformer, BOTH the
    primary and secondary/regulated side -- e.g. 150/150r, 9/9r, 25/25r,
    160/160r on this feeder. Found in Phase 7, not Phase 4: a
    regulator's effective admittance depends on its CURRENT TAP
    POSITION, a discrete control state that changes over time in
    response to loading (confirmed empirically -- reg1a's tap moved
    from 6 to 5 after 200 timesteps of real, time-varying load, starting
    from the same nominal-load compile this module's Y-bus is always
    built from). A zero-injection constraint at a regulator-adjacent
    bus is only as exact as the Y-bus's frozen tap snapshot -- for a
    timestamp far from that snapshot (Phase 6's test split is 51+ days
    / 73,000+ timesteps past it), the mismatch is large enough to break
    Phase 7's Gauss-Newton solve, which treats zero-injection as EXACT.

    THIS FUNCTION ONLY IDENTIFIES the regulator-adjacent bus set -- it
    does not, by itself, decide what happens to those nodes. Two things
    were tried, in this order, both logged in
    reports/phase7_state_estimator_report.txt:

    1. EXCLUDE them entirely (drop the constraint, do nothing else).
       Tested both secondary-only (dof=-17, and the Gauss-Newton solve
       still diverged from a correct starting point -- 0.98 pu error
       after 30 iterations, warm-started AT x_true) and primary+
       secondary (dof=-29, MORE rank given up, but converged cleanly --
       0.05-0.09 pu error). Neither is acceptable on its own: the first
       still breaks convergence, the second gives up dof/rank Phase
       4/5's full-rank finding relied on.
    2. KEEP them in the problem as WEIGHTED pseudo-measurements instead
       of dropping them (enumerate_regulator_adjacent_nodes /
       build_regulator_pseudo_entries, below) -- target 0 like a true
       zero-injection node, but with a real, empirically-estimated
       sigma reflecting tap uncertainty specifically, rather than
       asserting exactly 0. This is the design actually shipped: it
       restores dof to a small NON-NEGATIVE value (consistent with
       Phase 4/5) while converging as cleanly as the broad-exclusion
       case, because a weighted row with nonzero sigma never creates an
       infeasible hard constraint the way excluding-then-still-
       expecting-0 does. Uses the broader (primary+secondary) bus set
       from step 1, carried forward rather than re-litigated, since it
       is the more conservative (larger) of the two candidate sets."""

    buses = set()
    for rc in dss.RegControls.AllNames():
        dss.RegControls.Name(rc)
        xfmr = dss.RegControls.Transformer()
        dss.Circuit.SetActiveElement(f"Transformer.{xfmr}")
        for b in dss.CktElement.BusNames():
            buses.add(b.split(".")[0].lower())
    return buses


def _load_bus_phases():
    """{(bus_root_lower, phase_num)} for every (bus,phase) a Load object
    is attached to. Requires a compiled circuit already active. Shared
    by enumerate_zero_injection_nodes and enumerate_regulator_adjacent_
    nodes so both use the identical, bug-fixed load-phase identification.

    BUG FIXED HERE (found in Phase 7, building on this function):
    truncating to node_order[:num_phases] is correct for a Wye load
    (node_order lists the real phases first, then a trailing 0 for
    neutral, e.g. [1,2,3,0] for a 3-phase Wye load, num_phases=3) but
    WRONG for a Phases=1 Conn=Delta load (e.g. S35a: node_order=[1,2],
    num_phases=1 -- num_phases counts ONE complex power value, but the
    load's current physically flows through BOTH delta terminals, so
    BOTH phases carry nonzero injection). Confirmed via h_zi(x_true):
    near-exact (~1e-11 kW) at every OTHER zero-injection node but off by
    hundreds of thousands of kW at S35a/S76a/b/c's un-truncated second
    terminal specifically. Using every NONZERO node_order entry (0 =
    neutral/ground, never a real phase-node) handles both connection
    types with one rule, instead of special-casing Delta."""

    load_bus_phases = set()
    for name in dss.Loads.AllNames():
        dss.Circuit.SetActiveElement(f"Load.{name}")
        node_order = dss.CktElement.NodeOrder()
        bus_root = dss.CktElement.BusNames()[0].split(".")[0].lower()
        for phase_num in node_order:
            if phase_num != 0:
                load_bus_phases.add((bus_root, phase_num))
    return load_bus_phases


def enumerate_zero_injection_nodes(all_node_names):
    """Phase-nodes with no Load object attached at that (bus,phase),
    not the substation source bus, and not regulator-adjacent (see
    _regulator_buses). Requires a compiled circuit already active.

    BUG FIXED HERE (found in Phase 7, building on this function): the
    source bus (e.g. "150") has no Load object either -- a naive
    "no load => zero injection" check wrongly included it, even though
    the ENTIRE feeder's power enters the network there via the Vsource
    (P_injection there is the largest, not the smallest, on the whole
    feeder). Confirmed the consequence directly, not just reasoned
    about it: evaluating h_zi(x_true) at the real solved state showed
    ZI_150_*'s residual at ~58,700,000 kW (the whole feeder's power,
    essentially) against a median |h_zi(x_true)| of 5e-11 kW across the
    other 363 genuinely load-free nodes -- i.e. every OTHER zero-
    injection row was already numerically correct to float64 precision;
    only the source bus's 3 phase-nodes were wrong. This matters far
    more here than in Phase 4: Phase 4's rank/critical-measurement
    sweep never evaluates h_zi(x) away from flat start against a target
    of exactly 0 (rank is a structural property of the Jacobian, blind
    to what h0 numerically equals), so the bug was invisible there --
    Phase 7's Gauss-Newton solve, which DOES drive h_zi(x) toward 0 as
    a hard constraint, immediately and visibly breaks on it (the
    optimiser tries to force the source bus's injection to zero, which
    is physically impossible, and diverges).

    Regulator-adjacent nodes are excluded from THIS (hard-constraint)
    set for the tap-drift reason documented on _regulator_buses -- they
    are not discarded, just moved to enumerate_regulator_adjacent_nodes/
    build_regulator_pseudo_entries below, where they are used as
    WEIGHTED measurements instead."""

    load_bus_phases = _load_bus_phases()
    source_bus = _source_bus()
    regulator_buses = _regulator_buses()

    zero_injection = []
    for node_name in all_node_names:
        bus_root, phase_str = node_name.split(".")
        phase_num = int(phase_str)
        if bus_root.lower() == source_bus:
            continue
        if bus_root.lower() in regulator_buses:
            continue
        if (bus_root.lower(), phase_num) not in load_bus_phases:
            zero_injection.append(node_name)

    return zero_injection


def enumerate_regulator_adjacent_nodes(all_node_names):
    """The regulator-adjacent subset of what would otherwise qualify as
    a zero-injection node (no Load object, not the source bus) but is
    excluded from enumerate_zero_injection_nodes's HARD-constraint set
    -- see _regulator_buses's docstring for why. Returned separately
    rather than just dropped: build_regulator_pseudo_entries below turns
    these into WEIGHTED pseudo-measurements (target 0, empirically
    estimated sigma) instead of silently giving up the rank they
    represent."""

    load_bus_phases = _load_bus_phases()
    source_bus = _source_bus()
    regulator_buses = _regulator_buses()

    nodes = []
    for node_name in all_node_names:
        bus_root, phase_str = node_name.split(".")
        phase_num = int(phase_str)
        if bus_root.lower() == source_bus:
            continue
        if (bus_root.lower(), phase_num) in load_bus_phases:
            continue
        if bus_root.lower() in regulator_buses:
            nodes.append(node_name)

    return nodes


def build_zero_injection_entries(all_node_names):
    """One P_injection + one Q_injection 'entry' per zero-injection
    phase-node, in the SAME shape as a sensor-registry entry (id, node,
    phase, quantity), so jacobian.build_jacobian() and sensor_layer-
    style consumers can treat them identically. h() is not called for
    these -- their value is known EXACTLY as 0, by definition (that is
    what makes them free, exact constraints, not measurements with
    noise) -- callers should set z=0 directly, not measure()."""

    nodes = enumerate_zero_injection_nodes(all_node_names)

    entries = []
    for node_name in nodes:
        bus_root, phase_str = node_name.split(".")
        phase = PHASE_NUM_TO_LETTER[int(phase_str)]
        base_id = f"ZI_{bus_root}_{phase}"
        entries.append({
            "id": f"{base_id}_P", "node": bus_root, "phase": phase,
            "quantity": "P_injection", "sigma": 0.0, "enabled": True,
            "note": "zero-injection equality constraint (exact, Sha25 Sec II)",
        })
        entries.append({
            "id": f"{base_id}_Q", "node": bus_root, "phase": phase,
            "quantity": "Q_injection", "sigma": 0.0, "enabled": True,
            "note": "zero-injection equality constraint (exact, Sha25 Sec II)",
        })

    return entries


def build_regulator_pseudo_entries(all_node_names, stats_by_node):
    """WEIGHTED (not hard-constrained) entries for the regulator-
    adjacent nodes enumerate_regulator_adjacent_nodes identifies.
    stats_by_node: {node_name: {'mean_p','mean_q','sigma_p','sigma_q'}},
    from observability.estimator_model._estimate_regulator_mean_sigma --
    PER-NODE empirical mean AND sigma, not a hardcoded z_value=0.0.

    NOT the same target convention as a hard zero-injection constraint:
    direct inspection of the per-node ground-truth distribution (Phase
    7 finding) showed some of these nodes read a small but PERSISTENTLY
    NONZERO injection, not "0 plus occasional tap-drift noise" -- so
    the honest target for a WEIGHTED measurement here is whatever the
    held-out data says is typical for that specific node, exactly like
    a Phase-5 load pseudo-measurement, not an assumption that these are
    secretly zero-injection nodes with extra uncertainty bolted on.
    Kept in its own 'REGZI_' id namespace, distinct from both 'ZI_'
    (hard, exact-zero, zero-injection) and 'PSEUDO_' (unmetered LOAD
    nodes, Phase 5) -- three different reasons for uncertainty, three
    different code paths, per RES.md's explicit instruction not to
    conflate them."""

    nodes = enumerate_regulator_adjacent_nodes(all_node_names)

    entries = []
    for node_name in nodes:
        bus_root, phase_str = node_name.split(".")
        phase = PHASE_NUM_TO_LETTER[int(phase_str)]
        base_id = f"REGZI_{bus_root}_{phase}"
        stats = stats_by_node[node_name]
        entries.append({
            "id": f"{base_id}_P", "node": bus_root, "phase": phase,
            "quantity": "P_injection", "sigma": stats["sigma_p"], "z_value": stats["mean_p"],
            "note": "regulator-adjacent node, WEIGHTED not hard -- target "
                    "and sigma both estimated empirically from held-out "
                    "Phase 6 train-split residuals (Phase 7 finding: not "
                    "all of these read ~0, so the target is the data's own "
                    "mean, not an assumed 0)",
        })
        entries.append({
            "id": f"{base_id}_Q", "node": bus_root, "phase": phase,
            "quantity": "Q_injection", "sigma": stats["sigma_q"], "z_value": stats["mean_q"],
            "note": "regulator-adjacent node, WEIGHTED not hard -- see P entry note",
        })

    return entries
