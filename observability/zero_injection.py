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


def _capacitor_buses():
    """RES.md D10 Phase 7 CONSTRAINT-FEASIBILITY FIX (Block 6.1,
    permanent as of the HIGH-LOAD INVESTIGATION task): every bus a
    Capacitor element is attached to -- e.g. c83 (600 kvar, 3-phase)
    on this feeder. GENUINE OMISSION, not a near-zero-impedance-branch
    case like _regulator_buses/_switch_buses: a shunt capacitor
    injects a KNOWN, nonzero reactive power, exactly like a Load
    injects known nonzero real/reactive power -- but _load_bus_
    phases() only ever checked dss.Loads.AllNames(), never dss.
    Capacitors.AllNames(), so a capacitor-connected phase with no Load
    object was wrongly treated as genuinely zero-injection.

    Confirmed via direct evidence (validation/
    phase7_pure_feasibility_check.py, reports/
    phase7_constraint_feasibility_report.txt Steps 1-3): minimising
    ||c(x)||^2 alone (no measurement objective at all) from flat start
    gets stuck at max|c|=24.4 kvar, concentrated at bus 83's rows plus
    a LEAST-SQUARES SPREADING of that same irreducible residual onto
    electrically-adjacent buses on the same radial lateral
    (78->80->81->82->83) and the switch-adjacent buses 60/97/197 --
    EXCLUDING ONLY bus 83's rows drops max|c| to 0.028 kvar, i.e.
    every other row becomes near-exactly feasible, proving those were
    spreading artifacts of this ONE root cause, not independent
    infeasibilities.

    Not a wrong TARGET (h_zi(x_true)=0 exactly at bus 83, reports/
    phase7_state_estimator_report.txt Part D.2 -- the capacitor's
    admittance is already folded into the Y-bus) -- the difficulty is
    bus 83's "strongly nonlinear (quadratic-in-V) self-susceptance
    term" making this row hard for a local linearisation-based solve
    to navigate from a distant start. Demoted via the IDENTICAL
    existing regulator-demotion pattern, preserving rank/dof rather
    than dropping the rows.

    MADE PERMANENT here (reports/phase7_constraint_feasibility_report.
    txt first tried and reverted this exact change, because it
    regressed estimation/state_estimator.py's hand-rolled solver,
    which has its OWN separate, already-documented construction defect
    -- reports/phase7_solver_instability_fix_report.txt). This task
    (reports/phase7_highload_investigation_report.txt) re-applies it
    permanently because the model this function feeds is now paired
    with estimation/state_estimator_scipy.py (an independent solve
    path) going forward, not the hand-rolled one -- that hand-rolled
    solver remains broken and is NOT re-tested or re-validated against
    this change."""

    buses = set()
    for cap in dss.Capacitors.AllNames():
        dss.Circuit.SetActiveElement(f"Capacitor.{cap}")
        for b in dss.CktElement.BusNames():
            buses.add(b.split(".")[0].lower())
    return buses


def _switch_buses():
    """RES.md D10 Phase 7 HIGH-LOAD INVESTIGATION (Block 7.1): every
    bus touching a near-zero-impedance SWITCH line (this feeder's
    sw1-sw8, e.g. 13/152, 18/135, 61/61s, 97/197, 151/300_open,
    54/94_open -- confirmed directly, dss.Lines.IsSwitch() returns
    False for all lines here, so identified via R1=0.001/X1=0.0/
    Length=0.001, the SAME near-zero-impedance signature
    _regulator_buses's docstring already describes, and this
    circuit's own 'sw<N>' naming convention).

    First tried, at FLAT START only, in reports/
    phase7_node_demotion_test_report.txt -- found to regress the HAND-
    ROLLED solver (estimation/state_estimator.py) across all 6 pilot
    points, and reverted. Re-introduced here because reports/
    phase7_highload_investigation_report.txt's Step 2 found, by
    decomposing the residual AT THE ACTUAL HIGH-LOAD SOLVE LANDING
    POINT (not flat start, where the bus-83 capacitor fix's own
    diagnostic ran) using estimation.state_estimator_scipy's
    independent trust-constr path, that essentially ALL remaining
    switch-pair buses (sw2/3/5/6/7/8 -- sw1 and sw4's buses are
    already regulator-adjacent via 150r/160) show large, near-exactly
    opposite-sign, near-equal-magnitude residuals at high load,
    confirming the SAME category of near-zero-impedance-branch
    fragility the flat-start-only check could not see (the WLS
    objective pulls a high-load solve far enough from flat start for
    these specific branches' voltage mismatch to become significant,
    where it was not at flat start). Tested THIS TIME against the
    scipy solve path (not the hand-rolled one, which remains broken
    for separate, already-documented reasons) and WITH the bus-83
    capacitor fix already in place -- a genuinely different
    experiment from the one reverted in phase7_node_demotion_test_
    report.txt, not a blind repeat."""

    buses = set()
    for name in dss.Lines.AllNames():
        if not name.lower().startswith("sw"):
            continue
        dss.Circuit.SetActiveElement(f"Line.{name}")
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
    regulator_buses = _regulator_buses() | _capacitor_buses() | _switch_buses()

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
    regulator_buses = _regulator_buses() | _capacitor_buses() | _switch_buses()

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
