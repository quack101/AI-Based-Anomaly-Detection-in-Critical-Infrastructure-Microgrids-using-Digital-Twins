from pathlib import Path
from collections import defaultdict
import opendssdirect as dss

BASE_DIR = Path(__file__).resolve().parent.parent

MASTER_DSS = (
    BASE_DIR /
    "123Bus" /
    "IEEE123Master.dss"
)

def compile_feeder():

    dss.Text.Command(f'Compile "{MASTER_DSS}"')

    dss.Solution.Solve()

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS solution did not converge.")

    print("IEEE-123 feeder compiled.")

def get_load_telemetry(load_name, debug=False):
    dss.Loads.Name(load_name)

    powers = dss.CktElement.Powers()
    voltages = dss.CktElement.VoltagesMagAng()
    currents = dss.CktElement.CurrentsMagAng()

    # Total real/reactive power
    p_kw = sum(powers[0::2])
    q_kvar = sum(powers[1::2])

    # Average voltage magnitude
    voltage_mag = voltages[0::2]

    voltage = sum(voltage_mag) / len(voltage_mag)

    # Average current magnitude
    current_mag = currents[0::2]

    current = sum(current_mag) / len(current_mag)
    #debug
    bus = dss.CktElement.BusNames()[0].split(".")[0]
    dss.Circuit.SetActiveBus(bus)
    pu_voltage = dss.Bus.puVmagAngle()[0]
    dss.Loads.Name(load_name)
    if debug:
        print("Bus:", bus)
        print("Base kV:", dss.Bus.kVBase())
        print("PU:", dss.Bus.puVmagAngle())
    return {
        "voltage_v": voltage,
        "voltage_pu": pu_voltage,
        "current_a": current,
        "p_kw": p_kw,
        "q_kvar": q_kvar,
    }


def get_circuit_topology():
    """
    Extract once at startup: energised phase-node names and their count.

    dss.Circuit.AllNodeNames() enumerates exactly the phase-nodes present
    in the solved circuit (bus.phase), which is N_phase_nodes per RES.md
    D4 -- not the bus count (dss.Circuit.NumBuses()).
    """
    all_node_names = dss.Circuit.AllNodeNames()
    n_phase_nodes = len(all_node_names)

    return n_phase_nodes, all_node_names


def get_solution_metrics(solve_time_s):
    """
    Per-timestamp solve diagnostics for Phase-0 validation:
    convergence, iteration count, V_pu min/max per phase across all
    energised nodes, total system losses, feeder-head P/Q, solve wall-time.
    """
    converged = dss.Solution.Converged()
    iterations = dss.Solution.Iterations()

    node_names = dss.Circuit.AllNodeNames()
    v_pu = dss.Circuit.AllBusMagPu()

    v_pu_by_phase = defaultdict(list)
    for name, v in zip(node_names, v_pu):
        phase = name.split(".")[-1]
        v_pu_by_phase[phase].append(v)

    v_pu_per_phase = {
        phase: {"min": min(values), "max": max(values)}
        for phase, values in v_pu_by_phase.items()
    }

    losses_w, losses_var = dss.Circuit.Losses()
    losses_kw = losses_w / 1000.0
    losses_kvar = losses_var / 1000.0

    # TotalPower() is the power delivered TO the circuit by sources,
    # reported with generator sign convention (negative = delivered).
    head_p_kw, head_q_kvar = dss.Circuit.TotalPower()
    feeder_head_p_kw = -head_p_kw
    feeder_head_q_kvar = -head_q_kvar

    return {
        "converged": converged,
        "iterations": iterations,
        "v_pu_per_phase": v_pu_per_phase,
        "losses_kw": losses_kw,
        "losses_kvar": losses_kvar,
        "feeder_head_p_kw": feeder_head_p_kw,
        "feeder_head_q_kvar": feeder_head_q_kvar,
        "solve_time_s": solve_time_s,
    }

