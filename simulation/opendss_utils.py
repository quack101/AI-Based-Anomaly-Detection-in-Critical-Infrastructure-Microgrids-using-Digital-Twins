from pathlib import Path
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

def get_load_telemetry(load_name):
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

