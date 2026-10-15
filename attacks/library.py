"""
Attack library -- one interface for every FDI class used in Phase 8:

    attack(z_series, timestamps, spec, context=None) -> (z_attacked, labels)

z_series: DataFrame, one row per timestep, one column per telemetry
channel id (the registry ids, as in {split}_telemetry.csv). Only these
columns are ever changed -- an attack is z_delivered = z + a at Layer 2,
nothing else. timestamps: the matching real timestamps
('YYYY-MM-DD HH:MM:SS'; the hour selects the pseudo bucket where one is
needed). No attack reads x_true or any ground-truth column.

spec (AttackSpec): type, start (row position), duration (rows), profile
(step | ramp | pulse), targets, magnitude, seed, plus per-type fields.
  step  -- factor 1 on every row of the window
  ramp  -- factor (t - start + 1) / duration
  pulse -- factor 1 for the first PULSE_PERIOD // 2 rows of every
           PULSE_PERIOD-row block in the window, 0 otherwise
labels: one row per timestep -- attacked (1 where a != 0 on that row),
attack_type, target_ids (';'-joined channels with a != 0), plus
per-type diagnostics (n_capped for A1/A2; dP/total/loss changes for A3;
size for A4/A5).

Types:
  A1 random     -- Gaussian bias b_j = m * sigma_j * g_j (g ~ N(0,1)) on
                   k random real channels (k = spec.k, or all 165).
  A2 targeted   -- 1-3 channels (P, Q, V) of ONE meter, b_j = +/- m*sigma_j
                   (sign per channel from the seed).
  A3 load redistribution -- two metered loads, +dP on one and -dP on the
                   other (dP = magnitude * the smaller load's P), Q at
                   each load's own power factor. The attacker's power flow:
                   metered loads at their clean telemetry, unmetered at the
                   static pseudo values (observability/pf_start.py) -- then
                   the same with the shift, taps HELD at the clean solve's
                   positions (ControlMode=OFF). a = h(shifted) - h(clean)
                   from sensors.sensor_layer (noise-free), added to the
                   noisy z. access 'full': a on every telemetry channel;
                   'partial': only on the two loads' own meters (P, Q, V).
  A4 / A5       -- wrappers around attacks/a4_kept.py (kept-direction
                   linear A4 / Newton-corrected A5), size = magnitude pu,
                   re-synthesised at every attacked row from that row's
                   clean estimate, same rng seed every row.

Physical cap (A1, A2): |b_j| <= CAP_FRACTION * p95(|z_j|) over the TRAIN
split; logged per row as n_capped.

real channels = model.weighted_meas_entries (165: the registry minus the
3 P_head rows the estimator does not weight).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

from sensors.sensor_layer import build_measurement_dict

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_TELEMETRY = BASE_DIR / "simulation" / "phase6_corpus" / "train_telemetry.csv"
CAP_FRACTION = 0.5
PULSE_PERIOD = 10
ATTACK_TYPES = ("A1", "A2", "A3", "A4", "A5")
PROFILES = ("step", "ramp", "pulse")
_METER_RE = re.compile(r"^(M\d{3})_([PQV])(?:_([ABC]))?$")


@dataclass
class AttackSpec:
    type: str
    start: int
    duration: int
    profile: str = "step"
    targets: list = None      # A1/A2: channel ids; A3: [load_up, load_down]
    magnitude: float = 1.0    # A1/A2: m (x sigma); A3: dP fraction; A4/A5: size in pu
    seed: int = 0
    k: int = None             # A1: number of random channels (None = all real channels)
    meter: str = None         # A2: meter group, e.g. "M010" or "M043_B"
    quantities: tuple = ("P", "Q", "V")  # A2: which of the meter's channels
    access: str = "full"      # A3: "full" | "partial"


class AttackContext:
    """Shared resources, built lazily: the estimator model (one compiled
    feeder in this process), the estimator, sigmas, real-channel list,
    meter groups and the train-split caps."""

    def __init__(self, model=None, est=None, cap_fraction=CAP_FRACTION):
        self._model, self._est = model, est
        self.cap_fraction = cap_fraction
        self._caps = None

    @property
    def model(self):
        if self._model is None:
            from observability.estimator_model import build_estimator_model
            self._model = build_estimator_model("multiphase")
        return self._model

    @property
    def est(self):
        if self._est is None:
            from estimation.state_estimator import StateEstimator
            self._est = StateEstimator(self.model)
        return self._est

    @property
    def sigma(self):
        return {e["id"]: e["sigma"] for e in self.model.meas_entries}

    @property
    def real_ids(self):
        return [e["id"] for e in self.model.weighted_meas_entries]

    @property
    def meter_groups(self):
        groups = {}
        for cid in self.real_ids:
            mt = _METER_RE.match(cid)
            if mt:
                key = mt.group(1) if mt.group(3) is None else f"{mt.group(1)}_{mt.group(3)}"
                groups.setdefault(key, {})[mt.group(2)] = cid
        return groups

    @property
    def caps(self):
        if self._caps is None:
            cols = pd.read_csv(TRAIN_TELEMETRY, nrows=0).columns
            ids = [c for c in cols if c not in ("timestamp", "real_timestamp")]
            train = pd.read_csv(TRAIN_TELEMETRY, usecols=ids)
            self._caps = (self.cap_fraction * train.abs().quantile(0.95)).to_dict()
        return self._caps


def profile_factors(profile, duration):
    t = np.arange(duration)
    if profile == "step":
        return np.ones(duration)
    if profile == "ramp":
        return (t + 1) / duration
    if profile == "pulse":
        return ((t % PULSE_PERIOD) < PULSE_PERIOD // 2).astype(float)
    raise ValueError(f"unknown profile {profile!r}")


def _capped_bias(ids, raw, ctx):
    caps = ctx.caps
    b = {}
    n_capped = 0
    for cid, v in zip(ids, raw):
        cap = caps[cid]
        if abs(v) > cap:
            n_capped += 1
            v = np.sign(v) * cap
        b[cid] = float(v)
    return b, n_capped


def _a1_bias(spec, ctx, rng):
    real = ctx.real_ids
    if spec.targets:
        ids = list(spec.targets)
    else:
        k = len(real) if spec.k is None else spec.k
        ids = sorted(rng.choice(real, size=k, replace=False).tolist(), key=real.index)
    sig = ctx.sigma
    raw = [spec.magnitude * sig[c] * rng.standard_normal() for c in ids]
    return _capped_bias(ids, raw, ctx)


def _a2_bias(spec, ctx, rng):
    groups = ctx.meter_groups
    if spec.targets:
        ids = list(spec.targets)
    else:
        meter = spec.meter or sorted(groups)[int(rng.integers(len(groups)))]
        ids = [groups[meter][q] for q in spec.quantities if q in groups[meter]]
    sig = ctx.sigma
    raw = [spec.magnitude * sig[c] * (1.0 if rng.random() < 0.5 else -1.0) for c in ids]
    return _capped_bias(ids, raw, ctx)


def _metered_loads(model):
    loads = {}
    for e in model.meas_entries:
        if e["quantity"] in ("P_injection", "Q_injection") and "load_name" in e:
            d = loads.setdefault(e["load_name"], {"P": [], "Q": [], "phases": set()})
            d["P" if e["quantity"] == "P_injection" else "Q"].append(e["id"])
            d["phases"].add(e["phase"])
    return loads


def _pick_a3_loads(model, z_row, rng):
    """Two metered single-phase loads on the same phase (so per-phase
    feeder-head flows are unchanged up to losses), both with P > 0."""
    loads = _metered_loads(model)
    by_phase = {}
    for name, d in loads.items():
        if len(d["phases"]) == 1 and len(d["P"]) == 1 and z_row[d["P"][0]] > 0:
            by_phase.setdefault(next(iter(d["phases"])), []).append(name)
    phases = sorted(p for p, names in by_phase.items() if len(names) >= 2)
    phase = phases[int(rng.integers(len(phases)))]
    pair = rng.choice(sorted(by_phase[phase]), size=2, replace=False).tolist()
    return pair


def _a3_vector(spec, ctx, z_row, hour, factor, rng_pair):
    model = ctx.model
    loads = _metered_loads(model)
    up, down = list(spec.targets) if spec.targets else _pick_a3_loads(model, z_row, rng_pair)
    p = {n: float(sum(z_row[i] for i in loads[n]["P"])) for n in (up, down)}
    q = {n: float(sum(z_row[i] for i in loads[n]["Q"])) for n in (up, down)}
    dP = spec.magnitude * factor * min(p[up], p[down])

    model.pf_start(z_row, hour)  # attacker's clean power flow (taps reset, all 91 loads set)
    h_clean = build_measurement_dict(model.meas_entries, add_noise=False)
    loss_clean = np.array(dss.Circuit.Losses()) / 1000.0
    load_total_clean = _total_load()

    new_p = {up: p[up] + dP, down: p[down] - dP}
    for n in (up, down):
        dss.Loads.Name(n)
        dss.Loads.kW(new_p[n])
        dss.Loads.kvar(q[n] * new_p[n] / p[n])
    dss.Text.Command("Set ControlMode=OFF")
    try:
        dss.Solution.Solve()
        converged = bool(dss.Solution.Converged())
        h_shift = build_measurement_dict(model.meas_entries, add_noise=False)
        loss_shift = np.array(dss.Circuit.Losses()) / 1000.0
        load_total_shift = _total_load()
    finally:
        dss.Text.Command("Set ControlMode=STATIC")

    if spec.access == "full":
        ids = [e["id"] for e in model.meas_entries]
    elif spec.access == "partial":
        ids = [e["id"] for e in model.meas_entries if e.get("load_name") in (up, down)]
    else:
        raise ValueError(f"unknown access {spec.access!r}")
    a = {c: float(h_shift[c] - h_clean[c]) for c in ids}
    diag = {"a3_load_up": up, "a3_load_down": down, "a3_dP_kw": dP,
            "a3_dP_total_kw": float(load_total_shift[0] - load_total_clean[0]),
            "a3_dQ_total_kvar": float(load_total_shift[1] - load_total_clean[1]),
            "a3_dP_consumed_kw": float(load_total_shift[2] - load_total_clean[2]),
            "a3_dQ_consumed_kvar": float(load_total_shift[3] - load_total_clean[3]),
            "a3_dLoss_kw": float(loss_shift[0] - loss_clean[0]),
            "a3_dLoss_kvar": float(loss_shift[1] - loss_clean[1]),
            "a3_pf_converged": converged}
    return a, diag


def _total_load():
    """[set-point kW, set-point kvar, consumed kW, consumed kvar] summed
    over all loads. Set-point = the Load element's kW/kvar; consumed = its
    terminal power from the solved circuit (differs from the set point
    for this feeder's voltage-dependent load models)."""
    p = qv = pc = qc = 0.0
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        p += dss.Loads.kW()
        qv += dss.Loads.kvar()
        dss.Circuit.SetActiveElement(f"Load.{name}")
        pw = dss.CktElement.Powers()
        pc += sum(pw[0::2])
        qc += sum(pw[1::2])
    return np.array([p, qv, pc, qc])


def _a45_vector(spec, ctx, z_row, real_ts, factor):
    from attacks.a4_kept import kept_subspace_basis, synthesize_kept_attack, a5_from_c
    from estimation.bad_data_detection import unattackable_jacobian

    est, model = ctx.est, ctx.model
    clean = est.estimate(z_row, real_ts)
    basis = kept_subspace_basis(est, clean, real_ts)
    rng = np.random.default_rng(spec.seed)
    atk = synthesize_kept_attack(basis, rng, spec.magnitude * factor)
    a = atk["a"]
    if spec.type == "A5":
        unattack = unattackable_jacobian(est, clean, real_ts)["unattackable_entries"]
        a, _, _, _, _ = a5_from_c(model, model.state, clean, unattack, basis, atk["c"])
    return dict(zip(basis["meas_ids"], a.tolist())), {"size_pu": spec.magnitude * factor}


def attack(z_series, timestamps, spec, context=None):
    if spec.type not in ATTACK_TYPES:
        raise ValueError(f"unknown attack type {spec.type!r}")
    n = len(z_series)
    if spec.start < 0 or spec.duration < 1 or spec.start + spec.duration > n:
        raise ValueError("attack window outside the series")
    ctx = context or AttackContext()
    timestamps = list(timestamps)
    z_att = z_series.copy()
    factors = profile_factors(spec.profile, spec.duration)
    labels = pd.DataFrame({"timestamp": timestamps, "attacked": 0, "attack_type": "", "target_ids": ""},
                          index=z_series.index)

    rng = np.random.default_rng(spec.seed)
    bias = None
    if spec.type in ("A1", "A2"):
        bias, n_capped = (_a1_bias if spec.type == "A1" else _a2_bias)(spec, ctx, rng)

    for w, f in enumerate(factors):
        pos = spec.start + w
        if f == 0:
            continue
        z_row = z_series.iloc[pos].to_dict()
        diag = {}
        if bias is not None:
            a = {c: v * f for c, v in bias.items()}
            diag = {"n_capped": n_capped, "n_targets": len(bias)}
        elif spec.type == "A3":
            a, diag = _a3_vector(spec, ctx, z_row, int(str(timestamps[pos])[11:13]), f,
                                 np.random.default_rng(spec.seed))
        else:
            a, diag = _a45_vector(spec, ctx, z_row, str(timestamps[pos]), f)
        nz = [c for c, v in a.items() if v != 0.0]
        for c in nz:
            z_att.iat[pos, z_att.columns.get_loc(c)] = z_row[c] + a[c]
        idx = z_series.index[pos]
        labels.loc[idx, ["attacked", "attack_type", "target_ids"]] = [int(bool(nz)), spec.type, ";".join(nz)]
        for kk, v in diag.items():
            if kk not in labels.columns:
                labels[kk] = None
            labels.at[idx, kk] = v
    return z_att, labels
