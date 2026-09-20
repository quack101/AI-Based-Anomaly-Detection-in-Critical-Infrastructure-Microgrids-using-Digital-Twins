"""
RES.md D10 Phase 5 -- pseudo-measurements at the 42 unmetered LOAD
nodes (D4 Sec 4.3 row 1). DISTINCT from Phase 4's zero-injection
constraints (observability/zero_injection.py): zero-injection nodes
carry NO load and are known to be exactly 0; pseudo-measurement nodes
have REAL, just-unmeasured consumption, estimated from historical data
with an honest, nonzero uncertainty. Conflating the two is a modelling
error RES.md explicitly names -- kept as two separate modules, two
separate id namespaces ("ZI_*" vs "PSEUDO_*"), and this module never
touches a load-free node.

sigma_pseudo -- ESTIMATED, not asserted (RES.md's own instruction: "A
fixed hand-picked sigma_pseudo is the first thing a reviewer familiar
with this line will attack"). Method: empirical per-node, per-time-
bucket variance from HELD-OUT historical data --
  1. Split each unmetered load's IDEAL-corpus series (dataset/
     node_loads_91/, 60 days, 1-min) into a TRAIN prefix and a HELD-OUT
     suffix (75/25 by default -- 45 train days, 15 held-out days).
  2. Bucket by hour-of-day (24 buckets/node) -- captures the dominant
     diurnal load-shape variation without needing a richer model.
  3. The pseudo-measurement VALUE for a given hour is the TRAIN-bucket
     mean P,Q (this is what a static, offline generator would predict).
  4. sigma_pseudo for that (node, hour) is the sample std of the HELD-
     OUT bucket's actual values around the TRAIN-derived mean -- an
     honest OUT-OF-SAMPLE uncertainty estimate, not the (optimistic)
     in-sample residual.
This is the "minimum acceptable" method RES.md names -- costs almost
nothing, does not require adopting a learned generator (Ald25's
WaveNet-LSTM + Monte Carlo dropout), and is directly reproducible from
data already in the repo.

SECURITY PROPERTY (RES.md's own framing, stated as a deliberate design
choice here, not an accuracy shortfall): this generator is STATIC and
OFFLINE-COMPUTED -- it conditions ONLY on historical data fixed at
generation time, never on live measurements. Ald25's own generator
consumes live main-branch-flow currents as an input feature; if an
attacker manipulates those upstream measurements, the generator
launders attacker-influenced signal into what the estimator treats as
an INDEPENDENT pseudo-measurement -- a circularity channel. A pseudo-
measurement that never re-conditions on live data at inference is
structurally immune to that channel: nothing an attacker does to
Layer-2 telemetry at run time can change what this module predicts for
a given (node, hour), because the prediction was computed once, offline,
before any live measurement (attacked or not) existed. Residual
exposure, acknowledged rather than hidden: STALE-MODEL DRIFT -- the
historical shape can become a worse predictor as consumption patterns
change over time, which would show up as sigma_pseudo being too
optimistic relative to the true current spread, not as a live-data
attack surface.
"""

import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"

TRAIN_FRACTION = 0.75  # 45 of 60 days train, 15 held out
SIGMA_FLOOR = 1e-6  # avoid a degenerate sigma=0 at a bucket with zero held-out spread


def _corpus_path(load_name):
    available = {
        f.stem[len("node_"):].lower(): f
        for f in DATASET_DIR.glob("node_*.csv")
    }
    path = available.get(load_name.lower())
    if path is None:
        raise FileNotFoundError(f"no corpus file for {load_name!r} in {DATASET_DIR}")
    return path


def estimate_sigma_pseudo(load_name, train_fraction=TRAIN_FRACTION):
    """Returns {hour: {'mean_p','mean_q','sigma_p','sigma_q','n_train','n_heldout'}}
    for one unmetered load, per Sec 1-4 of the module docstring."""

    df = pd.read_csv(_corpus_path(load_name))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx]
    held_out = df.iloc[split_idx:]

    result = {}
    for hour in range(24):
        train_bucket = train[train["hour"] == hour]
        held_bucket = held_out[held_out["hour"] == hour]

        mean_p = float(train_bucket["P_kW"].mean())
        mean_q = float(train_bucket["Q_kvar"].mean())

        if len(held_bucket) > 1:
            sigma_p = float(((held_bucket["P_kW"] - mean_p) ** 2).mean() ** 0.5)
            sigma_q = float(((held_bucket["Q_kvar"] - mean_q) ** 2).mean() ** 0.5)
        else:
            sigma_p = sigma_q = float("nan")

        result[hour] = {
            "mean_p": mean_p, "mean_q": mean_q,
            "sigma_p": max(sigma_p, SIGMA_FLOOR) if not math.isnan(sigma_p) else None,
            "sigma_q": max(sigma_q, SIGMA_FLOOR) if not math.isnan(sigma_q) else None,
            "n_train": int(len(train_bucket)), "n_heldout": int(len(held_bucket)),
        }

    return result


def summarize_sigma_pseudo(per_hour):
    """One representative (mean_p, mean_q, sigma_p, sigma_q) per node --
    the mean across the 24 hour-buckets. Full per-hour detail remains
    available (estimate_sigma_pseudo's return value) for a future
    estimator that wants finer time resolution than this summary."""

    valid = [v for v in per_hour.values() if v["sigma_p"] is not None]
    n = len(valid)
    return {
        "mean_p": sum(v["mean_p"] for v in valid) / n,
        "mean_q": sum(v["mean_q"] for v in valid) / n,
        "sigma_p": sum(v["sigma_p"] for v in valid) / n,
        "sigma_q": sum(v["sigma_q"] for v in valid) / n,
        "n_hours_estimated": n,
    }


def build_pseudo_entries_for_hour(bus_phase_by_load, per_hour_by_load, hour):
    """RES.md D10 Phase 7 -- the estimator's per-timestamp pseudo-
    measurement build: given PRE-COMPUTED per-hour tables (one
    estimate_sigma_pseudo() call per load, done ONCE at estimator
    construction -- this function does no file I/O and touches no
    corpus data itself, so it is cheap to call every timestamp) and the
    HOUR the timestamp being estimated falls in, returns entries
    carrying BOTH the pseudo-measurement's static predicted VALUE
    (z_value: that hour's train-bucket mean -- there is no live
    measure() path for a PSEUDO_ id, unlike a real registry entry, so
    the value must travel with the entry rather than being computed by
    sensors.measurement_functions) and its sigma for that hour. Still
    static/offline per the module's own security property -- 'hour'
    only selects which PRE-COMPUTED bucket to read, it is not a live
    signal the generator conditions on."""

    entries = []
    for load_name, (bus, phase) in bus_phase_by_load.items():
        bucket = per_hour_by_load[load_name][hour]
        base_id = f"PSEUDO_{load_name}"
        entries.append({
            "id": f"{base_id}_P", "node": bus, "phase": phase,
            "quantity": "P_injection", "load_name": load_name,
            "sigma": bucket["sigma_p"], "z_value": bucket["mean_p"],
            "note": "pseudo-measurement, unmetered load (D4 4.3 row 1); "
                    "static hour-of-day bucket mean/sigma",
        })
        entries.append({
            "id": f"{base_id}_Q", "node": bus, "phase": phase,
            "quantity": "Q_injection", "load_name": load_name,
            "sigma": bucket["sigma_q"], "z_value": bucket["mean_q"],
            "note": "pseudo-measurement, unmetered load (D4 4.3 row 1); "
                    "static hour-of-day bucket mean/sigma",
        })
    return entries


def build_pseudo_entries_with_estimated_sigma(unmetered_loads, resolve_bus_and_phase):
    """unmetered_loads: list of load names. resolve_bus_and_phase(name)
    -> (bus, phase), e.g. observability.sweep's own resolver (kept as a
    caller-supplied callback so this module never has to import
    opendssdirect / touch a live circuit itself -- pure data-processing
    over the historical corpus, consistent with being an OFFLINE,
    static generator; see module docstring's security property).

    Returns (entries, summaries) -- entries in the same shape as a
    registry entry (P_injection/Q_injection, PSEUDO_ id prefix,
    load_name), sigma now the ESTIMATED value (Sec summarize above)
    rather than a flat SIGMA_FRAC_PSEUDO placeholder."""

    entries = []
    summaries = {}

    for load_name in unmetered_loads:
        bus, phase = resolve_bus_and_phase(load_name)
        per_hour = estimate_sigma_pseudo(load_name)
        summary = summarize_sigma_pseudo(per_hour)
        summaries[load_name] = {"per_hour": per_hour, "summary": summary}

        base_id = f"PSEUDO_{load_name}"
        entries.append({
            "id": f"{base_id}_P", "node": bus, "phase": phase,
            "quantity": "P_injection", "load_name": load_name,
            "sigma": summary["sigma_p"],
            "note": "pseudo-measurement, unmetered load (D4 4.3 row 1); "
                    "sigma estimated from held-out IDEAL-corpus variance",
        })
        entries.append({
            "id": f"{base_id}_Q", "node": bus, "phase": phase,
            "quantity": "Q_injection", "load_name": load_name,
            "sigma": summary["sigma_q"],
            "note": "pseudo-measurement, unmetered load (D4 4.3 row 1); "
                    "sigma estimated from held-out IDEAL-corpus variance",
        })

    return entries, summaries
