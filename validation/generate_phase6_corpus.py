"""
RES.md D10 Phase 6 -- generate the normal-operation (unattacked) corpus:
clean, labelled telemetry across the full 60-day IDEAL corpus window,
split by TIME (never randomly -- temporal leakage would invalidate
every downstream ML result, per RES.md's own Musleh/Irfan basis),
with the full x_true ground-truth state persisted per timestamp for
Phase 11's later impact quantification.

DEPLOYMENT DECISION (RES.md D10 Phase 6's own framing): the multiphase
registry variant now has its Phase-5-identified minimum sensor set
enabled (106 meter P,Q + 6 substation head P,Q = 112 entries;
config/sensor_registry_multiphase.json) -- config/sensor_registry_
multiphase.json's "enabled" flags were flipped for exactly the 6
HEAD_P/Q_A/B/C entries as this phase's first action, so the corpus this
script generates matches the sensor configuration Phase 5 actually
validated (rank=553/553, zero critical measurements), not an
undeployed one. Zero-injection (366) and pseudo-measurements (84) are
NOT registry entries (RES.md D2/D4: they are estimator input, not
sensor readings) and are not part of telemetry.raw -- they remain
Phase 7's estimator-construction concern, built from
observability.zero_injection / observability.pseudo_measurements at
estimation time, not generated into this corpus.

Layer boundary: this script is the L1 ground-truth generator + L2
sensor layer, exactly like streaming/producer.py's Producer class (RES.md
D2/D6.0 permit this). It does NOT reuse Producer directly -- Producer's
per-timestep method builds ONE Kafka message per REPRESENTATIVE METER
(load_name-keyed) and has no path for head entries (which have no
load_name) without changing the live wire protocol's message schema
and streaming/consumer.py's meter-count-based window-completion logic,
which is out of this phase's scope (a live-pipeline wiring change, not
a corpus-generation one). This script instead writes a flat, wide
per-timestamp record (all 112 registry ids as columns) directly to
files -- the right shape for downstream training/evaluation, and
exactly what "telemetry.raw's content" means for a corpus, without
requiring a live Kafka broker (none is available in this environment,
consistent with every other phase's validation).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import opendssdirect as dss
import pandas as pd

from simulation.opendss_utils import compile_feeder, get_solution_metrics
from sensors.registry import load_registry, enabled_entries
from sensors.sensor_layer import build_measurement_dict
from observability.state_vector import build_state_vector, extract_solved_state

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "node_loads_91"
REPRESENTATIVE_METERS = BASE_DIR / "simulation" / "representative_meters.json"
CORPUS_DIR = BASE_DIR / "simulation" / "phase6_corpus"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
RNG_SEED = 42  # one continuous stream across the whole 60-day run,
                # matching how a real telemetry stream is one continuous
                # process, not three independently-seeded ones per split

GROUND_TRUTH_FIELDS = [
    "timestamp", "real_timestamp", "converged", "iterations",
    "v_pu_min", "v_pu_max", "losses_kw", "losses_kvar",
    "feeder_head_p_kw", "feeder_head_q_kvar", "solve_time_s",
]
LABEL_FIELDS = ["timestamp", "real_timestamp", "attacked_meters", "attack_class", "magnitude", "stealth_flag"]


def load_inputs():
    with open(REPRESENTATIVE_METERS) as f:
        representative_meters = json.load(f)

    available = {f.stem[len("node_"):].lower(): f for f in DATASET_DIR.glob("node_*.csv")}
    missing = [m["load_name"] for m in representative_meters if m["load_name"].lower() not in available]
    if missing:
        raise ValueError(f"{len(missing)} representative meter(s) have no corpus file: {missing}")

    datasets = [pd.read_csv(available[m["load_name"].lower()]) for m in representative_meters]
    return representative_meters, datasets


def compute_splits(real_timestamps):
    """Time-based, contiguous split by CALENDAR DAY -- never a random
    shuffle. Returns [("train", lo, hi), ("val", lo, hi), ("test", lo, hi)]
    as half-open row-index ranges, and the exact date boundaries."""

    # pd.DatetimeIndex(...).date works uniformly whether real_timestamps
    # arrives as a plain Series of strings (the real corpus's own
    # "timestamp" column) or an already-parsed DatetimeIndex (as in
    # tests) -- pd.to_datetime(x).dt.date only works for the former,
    # since .dt is a Series accessor and pd.to_datetime() on an
    # already-DatetimeIndex input returns a DatetimeIndex, not a Series.
    dates = pd.DatetimeIndex(pd.to_datetime(real_timestamps)).date
    unique_days = sorted(set(dates))
    n_days = len(unique_days)

    n_train_days = round(n_days * TRAIN_FRAC)
    n_val_days = round(n_days * VAL_FRAC)
    n_test_days = n_days - n_train_days - n_val_days  # remainder, absorbs rounding

    train_days = unique_days[:n_train_days]
    val_days = unique_days[n_train_days:n_train_days + n_val_days]
    test_days = unique_days[n_train_days + n_val_days:]

    day_to_split = {}
    for d in train_days:
        day_to_split[d] = "train"
    for d in val_days:
        day_to_split[d] = "val"
    for d in test_days:
        day_to_split[d] = "test"

    split_of_row = np.array([day_to_split[d] for d in dates])

    boundaries = {
        "train": (str(train_days[0]), str(train_days[-1]), len(train_days)),
        "val": (str(val_days[0]), str(val_days[-1]), len(val_days)),
        "test": (str(test_days[0]), str(test_days[-1]), len(test_days)),
    }
    return split_of_row, boundaries


def main(add_noise=True, seed=RNG_SEED):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    representative_meters, datasets = load_inputs()
    num_rows = min(len(df) for df in datasets)

    real_timestamps = datasets[0]["timestamp"].iloc[:num_rows]
    split_of_row, boundaries = compute_splits(real_timestamps)

    print("Split boundaries (calendar days, inclusive):")
    for name, (lo, hi, n_days) in boundaries.items():
        print(f"  {name}: {lo} .. {hi}  ({n_days} days)")

    compile_feeder()
    state = build_state_vector()

    registry = load_registry(variant="multiphase")
    entries = enabled_entries(registry)
    print(f"Registry: {len(entries)} enabled entries (multiphase variant, "
          f"Phase 5/6 minimum-set deployment)")

    rng = random.Random(seed) if add_noise else None

    entry_ids = [e["id"] for e in entries]

    split_names = ["train", "val", "test"]
    split_row_counts = {s: int((split_of_row == s).sum()) for s in split_names}
    print("Row counts per split:", split_row_counts)

    x_v_buffers = {s: np.zeros((split_row_counts[s], state.n_phase_nodes)) for s in split_names}
    x_delta_buffers = {s: np.zeros((split_row_counts[s], len(state.delta_index))) for s in split_names}
    split_cursor = {s: 0 for s in split_names}

    telemetry_files = {
        s: open(CORPUS_DIR / f"{s}_telemetry.csv", "w", newline="") for s in split_names
    }
    ground_truth_files = {
        s: open(CORPUS_DIR / f"{s}_ground_truth.csv", "w", newline="") for s in split_names
    }
    label_files = {
        s: open(CORPUS_DIR / f"{s}_labels.csv", "w", newline="") for s in split_names
    }

    import csv
    telemetry_writers = {
        s: csv.DictWriter(telemetry_files[s], fieldnames=["timestamp", "real_timestamp"] + entry_ids)
        for s in split_names
    }
    ground_truth_writers = {
        s: csv.DictWriter(ground_truth_files[s], fieldnames=GROUND_TRUTH_FIELDS) for s in split_names
    }
    label_writers = {
        s: csv.DictWriter(label_files[s], fieldnames=LABEL_FIELDS) for s in split_names
    }
    for s in split_names:
        telemetry_writers[s].writeheader()
        ground_truth_writers[s].writeheader()
        label_writers[s].writeheader()

    t0 = time.time()
    convergence_failures = []

    for t in range(num_rows):
        real_ts = None
        for i, (meter, df) in enumerate(zip(representative_meters, datasets)):
            row = df.iloc[t]
            if i == 0:
                real_ts = str(row["timestamp"])
            dss.Loads.Name(meter["load_name"])
            dss.Loads.kW(float(row["P_kW"]))
            dss.Loads.kvar(float(row["Q_kvar"]))

        solve_start = time.perf_counter()
        dss.Solution.Solve()
        solve_time_s = time.perf_counter() - solve_start

        metrics = get_solution_metrics(solve_time_s)
        if not metrics["converged"]:
            convergence_failures.append(t)

        x_v, x_delta = extract_solved_state(state)
        z_dict = build_measurement_dict(entries, add_noise=add_noise, rng=rng)

        split = split_of_row[t]
        i_row = split_cursor[split]
        x_v_buffers[split][i_row, :] = x_v
        x_delta_buffers[split][i_row, :] = x_delta
        split_cursor[split] += 1

        telemetry_row = {"timestamp": t, "real_timestamp": real_ts}
        telemetry_row.update(z_dict)
        telemetry_writers[split].writerow(telemetry_row)

        per_phase = metrics["v_pu_per_phase"]
        ground_truth_writers[split].writerow({
            "timestamp": t, "real_timestamp": real_ts,
            "converged": metrics["converged"], "iterations": metrics["iterations"],
            "v_pu_min": min(p["min"] for p in per_phase.values()),
            "v_pu_max": max(p["max"] for p in per_phase.values()),
            "losses_kw": metrics["losses_kw"], "losses_kvar": metrics["losses_kvar"],
            "feeder_head_p_kw": metrics["feeder_head_p_kw"],
            "feeder_head_q_kvar": metrics["feeder_head_q_kvar"],
            "solve_time_s": metrics["solve_time_s"],
        })

        label_writers[split].writerow({
            "timestamp": t, "real_timestamp": real_ts,
            "attacked_meters": "[]", "attack_class": "", "magnitude": 0.0, "stealth_flag": False,
        })

        if t % 10000 == 0:
            print(f"  t={t}  split={split}  converged={metrics['converged']}  "
                  f"elapsed={time.time()-t0:.1f}s")

    for s in split_names:
        telemetry_files[s].close()
        ground_truth_files[s].close()
        label_files[s].close()
        np.savez_compressed(
            CORPUS_DIR / f"{s}_x_true.npz",
            x_v=x_v_buffers[s], x_delta=x_delta_buffers[s],
            v_index=np.array(list(state.v_index.items()), dtype=object),
            delta_index=np.array(list(state.delta_index.items()), dtype=object),
        )

    print(f"Done in {time.time()-t0:.1f}s. Convergence failures: {len(convergence_failures)} "
          f"of {num_rows} ({convergence_failures[:10]})")

    return {
        "num_rows": num_rows, "split_row_counts": split_row_counts,
        "boundaries": boundaries, "convergence_failures": convergence_failures,
        "entry_ids": entry_ids,
    }


if __name__ == "__main__":
    main()
