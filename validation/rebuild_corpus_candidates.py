"""
Step 1 of the A-vs-B task: regenerate the set-A Phase-6 corpus and find
which registry version produced the ORIGINAL corpus, by comparing
SHA256 hashes against reports/corpus_hashes_A.csv.

Why candidates: the corpus noise is value += rng.gauss(0.0, sigma) per
enabled registry entry (sensors/sensor_layer.py), and the multiphase
registry changed three times on 2026-10-10 (0366af8 enabled 56 V
channels and widened M042_Q; 2dd08f7 left multiphase sigmas unchanged;
81ff01a recalibrated 103 sigmas). Which one was active when the
original corpus was generated is not recorded.

How one pass covers all candidates exactly: CPython's
random.gauss(mu, sigma) returns mu + z*sigma where z depends only on
the generator state, never on sigma. So for candidates sharing the same
enabled-entry ORDER (all 168-entry ones), one Random(42) stream drawn as
gauss(0.0, 1.0) gives z, and value + (0.0 + z*sigma_c) is bit-identical
to what validation/generate_phase6_corpus.py would write with sigma_c.
The 112-entry candidate consumes the stream differently, so it gets its
own Random(42). Everything else mirrors generate_phase6_corpus.main()
line for line (same loop, same load updates for the 49 set-A meters,
same CSV writers, same ground-truth/label rows).

Candidates (multiphase registry):
  T112        ae0fcfb file as-is            112 enabled, original sigmas
  T168_orig   ae0fcfb sigmas, 0366af8 flags 168 enabled, pre-10-Oct sigmas
  T168_m042   0366af8 file (== 2dd08f7)     168 enabled, M042_Q widened
  T168_recal  81ff01a file (current)        168 enabled, recalibrated

Writes (never into simulation/phase6_corpus/):
  simulation/phase6_corpus_Arebuilt/{split}_{ground_truth,labels}.csv,
  {split}_x_true.npz, and _candidates/<name>/{split}_telemetry.csv

Run: python -m validation.rebuild_corpus_candidates [--max-rows N]
"""

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import opendssdirect as dss

from sensors.measurement_functions import measure
from observability.state_vector import build_state_vector, extract_solved_state
from validation.generate_phase6_corpus import (
    GROUND_TRUTH_FIELDS, LABEL_FIELDS, RNG_SEED, compute_splits, load_inputs,
)
from simulation.opendss_utils import compile_feeder, get_solution_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "simulation" / "phase6_corpus_Arebuilt"
CAND_DIR = OUT_DIR / "_candidates"
SPLITS = ["train", "val", "test"]


def _registry_at(commit):
    text = subprocess.run(["git", "show", f"{commit}:config/sensor_registry_multiphase.json"],
                          cwd=BASE_DIR, capture_output=True, text=True, check=True).stdout
    return json.loads(text)


def build_candidates():
    r_ae, r_0366, r_81 = _registry_at("ae0fcfb"), _registry_at("0366af8"), _registry_at("81ff01a")
    sig_ae = {e["id"]: e["sigma"] for e in r_ae}
    en168 = [e for e in r_0366 if e["enabled"]]
    en112 = [e for e in r_ae if e["enabled"]]
    sig = lambda reg: {e["id"]: e["sigma"] for e in reg}
    return {
        "T112": (en112, sig_ae),
        "T168_orig": (en168, sig_ae),
        "T168_m042": (en168, sig(r_0366)),
        "T168_recal": (en168, sig(r_81)),
    }


def main(max_rows=None):
    assert OUT_DIR.name != "phase6_corpus"
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    representative_meters, datasets = load_inputs()  # set A meters (representative_meters.json)
    num_rows = min(len(df) for df in datasets)
    if max_rows:
        num_rows = min(num_rows, max_rows)
    real_timestamps = datasets[0]["timestamp"].iloc[:min(len(df) for df in datasets)]
    split_of_row, boundaries = compute_splits(real_timestamps)

    compile_feeder()
    state = build_state_vector()

    cands = build_candidates()
    en168 = cands["T168_recal"][0]
    en112 = cands["T112"][0]
    union = {e["id"]: e for e in en168}
    for e in en112:
        union.setdefault(e["id"], e)
    print("candidates:", {k: len(v[0]) for k, v in cands.items()}, "| union entries:", len(union))

    rng168, rng112 = random.Random(RNG_SEED), random.Random(RNG_SEED)

    split_row_counts = {s: int((split_of_row[:num_rows] == s).sum()) for s in SPLITS}
    x_v_buf = {s: np.zeros((split_row_counts[s], state.n_phase_nodes)) for s in SPLITS}
    x_d_buf = {s: np.zeros((split_row_counts[s], len(state.delta_index))) for s in SPLITS}
    cursor = {s: 0 for s in SPLITS}

    files, tw = [], {}
    for name, (entries, _) in cands.items():
        (CAND_DIR / name).mkdir(exist_ok=True)
        for s in SPLITS:
            f = open(CAND_DIR / name / f"{s}_telemetry.csv", "w", newline="")
            files.append(f)
            w = csv.DictWriter(f, fieldnames=["timestamp", "real_timestamp"] + [e["id"] for e in entries])
            w.writeheader()
            tw[(name, s)] = w
    gw, lw = {}, {}
    for s in SPLITS:
        fg = open(OUT_DIR / f"{s}_ground_truth.csv", "w", newline="")
        fl = open(OUT_DIR / f"{s}_labels.csv", "w", newline="")
        files += [fg, fl]
        gw[s] = csv.DictWriter(fg, fieldnames=GROUND_TRUTH_FIELDS); gw[s].writeheader()
        lw[s] = csv.DictWriter(fl, fieldnames=LABEL_FIELDS); lw[s].writeheader()

    t0 = time.time()
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
        x_v, x_delta = extract_solved_state(state)

        clean = {eid: measure(e) for eid, e in union.items()}
        z168 = [rng168.gauss(0.0, 1.0) for _ in en168]
        z112 = [rng112.gauss(0.0, 1.0) for _ in en112]

        split = split_of_row[t]
        r = cursor[split]
        x_v_buf[split][r, :] = x_v
        x_d_buf[split][r, :] = x_delta
        cursor[split] += 1

        for name, (entries, sigmas) in cands.items():
            zs = z112 if name == "T112" else z168
            row = {"timestamp": t, "real_timestamp": real_ts}
            for e, z in zip(entries, zs):
                v = clean[e["id"]]
                v += 0.0 + z * sigmas[e["id"]]
                row[e["id"]] = v
            tw[(name, split)].writerow(row)

        per_phase = metrics["v_pu_per_phase"]
        gw[split].writerow({
            "timestamp": t, "real_timestamp": real_ts,
            "converged": metrics["converged"], "iterations": metrics["iterations"],
            "v_pu_min": min(p["min"] for p in per_phase.values()),
            "v_pu_max": max(p["max"] for p in per_phase.values()),
            "losses_kw": metrics["losses_kw"], "losses_kvar": metrics["losses_kvar"],
            "feeder_head_p_kw": metrics["feeder_head_p_kw"],
            "feeder_head_q_kvar": metrics["feeder_head_q_kvar"],
            "solve_time_s": metrics["solve_time_s"],
        })
        lw[split].writerow({"timestamp": t, "real_timestamp": real_ts,
                            "attacked_meters": "[]", "attack_class": "", "magnitude": 0.0,
                            "stealth_flag": False})
        if t % 5000 == 0:
            print(f"  t={t} split={split} elapsed={time.time()-t0:.1f}s", flush=True)

    for f in files:
        f.close()
    for s in SPLITS:
        np.savez_compressed(
            OUT_DIR / f"{s}_x_true.npz", x_v=x_v_buf[s], x_delta=x_d_buf[s],
            v_index=np.array(list(state.v_index.items()), dtype=object),
            delta_index=np.array(list(state.delta_index.items()), dtype=object),
        )
    print(f"Done {num_rows} rows in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-rows", type=int, default=None)
    main(p.parse_args().max_rows)
