"""
streaming/run_demo.py -- one reproducible live Kafka run (producer ->
injector -> detector_consumer -> alarms), all four stages at multiphase.

  python -m streaming.run_demo <run_name> --start N [--noise 0|1] [--replay CSV]
                               [--steps 400] [--warmup W] [--sleep 0.2] [--functional]

The attack windows are relative to --start (clean 140 steps, A1 60 steps,
clean 60, A2 60, clean 80 = 400 steps):
  A1: k=20 channels, mixed class, step profile, P/Q bias 10% of |z_t|, V bias 0.01 pu
  A2: one meter's P/Q/V channels, step profile, same levels
Artifacts go to streaming/demo_runs/<run_name>/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           REGISTRY_VARIANT="multiphase")
W = 10


def alarm_end_offset():
    from kafka import KafkaConsumer, TopicPartition
    c = KafkaConsumer(bootstrap_servers="localhost:9092")
    tp = TopicPartition("alarms", 0)
    off = c.end_offsets([tp])[tp]
    c.close()
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--noise", default="1")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--sleep", default="0.2")
    ap.add_argument("--functional", action="store_true")
    a = ap.parse_args()

    run_dir = BASE / "streaming" / "demo_runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    S = a.start
    specs = [
        {"type": "A1", "start": S + 140, "duration": 60, "profile": "step", "seed": 20261, "k": 20,
         "channel_class": "mixed", "pq_rel": 0.10, "v_abs_pu": 0.01},
        {"type": "A2", "start": S + 260, "duration": 60, "profile": "step", "seed": 20262,
         "pq_rel": 0.10, "v_abs_pu": 0.01},
    ]
    json.dump(specs, open(run_dir / "attacks.json", "w"), indent=2)

    t0 = time.time()
    subprocess.run([sys.executable, "-m", "streaming.demo_tools", "reset"], cwd=BASE, env=ENV, check=True)
    env = dict(ENV, KAFKA_GROUP_SUFFIX="-" + a.run_name, INJECTOR_ATTACKS=str(run_dir / "attacks.json"),
               INJECTOR_IDLE_MS="180000", DETECTOR_IDLE_MS="180000",
               PRODUCER_NOISE=a.noise, PRODUCER_START=str(S), PRODUCER_MAX_STEPS=str(a.steps),
               PRODUCER_WARMUP=str(a.warmup), PRODUCER_SLEEP_S=a.sleep,
               PRODUCER_TRUTH_LOG=str(run_dir / "truth.csv"))
    if a.replay:
        env["PRODUCER_REPLAY"] = a.replay
    logs = {n: open(run_dir / f"{n}.log", "w") for n in ("producer", "injector", "detector")}
    procs = {
        "detector": subprocess.Popen([sys.executable, "-m", "streaming.detector_consumer"], cwd=BASE, env=env,
                                     stdout=logs["detector"], stderr=subprocess.STDOUT),
        "injector": subprocess.Popen([sys.executable, "-m", "streaming.injector"], cwd=BASE, env=env,
                                     stdout=logs["injector"], stderr=subprocess.STDOUT),
        "producer": subprocess.Popen([sys.executable, "-m", "streaming.producer"], cwd=BASE, env=env,
                                     stdout=logs["producer"], stderr=subprocess.STDOUT),
    }
    expected = a.steps - (W - 1)
    last, last_change = -1, time.time()
    while True:
        time.sleep(5)
        off = alarm_end_offset()
        if off != last:
            last, last_change = off, time.time()
            print(f"  alarms published: {off}/{expected}  ({time.time() - t0:.0f}s)", flush=True)
        if off >= expected:
            break
        if time.time() - last_change > 240:
            print("  no progress for 240 s -- stopping", flush=True)
            break
        if procs["detector"].poll() is not None:
            print("  detector exited early", flush=True)
            break
    time.sleep(3)
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for f in logs.values():
        f.close()
    elapsed = time.time() - t0
    print(f"pipeline finished in {elapsed:.0f}s", flush=True)
    json.dump({"run": a.run_name, "start": S, "steps": a.steps, "noise": a.noise, "replay": a.replay,
               "warmup": a.warmup, "sleep_s": a.sleep, "pipeline_seconds": elapsed, "specs": specs},
              open(run_dir / "run.json", "w"), indent=2)

    def tool(*args):
        subprocess.run([sys.executable, "-m", "streaming.demo_tools", *args], cwd=BASE, env=ENV, check=False)
    tool("dump", str(run_dir))
    label = (f"{a.run_name}: " + ("PIPELINE FUNCTIONAL TEST (noise-free) - not detector performance" if a.functional
                                   else ("REPLAY of the test corpus through Kafka (not live OpenDSS)" if a.replay
                                         else "live OpenDSS producer, corpus noise (S1), test-split start")))
    tool("analyze", str(run_dir), label, *(["--functional"] if a.functional else []))
    if a.functional:
        tool("gate", str(run_dir), "20")
    else:
        tool("diagnose", str(run_dir))


if __name__ == "__main__":
    main()
