"""
streaming/run_demo.py -- one reproducible live Kafka run (producer ->
injector -> detector_consumer -> alarms), all four stages at multiphase.

  python -m streaming.run_demo <run_name> --start N [--noise 0|1] [--replay CSV]
                               [--steps 400] [--warmup W] [--sleep 0.2] [--functional]
                               [--live] [--save-gif]
  python -m streaming.run_demo --run B [--steps 400] [--save-gif]

--run B is the Run B preset (the settings of runB_realistic: multiphase, S1
corpus noise, start 81640 inside the test split, 120-step warm-up, sleep 0.2)
written to streaming/demo_runs/runB_live/ so runB_realistic is not overwritten.
It checks the broker, then starts producer, injector, detector_consumer and
the live viewer (streaming/live_view.py). --live adds the viewer to any run;
--save-gif also records it to streaming/demo_runs/live_demo.gif.

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
           REGISTRY_VARIANT="multiphase",
           PYTHONIOENCODING="utf-8")   # stage logs print repo paths; a non-ASCII path crashes cp1252 stdout
W = 10
RUN_PRESETS = {"B": {"run_name": "runB_live", "start": 81640, "noise": "1", "warmup": 120, "sleep": "0.2"}}
COMPOSE = BASE / "streaming" / "docker-compose.yml"


def alarm_end_offset():
    from kafka import KafkaConsumer, TopicPartition
    c = KafkaConsumer(bootstrap_servers="localhost:9092")
    tp = TopicPartition("alarms", 0)
    off = c.end_offsets([tp])[tp]
    c.close()
    return off


def broker_check(timeout_s=90):
    """Kafka at localhost:9092 answering? If not and docker is on PATH, bring up
    streaming/docker-compose.yml and wait for it. Exits with instructions otherwise."""
    import shutil
    try:
        from kafka import KafkaAdminClient
    except ImportError:
        sys.exit("broker check FAILED: kafka-python is not installed (pip install kafka-python)")

    def up():
        try:
            KafkaAdminClient(bootstrap_servers="localhost:9092", request_timeout_ms=3000).close()
            return True
        except Exception:
            return False
    if up():
        print("broker check: Kafka reachable at localhost:9092", flush=True)
        return
    if shutil.which("docker"):
        print("broker check: no broker -- docker compose up -d", flush=True)
        subprocess.run(["docker", "compose", "-f", str(COMPOSE), "up", "-d"], check=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if up():
                print(f"broker check: Kafka up after {time.time() - t0:.0f}s", flush=True)
                return
            time.sleep(3)
    sys.exit("broker check FAILED: no Kafka at localhost:9092 (docker compose -f streaming/docker-compose.yml "
             "up -d). Offline backup: streaming/demo_runs/live_demo.gif, or "
             "python -m streaming.live_view --replay streaming/demo_runs/runB_realistic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", nargs="?")
    ap.add_argument("--run", choices=sorted(RUN_PRESETS), default=None, help="preset; implies --live")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--noise", default="1")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--sleep", default="0.2")
    ap.add_argument("--functional", action="store_true")
    ap.add_argument("--live", action="store_true", help="also start streaming.live_view")
    ap.add_argument("--save-gif", action="store_true", help="viewer records streaming/demo_runs/live_demo.gif")
    a = ap.parse_args()
    if a.run:
        for k, v in RUN_PRESETS[a.run].items():
            if k != "run_name" or a.run_name is None:
                setattr(a, k, v)
    if a.run_name is None or a.start is None:
        ap.error("run_name and --start are required (or use --run B)")
    live = a.live or a.save_gif or a.run is not None
    if live:
        broker_check()

    run_dir = BASE / "streaming" / "demo_runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("alarms.csv", "labels.csv", "summary.json", "timeline.png", "diagnosis.json",
                  "gate_real_kafka.csv", "truth.csv"):
        (run_dir / stale).unlink(missing_ok=True)   # never analyse a previous run's outputs
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
    viewer = None
    if live:
        title = (f"LIVE -- {a.run_name}: OpenDSS producer -> Kafka -> injector -> detector  "
                 f"(start {S}, noise {'S1' if a.noise == '1' else 'off'}, A1 @ {S + 140}, A2 @ {S + 260})")
        viewer = subprocess.Popen([sys.executable, "-m", "streaming.live_view", "--xmin", str(S - 2),
                                   "--xmax", str(S + a.steps + 2), "--stop-at", str(S + a.steps - 1),
                                   "--idle-exit", "240", "--title", title,
                                   *(["--save-gif"] if a.save_gif else [])], cwd=BASE, env=env)
    expected = a.steps - (W - 1)
    last, last_change = -1, time.time()
    try:
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
    except BaseException:                  # crash or Ctrl+C: never leave the stages running orphaned
        for p in [*procs.values(), viewer]:
            if p is not None and p.poll() is None:
                p.terminate()
        raise
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
        return subprocess.run([sys.executable, "-m", "streaming.demo_tools", *args], cwd=BASE, env=ENV,
                              check=False).returncode
    if tool("dump", str(run_dir)) != 0:
        print(f"no alarms/labels to analyse -- see the stage logs in streaming/demo_runs/{a.run_name}/ "
              "(producer.log, injector.log, detector.log)", flush=True)
        if viewer is not None and viewer.poll() is None:     # 4: leave the window up; it has the run on it
            print("live viewer still open -- close its window to exit", flush=True)
            viewer.wait()
        sys.exit(1)
    label = (f"{a.run_name}: " + ("PIPELINE FUNCTIONAL TEST (noise-free) - not detector performance" if a.functional
                                   else ("REPLAY of the test corpus through Kafka (not live OpenDSS)" if a.replay
                                         else "live OpenDSS producer, corpus noise (S1), test-split start")))
    tool("analyze", str(run_dir), label, *(["--functional"] if a.functional else []))
    if a.functional:
        tool("gate", str(run_dir), "20")
    else:
        tool("diagnose", str(run_dir))
    if viewer is not None and viewer.poll() is None:
        print("live viewer still open -- close its window to exit", flush=True)
        viewer.wait()


if __name__ == "__main__":
    main()
