"""
Generate an A3 / A4 / A5 attack benchmark with the library's own attack()
(OpenDSS required -- run on Windows with the project venv).

Why: fdia_v1 has only 3+3 A3 windows and no A4/A5 windows. This builds many
more on the CLEAN test telemetry (simulation/phase6_corpus/test_telemetry.csv),
exactly as validation/build_fdia_v1.py does (same attack(), same AttackSpec
fields, same levels), one window at a time.

Output: dataset/attack_bench/
  windows.jsonl   one line per window: id, type, level, access, profile,
                  start (row in test split), duration, seed, seconds, diag, ok/error
  a_<id>.npy      attack vectors a = z_attacked - z_clean for the window rows,
                  shape (duration, 168), float64, columns = channel order of
                  test_telemetry.csv (without timestamp/real_timestamp)
  channels.json   that channel order
Results are appended as each window finishes, so a partial run is usable.

Self-check first: fdia_v1 test window 0 (A3) is rebuilt and compared with
dataset/fdia_v1/test_z.csv - corpus; the run stops if they differ.

Overrides (environment): BENCH_NAME (output folder under dataset/), A45_MAGS
(comma list), A45_REPS, SKIP_A3=1, BENCH_SEED.

Run (from latest-code): ..\\.venv-1\\Scripts\\python.exe -u -m validation.gen_attack_bench
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "simulation" / "phase6_corpus" / "test_telemetry.csv"
OUT = BASE / "dataset" / os.environ.get("BENCH_NAME", "attack_bench")
A3_LEVELS = [0.02, 0.05, 0.10]
PROFILES = ["step", "ramp", "pulse"]
A3_FULL_DURATIONS, A3_FULL_REPS = [15, 60, 240], 3          # 81 windows
A3_PART_DURATIONS, A3_PART_REPS = [15, 60], 2               # 36 windows
A45_MAGS = [float(x) for x in os.environ.get("A45_MAGS", "0.01,0.02,0.05").split(",")]
A45_REPS = int(os.environ.get("A45_REPS", "5"))
A45_DURATION = 15                                           # default: 30 windows
SKIP_A3 = os.environ.get("SKIP_A3", "0") == "1"
A45_DEADLINE_MIN = 240       # stop submitting A4/A5 windows after this many minutes
SEED0 = int(os.environ.get("BENCH_SEED", "20261105"))

_Z = _TS = _CTX = None


def _init():
    global _Z, _TS, _CTX
    import io, contextlib
    from attacks.library import AttackContext
    df = pd.read_csv(CORPUS)
    chan = [c for c in df.columns if c not in ("timestamp", "real_timestamp")]
    _Z = df[chan].reset_index(drop=True)
    _TS = df["real_timestamp"].tolist()
    with contextlib.redirect_stdout(io.StringIO()):
        _CTX = AttackContext()
        _ = _CTX.model, _CTX.caps
        _ = _CTX.est


def _spec(job):
    from attacks.library import AttackSpec
    if job["type"] == "A3":
        return AttackSpec("A3", job["start"], job["duration"], job["profile"], seed=job["seed"], zone=True,
                          group_size=4, magnitude=job["magnitude"], access=job["access"])
    return AttackSpec(job["type"], job["start"], job["duration"], job["profile"], seed=job["seed"],
                      magnitude=job["magnitude"])


def run_job(job):
    import io, contextlib
    from attacks.library import attack
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            za, lb = attack(_Z, _TS, _spec(job), _CTX)
        s, d = job["start"], job["duration"]
        a = za.iloc[s:s + d].to_numpy(float) - _Z.iloc[s:s + d].to_numpy(float)
        diag = {}
        for c in lb.columns:
            if c in ("timestamp", "attacked", "attack_type", "target_ids"):
                continue
            v = pd.to_numeric(lb[c].iloc[s:s + d], errors="coerce")
            if v.notna().any():
                diag[c] = float(v.mean())
        tg = [t for t in lb["target_ids"].iloc[s:s + d] if t]
        diag["n_target_channels"] = len(tg[0].split(";")) if tg else 0
        diag["attacked_rows"] = int(lb["attacked"].iloc[s:s + d].sum())
        return job, a, {"ok": True, "seconds": time.time() - t0, "diag": diag}
    except Exception as e:  # keep going; log the failure
        return job, None, {"ok": False, "seconds": time.time() - t0, "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc()[-1500:]}


def self_check():
    """Rebuild fdia_v1 test window 0 and compare with the shipped dataset."""
    man = json.load(open(BASE / "dataset" / "fdia_v1" / "manifest.json"))
    w = next(x for x in man["windows"]["test"] if x["type"] == "A3")
    lvl = {"2pct": 0.02, "5pct": 0.05, "10pct": 0.10}[w["level"].split("_")[0]]
    job = {"id": -1, "type": "A3", "magnitude": lvl, "access": w["level"].split("_")[1],
           "profile": w["profile"], "start": w["start"], "duration": w["duration"], "seed": w["seed"]}
    _, a, info = run_job(job)
    if not info["ok"]:
        print("SELF-CHECK FAILED TO RUN:", info["error"], info.get("trace", ""), flush=True)
        return False
    zt = pd.read_csv(BASE / "dataset" / "fdia_v1" / "test_z.csv")
    s, d = w["start"], w["duration"]
    ref = zt[list(_Z.columns)].iloc[s:s + d].to_numpy(float) - _Z.iloc[s:s + d].to_numpy(float)
    err = float(np.max(np.abs(a - ref)))
    scale = float(np.max(np.abs(ref)))
    print(f"SELF-CHECK fdia_v1 test window {w['window_id']} ({w['level']}, {w['profile']}, {d} rows): "
          f"max|a_rebuilt - a_shipped| = {err:.3e} (max|a| = {scale:.3e}), {info['seconds']:.1f}s", flush=True)
    return err <= 1e-6 * max(1.0, scale)


def plan():
    rng = np.random.default_rng(SEED0)
    n = len(pd.read_csv(CORPUS, usecols=["timestamp"]))
    jobs = []

    def add(**kw):
        d = kw["duration"]
        kw["start"] = int(rng.integers(60, n - d - 60))
        kw["seed"] = int(rng.integers(1, 2**31 - 1))
        kw["id"] = len(jobs)
        jobs.append(kw)

    for rep in range(0 if SKIP_A3 else max(A3_FULL_REPS, A3_PART_REPS)):
        for lvl in A3_LEVELS:
            for prof in PROFILES:
                if rep < A3_FULL_REPS:
                    for d in A3_FULL_DURATIONS:
                        add(type="A3", magnitude=lvl, access="full", profile=prof, duration=d)
                if rep < A3_PART_REPS:
                    for d in A3_PART_DURATIONS:
                        add(type="A3", magnitude=lvl, access="partial", profile=prof, duration=d)
    n_a3 = len(jobs)
    for rep in range(A45_REPS):           # rep-major so a partial run is balanced
        for m in A45_MAGS:
            for t in ("A4", "A5"):
                add(type=t, magnitude=m, access="", profile="step", duration=A45_DURATION)
    return jobs, n_a3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"python {sys.version.split()[0]}; cpus {os.cpu_count()}", flush=True)
    _init()
    json.dump(list(_Z.columns), open(OUT / "channels.json", "w"))
    if not self_check():
        print("ABORT: rebuilt window does not match fdia_v1", flush=True)
        sys.exit(2)
    jobs, n_a3 = plan()
    done = set()
    if (OUT / "windows.jsonl").exists():
        for line in open(OUT / "windows.jsonl"):
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["id"])
    todo = [j for j in jobs if j["id"] not in done]
    print(f"plan: {n_a3} A3 + {len(jobs) - n_a3} A4/A5 windows; {len(done)} already done; {len(todo)} to run",
          flush=True)
    workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    log = open(OUT / "windows.jsonl", "a")
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        pending = {}
        queue = list(todo)

        def submit_next():
            while queue and len(pending) < 2 * workers:
                j = queue.pop(0)
                if j["type"] != "A3" and (time.time() - t_start) / 60 > A45_DEADLINE_MIN:
                    continue
                pending[ex.submit(run_job, j)] = j

        submit_next()
        while pending:
            for fut in as_completed(list(pending)):
                job = pending.pop(fut)
                job, a, info = fut.result()
                if info["ok"]:
                    np.save(OUT / f"a_{job['id']}.npy", a)
                    n_ok += 1
                else:
                    n_fail += 1
                log.write(json.dumps({**job, **info}, default=float) + "\n"); log.flush()
                el = (time.time() - t_start) / 60
                print(f"[{el:6.1f} min] {job['type']} id {job['id']} {job.get('access','')} m={job['magnitude']} "
                      f"{job['profile']} d={job['duration']}: {'ok' if info['ok'] else 'FAIL ' + info['error']} "
                      f"({info['seconds']:.1f}s)  ok {n_ok} fail {n_fail} left {len(queue) + len(pending)}", flush=True)
                submit_next()
                break
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min: ok {n_ok}, failed {n_fail}", flush=True)


if __name__ == "__main__":
    main()
