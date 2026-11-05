"""
Build the A3/A4/A5 attack plans of the full demo (streaming/demo_schedule.py)
OFFLINE, with the library's own attacks.library.attack() and OpenDSS -- the
attacker's planning step. The live injector then only adds the stored vectors
(attacks/precomputed.py).

The attacker plans on the clean test telemetry for the same stream indices
(simulation/phase6_corpus/test_telemetry.csv; stream index 73440 = test row 0):
the same loads and noise level as the live producer, a different noise draw.
Each attack vector a_t is the attacker's change to the readings, so it is
re-applied unchanged to the live readings.

Run (Windows, from latest-code):  ..\\.venv-1\\Scripts\\python.exe -m validation.gen_demo_attacks [--start 80440]
"""

import argparse
import io
import contextlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "simulation" / "phase6_corpus" / "test_telemetry.csv"
TEST_ROW0 = 73440


def build(item):
    from attacks.library import AttackContext, AttackSpec, attack
    t0 = time.time()
    df = pd.read_csv(CORPUS)
    chan = [c for c in df.columns if c not in ("timestamp", "real_timestamp")]
    z = df[chan].reset_index(drop=True)
    ts = df["real_timestamp"].tolist()
    row0 = item["start"] - TEST_ROW0
    b = item["build"]
    kw = {k: v for k, v in b.items() if k in ("zone", "group_size", "access")}
    spec = AttackSpec(item["type"], row0, item["duration"], b["profile"], seed=b["seed"],
                      magnitude=item["magnitude"], **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = AttackContext()
        za, lb = attack(z, ts, spec, ctx)
    a = za.iloc[row0:row0 + item["duration"]].to_numpy(float) - z.iloc[row0:row0 + item["duration"]].to_numpy(float)
    out = BASE / item["plan"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, timestamps=np.arange(item["start"], item["start"] + item["duration"]),
             channels=np.array(chan), a=a)
    P = [j for j, c in enumerate(chan) if "_P" in c and c.startswith("M")]
    return (f"{item['type']}: {out.relative_to(BASE)}  rows {len(a)}, channels changed "
            f"{int((np.abs(a).max(0) > 0).sum())}, max|dP| {np.abs(a[:, P]).max():.1f} kW, "
            f"max|a| {np.abs(a).max():.4g}  ({time.time() - t0:.0f}s)")


def main():
    from streaming.demo_schedule import schedule_C
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=80440)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    items = [d for d in schedule_C(a.start) if "plan" in d]
    todo = [d for d in items if a.force or not (BASE / d["plan"]).exists()]
    print(f"{len(items)} plans, {len(todo)} to build", flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=len(todo)) as ex:
        for msg in ex.map(build, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
