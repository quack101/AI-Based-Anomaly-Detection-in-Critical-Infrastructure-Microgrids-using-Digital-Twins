"""
Why did the early IN-PROCESS test over-alarm while the Kafka run did not?

DetectorCore's power-flow start (observability/pf_start.py) rewrites all 91
loads and resets the regulator taps in the OpenDSS instance it lives in. If a
Producer shares that instance, its 42 undriven loads are no longer at their
authored nominal values, so its "true" circuit differs from the corpus's.
Over Kafka every stage is its own process, so this cannot happen.

This script scores the SAME producer settings two ways:
  interleaved    producer step, then detector step, in ONE process (shared circuit)
  generate-first all producer vectors first, detector built afterwards
and prints alarm counts and median scores. Run:
  python -m streaming.diagnose_inprocess
"""
import json

import numpy as np

from streaming.producer import Producer
from streaming.detector_core import DetectorCore, AE_W

S1 = json.load(open("D:/capstone_dss/config/sigma_overrides_s1.json"))["sigma"]


def summarize(tag, rec):
    r = np.array(rec, float)
    print(f"{tag:46s} alarms {int(r[:, 0].sum()):2d}/{len(r)}  median J {np.median(r[:, 1]):6.1f}  "
          f"maha {np.median(r[:, 3]):6.1f}  fusion {np.median(r[:, 4]):.3f}", flush=True)


def vectors(prod, cols, start, warmup, n):
    for t in range(start - warmup, start):
        prod.run_timestep(t)
    out = []
    for t in range(start, start + n):
        raw, _ = prod.run_timestep(t)
        merged = {}
        for m in raw:
            merged.update(m["measurements"])
            rts = m["real_timestamp"]
        out.append((np.array([merged[c] for c in cols]), rts))
    return out


def main(n=30):
    core = DetectorCore("multiphase")           # compiles its own copy in THIS process
    cols = core.cols
    for start, warmup, label in [(81640, 120, "test-split start (81640), warmup 120"), (0, 0, "t=0, no warmup")]:
        # interleaved: the producer and the detector share this process's OpenDSS instance
        prod = Producer(registry_variant="multiphase", add_noise=True, seed=0, sigma_overrides=S1)
        hist, rec = [], []
        for t in range(start - warmup, start):
            prod.run_timestep(t)
        for t in range(start, start + n):
            raw, _ = prod.run_timestep(t)
            merged = {}
            for m in raw:
                merged.update(m["measurements"]); rts = m["real_timestamp"]
            hist.append(np.array([merged[c] for c in cols])); hist = hist[-AE_W:]
            if len(hist) == AE_W:
                o = core.score_vector(np.stack(hist), str(rts))
                rec.append((o["alarm"], o["J"], o["maxLNR"], o["ae_mahalanobis"], o["fusion_score"]))
        summarize(f"interleaved (shared circuit)  {label}", rec)
        # generate-first: producer alone in a fresh compile, THEN score
        prod = Producer(registry_variant="multiphase", add_noise=True, seed=0, sigma_overrides=S1)
        vs = vectors(prod, cols, start, warmup, n)
        rec = []
        for i in range(AE_W - 1, len(vs)):
            window = np.stack([v for v, _ in vs[i - AE_W + 1:i + 1]])
            o = core.score_vector(window, str(vs[i][1]))
            rec.append((o["alarm"], o["J"], o["maxLNR"], o["ae_mahalanobis"], o["fusion_score"]))
        summarize(f"generate-first (separate circuit) {label}", rec)


if __name__ == "__main__":
    main()
