"""
Phase 2 live-path equivalence GATE (in-process; JSON serialization is
identical to Kafka's value_serializer json.dumps/loads, so this tests the
wire assembly + scoring without broker flakiness; the separate Kafka
smoke test confirms the real broker path).

Chains Producer (multiphase) -> per-timestep messages -> JSON round-trip
-> DetectorConsumer.ingest, and compares, for the scored timestamps:
  - the wire-assembled 168-vector vs the producer's own merged truth vector
  - DetectorConsumer's fusion_score/alarm vs DetectorCore.score_vector
    called directly on the producer's truth window.
Must match to 1e-6 (vector and fusion); alarms identical.

Run: python -m streaming.verify_phase2 [n_timesteps]
"""

import json
import sys

import numpy as np

from streaming.producer import Producer
from streaming.detector_core import DetectorCore, AE_W
from streaming.detector_consumer import DetectorConsumer


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    prod = Producer(registry_variant="multiphase", add_noise=False, seed=0)
    core = DetectorCore(registry_variant="multiphase")
    dc = DetectorConsumer(core, window_timeout_s=1e9)      # no timeout in-process
    cols = core.cols

    hist = []
    rows = []
    for t in range(n):
        raw, _ = prod.run_timestep(t)
        merged, rts = {}, None
        for m in raw:
            merged.update(m["measurements"]); rts = m["real_timestamp"]
        truth = np.array([merged[c] for c in cols], dtype=float)

        hist.append((t, truth)); hist = hist[-AE_W:]
        direct = None
        if len(hist) == AE_W:
            direct = core.score_vector(np.stack([v for _, v in hist]), str(rts))

        wire = None
        for m in raw:
            r = dc.ingest(json.loads(json.dumps(m)))    # JSON round-trip == Kafka serialization
            if r is not None and r.get("scored"):
                wire = r
        if direct is not None and wire is not None:
            rows.append({"t": t,
                         "dvec": float(np.max(np.abs(wire["current_vec"] - truth))),
                         "dfus": abs(wire["fusion_score"] - direct["fusion_score"]),
                         "alarm_match": bool(wire["alarm"] == direct["alarm"]),
                         "fusion": direct["fusion_score"], "alarm": direct["alarm"]})

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"scored {len(df)} of {n} timesteps (first {AE_W-1} are warmup)")
    print(f"  max |wire_vec - truth_vec|   = {df['dvec'].max():.3e}")
    print(f"  max |wire_fusion - direct|   = {df['dfus'].max():.3e}")
    print(f"  alarms identical             = {bool(df['alarm_match'].all())}")
    print(f"  alarms raised (clean region) = {int(df['alarm'].sum())}/{len(df)}")
    gate = df["dvec"].max() <= 1e-6 and df["dfus"].max() <= 1e-6 and bool(df["alarm_match"].all())
    print(f"\nPHASE 2 EQUIVALENCE GATE: {'PASS' if gate else 'FAIL'}")
    return gate


if __name__ == "__main__":
    main()
