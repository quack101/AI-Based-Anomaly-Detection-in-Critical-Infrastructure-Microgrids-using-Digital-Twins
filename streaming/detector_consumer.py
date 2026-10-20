"""
streaming/detector_consumer.py -- the live detector stage. Subscribes to
telemetry.delivered (registry_variant="multiphase"), buffers messages by
timestamp, assembles the 168-channel vector in the estimator's registry
order, and once W=10 CONSECUTIVE complete timesteps are buffered, calls
streaming.detector_core.DetectorCore.score_vector and publishes
{timestamp, fusion_score, alarm, top_channel, J, maxLNR} to "alarms".

No fabrication: a timestep whose 168-vector is still incomplete when its
window times out is logged "not scored" and skipped (which also breaks
the W=10 consecutiveness, so the next score waits for 10 fresh complete
steps). The frozen detector is loaded via DetectorCore; nothing here
retrains or reimplements it.
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from streaming.detector_core import DetectorCore, AE_W

DEFAULT_WINDOW_TIMEOUT_S = 5.0


class DetectorConsumer:
    def __init__(self, core, window_timeout_s=DEFAULT_WINDOW_TIMEOUT_S, clock=None):
        self.core = core
        self.cols = core.cols
        self.col_set = set(self.cols)
        self.W = AE_W
        self.window_timeout_s = window_timeout_s
        self._clock = clock or time.monotonic
        self._buf = OrderedDict()      # t -> {id: value}
        self._first_seen = {}          # t -> clock
        self._real_ts = {}             # t -> real_timestamp
        self._history = []             # [(t, vec)] consecutive complete timesteps

    def ingest(self, msg):
        t = msg["timestamp"]
        if t not in self._buf:
            self._buf[t] = {}
            self._first_seen[t] = self._clock()
        self._buf[t].update(msg.get("measurements", {}))
        self._real_ts[t] = msg.get("real_timestamp")

        complete = self.col_set <= set(self._buf[t])
        timed_out = (self._clock() - self._first_seen[t]) >= self.window_timeout_s
        if not (complete or timed_out):
            return None
        return self._close(t, complete)

    def flush_all(self):
        out = []
        for t in sorted(self._buf.keys()):
            complete = self.col_set <= set(self._buf[t])
            out.append(self._close(t, complete))
        return out

    def _close(self, t, complete):
        buf = self._buf.pop(t)
        self._first_seen.pop(t, None)
        rts = self._real_ts.pop(t, None)
        if not complete:
            self._history = []         # break consecutiveness; never fabricate
            missing = len(self.col_set - set(buf))
            return {"timestamp": t, "scored": False, "reason": "incomplete",
                    "n_present": len(buf), "n_missing": missing}
        vec = np.array([buf[c] for c in self.cols], dtype=float)
        if self._history and t == self._history[-1][0] + 1:
            self._history.append((t, vec))
        else:
            self._history = [(t, vec)]
        self._history = self._history[-self.W:]
        if len(self._history) < self.W:
            return {"timestamp": t, "scored": False, "reason": "warmup",
                    "have": len(self._history)}
        window = np.stack([v for _, v in self._history], axis=0)
        out = self.core.score_vector(window, str(rts))
        return {"timestamp": t, "scored": True, "real_timestamp": rts,
                "fusion_score": out["fusion_score"], "alarm": out["alarm"],
                "top_channel": out["top_channel"], "J": out["J"], "maxLNR": out["maxLNR"],
                "bdd_score": out["bdd_score"], "ae_mahalanobis": out["ae_mahalanobis"],
                "current_vec": vec}            # current_vec for the equivalence gate; not published live


def main():
    import os
    from kafka import KafkaConsumer, KafkaProducer
    from .kafka_config import BOOTSTRAP_SERVERS, TOPIC_DELIVERED, TOPIC_ALARMS

    variant = os.environ.get("REGISTRY_VARIANT", "multiphase")
    core = DetectorCore(registry_variant=variant)
    dc = DetectorConsumer(core)

    kafka_consumer = KafkaConsumer(
        TOPIC_DELIVERED,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="detector" + os.environ.get("KAFKA_GROUP_SUFFIX", ""),
        consumer_timeout_ms=int(os.environ.get("DETECTOR_IDLE_MS", "60000")),
    )
    kafka_producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"Detector consumer up (variant={variant}, J99={core.J99:.2f}, LNR99={core.LNR99:.3f}, "
          f"fusion thr={core.thr_f:.4f}). Waiting on {TOPIC_DELIVERED}...", flush=True)

    n_scored = n_alarm = n_skip = 0
    for record in kafka_consumer:
        res = dc.ingest(record.value)
        if res is None:
            continue
        if not res["scored"]:
            if res["reason"] == "incomplete":
                n_skip += 1
                print(f"  t={res['timestamp']} NOT SCORED (incomplete: {res['n_missing']} missing)", flush=True)
            continue
        n_scored += 1
        n_alarm += int(res["alarm"])
        kafka_producer.send(TOPIC_ALARMS, key=str(res["timestamp"]).encode(), value={
            "timestamp": res["timestamp"], "fusion_score": res["fusion_score"], "alarm": res["alarm"],
            "top_channel": res["top_channel"], "J": res["J"], "maxLNR": res["maxLNR"],
            "bdd_score": res["bdd_score"], "ae_mahalanobis": res["ae_mahalanobis"]})
        kafka_producer.flush()
        if res["alarm"] or res["timestamp"] % 50 == 0:
            print(f"  t={res['timestamp']} fusion={res['fusion_score']:.4f} alarm={res['alarm']} "
                  f"top={res['top_channel']}", flush=True)
    print(f"Detector done: scored {n_scored}, alarms {n_alarm}, skipped-incomplete {n_skip}", flush=True)


if __name__ == "__main__":
    main()
