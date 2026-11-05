"""
streaming/detector_consumer.py -- the live detector stage. Subscribes to
telemetry.delivered (registry_variant="multiphase"), buffers messages by
timestamp, assembles the 168-channel vector in the estimator's registry
order, and once W=10 CONSECUTIVE complete timesteps are buffered, calls
streaming.detector_core.DetectorCore.score_vector and publishes
{timestamp, fusion_score, alarm, top_channel, J, maxLNR} to "alarms".

With the peer detectors on (default; DETECTOR_PEER=0 turns them off) every
complete vector is also fed to streaming.peer_monitor.PeerMonitor, and the
published "alarm" is the combined system: fusion OR V-peer OR P-peer. The
record then also carries fusion_alarm, fusion_ratio, v_ratio, p_ratio (score /
own threshold), v_alarm, p_alarm and fired_by.

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
    def __init__(self, core, window_timeout_s=DEFAULT_WINDOW_TIMEOUT_S, clock=None, peer=None):
        self.core = core
        self.peer = peer               # optional streaming.peer_monitor.PeerMonitor
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
        peer = self.peer.push(t, vec) if self.peer is not None else None
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
        res = {"timestamp": t, "scored": True, "real_timestamp": rts,
               "fusion_score": out["fusion_score"], "alarm": out["alarm"],
               "top_channel": out["top_channel"], "J": out["J"], "maxLNR": out["maxLNR"],
               "bdd_score": out["bdd_score"], "ae_mahalanobis": out["ae_mahalanobis"],
               "current_vec": vec}             # current_vec for the equivalence gate; not published live
        if peer is not None:
            # combined system: fusion OR V-peer OR P-peer (each at its own ~1% threshold)
            fusion_ratio = out["fusion_score"] / self.core.thr_f
            fired = [n for n, on in (("fusion", out["alarm"]), ("V-peer", peer["v_alarm"]),
                                     ("P-peer", peer["p_alarm"])) if on]
            peer_leads = max(peer["v_ratio"], peer["p_ratio"]) > fusion_ratio
            res.update({"fusion_alarm": out["alarm"], "fusion_ratio": fusion_ratio,
                        "v_ratio": peer["v_ratio"], "p_ratio": peer["p_ratio"],
                        "v_alarm": peer["v_alarm"], "p_alarm": peer["p_alarm"],
                        "alarm": bool(fired), "fired_by": "+".join(fired),
                        "top_channel": peer["peer_top_channel"] if peer_leads else out["top_channel"]})
        return res


def main():
    import os
    from kafka import KafkaConsumer, KafkaProducer
    from .kafka_config import BOOTSTRAP_SERVERS, TOPIC_DELIVERED, TOPIC_ALARMS

    variant = os.environ.get("REGISTRY_VARIANT", "multiphase")
    core = DetectorCore(registry_variant=variant)
    peer = None
    if os.environ.get("DETECTOR_PEER", "1") == "1":
        from .peer_monitor import PeerMonitor
        peer = PeerMonitor(core.cols)
        print(f"Peer detectors on: V-peer thr={peer.V.thr:.3f}, P-peer thr={peer.P.thr:.3f}, "
              f"persistence {3} steps", flush=True)
    dc = DetectorConsumer(core, peer=peer)

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
        rec = {"timestamp": res["timestamp"], "fusion_score": res["fusion_score"], "alarm": res["alarm"],
               "top_channel": res["top_channel"], "J": res["J"], "maxLNR": res["maxLNR"],
               "bdd_score": res["bdd_score"], "ae_mahalanobis": res["ae_mahalanobis"]}
        for k in ("fusion_alarm", "fusion_ratio", "v_ratio", "p_ratio", "v_alarm", "p_alarm", "fired_by"):
            if k in res:
                rec[k] = res[k]
        kafka_producer.send(TOPIC_ALARMS, key=str(res["timestamp"]).encode(), value=rec)
        kafka_producer.flush()
        if res["alarm"] or res["timestamp"] % 50 == 0 or "v_ratio" in res:
            ratios = (f" F={res['fusion_ratio']:.2f} V={res['v_ratio']:.2f} P={res['p_ratio']:.2f}"
                      if "v_ratio" in res else "")
            print(f"  t={res['timestamp']} fusion={res['fusion_score']:.4f}{ratios} alarm={res['alarm']} "
                  f"{('by ' + res['fired_by']) if res.get('fired_by') else ''} top={res['top_channel']}", flush=True)
    print(f"Detector done: scored {n_scored}, alarms {n_alarm}, skipped-incomplete {n_skip}", flush=True)


if __name__ == "__main__":
    main()
