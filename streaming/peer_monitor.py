"""
streaming/peer_monitor.py -- the two peer-consistency detectors on the live
stream, next to the frozen fusion detector (streaming/detector_core.py).

  V-peer  validation/detect_a2_peer.py   each meter's VOLTAGE predicted from all
          other meters; clipped trailing means over 5/15/30 steps (A2: one
          tampered meter)
  P-peer  validation/detect_peer_power.py each meter's ACTIVE POWER predicted
          from all other meters (A3 load redistribution; also A4/A5)

Both are fitted on clean fdia_v1 train rows and thresholded at 99% of clean
val (the same calibration as the fusion threshold). Each alarms after
PERSIST = 3 consecutive steps above its threshold. Offline (reports/
overall_detection.md) the system "fusion OR V-peer OR P-peer" raised attack
detection on held-out benchmarks from 48% to 76% at ~1% false alarms.

push(t, vec) takes every COMPLETE 168-vector in stream order (canonical
channel order = DetectorCore.cols); a gap in t resets the history, exactly as
DetectorConsumer resets its own window. The fitted detectors are cached in
streaming/peer_fit_cache.pkl (rebuilt automatically if missing).
"""

import os
import pickle
from collections import deque
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("PEER_FIT_CACHE", BASE / "streaming" / "peer_fit_cache.pkl"))
PERSIST = 3
HISTORY = 30          # longest V-peer trailing window


def load_detectors(use_cache=True):
    if use_cache and CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from validation import detect_a2_peer, detect_peer_power
    dets = {"V": detect_a2_peer.fit_default(), "P": detect_peer_power.fit_default()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(dets, f)
    return dets


class PeerMonitor:
    def __init__(self, cols, detectors=None):
        self.dets = detectors or load_detectors()
        self.V, self.P = self.dets["V"], self.dets["P"]
        if list(self.V.cols) != list(cols) or list(self.P.cols) != list(cols):
            raise ValueError("peer detectors were fitted on a different channel order than the stream")
        self.cols = list(cols)
        self._hist = deque(maxlen=HISTORY)
        self._last_t = None
        self._run = {"V": 0, "P": 0}

    def push(self, t, vec):
        if self._last_t is None or t != self._last_t + 1:
            self._hist.clear()
            self._run = {"V": 0, "P": 0}
        self._last_t = t
        self._hist.append(np.asarray(vec, float))
        X = np.stack(self._hist)
        with np.errstate(all="ignore"):
            v = float(np.nan_to_num(self.V.score(X)[-1]))
            p = float(np.nan_to_num(self.P.score(X[-1:])[0]))
        vr, pr = v / self.V.thr, p / self.P.thr
        self._run["V"] = self._run["V"] + 1 if vr > 1 else 0
        self._run["P"] = self._run["P"] + 1 if pr > 1 else 0
        v_alarm, p_alarm = self._run["V"] >= PERSIST, self._run["P"] >= PERSIST
        with np.errstate(all="ignore"):
            if pr >= vr:
                top = self.cols[int(self.P.localize(X[-1:])[0])]
            else:
                top = self.cols[int(self.V.localize(X)[-1])]
        return {"v_ratio": vr, "p_ratio": pr, "v_alarm": bool(v_alarm), "p_alarm": bool(p_alarm),
                "peer_top_channel": top}
