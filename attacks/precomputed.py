"""
Precomputed attacks for the live stream (A3 / A4 / A5).

A3 (load redistribution) and A4/A5 (stealthy, estimator-subspace) attacks need
a power-flow solve or the state estimator to BUILD. The injector sits on the
wire (Layer 2) and must never touch the circuit (streaming/injector.py), so
these attacks are built OFFLINE -- by validation/gen_demo_attacks.py, with the
library's own attacks.library.attack() -- the way a real attacker would plan
them with their own model of the grid. On the stream the injector only ADDS
the stored vector a_t to the readings in transit: z_delivered = z_true + a_t.
Pure arithmetic, no circuit access.

File format (.npz): timestamps (n,) int stream indices; channels (m,) str
channel ids; a (n, m) float attack values (0 = untouched).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent


@dataclass
class PrecomputedAttack:
    type: str                 # "A3" | "A4" | "A5"
    start: int                # first attacked stream index
    duration: int
    plan: str                 # .npz path, absolute or relative to the repo root
    magnitude: float = 0.0    # for the label only (A3: dP fraction, A4/A5: size in pu)
    stealthy: bool = True     # built to pass the residual (BDD) test
    note: str = ""
    pq_rel: float = None      # injector label code reads these on every spec
    _rows: dict = field(default=None, repr=False)

    def _load(self):
        if self._rows is None:
            p = Path(self.plan)
            if not p.is_absolute():
                p = BASE / p
            z = np.load(p, allow_pickle=False)
            ts, ch, a = z["timestamps"].astype(int), [str(c) for c in z["channels"]], z["a"].astype(float)
            self._rows = {int(t): {c: float(v) for c, v in zip(ch, a[i]) if v != 0.0} for i, t in enumerate(ts)}
        return self._rows

    def apply(self, measurements, t):
        """(new_measurements, info). Outside [start, start + duration) or for a
        timestamp not in the plan, the readings pass through unchanged."""
        out = dict(measurements)
        if not (self.start <= t < self.start + self.duration):
            return out, {"attacked": False, "targets": []}
        row = self._load().get(int(t), {})
        hit = [c for c in out if c in row]
        for c in hit:
            out[c] = out[c] + row[c]
        return out, {"attacked": bool(hit), "targets": hit}
