"""
A3 (load redistribution) detector: per-meter peer-consistency test on ACTIVE POWER.

An fdia_v1 A3 window moves 2-10% of total metered P from 4 metered loads to 4
others and makes every channel consistent with the power flow, so the
residual (BDD) stays small. But the 8 targeted meters' P moves by 30-580% of
their normal value (measured on fdia_v1 val+test), which is implausible
given what the OTHER meters show.

Method (same family as validation/detect_a2_peer.py):
  - ridge regression predicts every meter's P channel(s) from all OTHER
    meters' channels, fitted on clean train (closed form, deterministic);
  - score_t = max over P channels of |prediction error| / (clean-train sd);
  - threshold: 99th percentile over clean val rows whose 30-row trailing
    window is clean (same calibration rule as the A2 detector);
  - alarm after K = 3 consecutive steps above it (validation/a2_persist_eval.py).
No temporal averaging: A3 changes are many sd per step, and P peer errors are
autocorrelated (lag-1 ~0.65), which makes trailing means heavy-tailed.
Tried and NOT kept (6 fdia_v1 A3 windows, no gain): AR(1) pre-whitening,
sum of the top-k squared errors (k = 2/4/8), trailing means (L = 5).

numpy only; new file.
"""

import re

import numpy as np

from validation.detect_a2_peer import load, meter_groups, untouched

RIDGE = 1.0
QUANTILE = 0.99
CAL_WINDOW = 30
_P = re.compile(r"^M\d{3}_P(?:_[ABC])?$")


class PeerPowerDetector:
    def __init__(self, ridge=RIDGE):
        self.ridge = ridge

    def fit(self, X_clean, cols):
        self.cols = cols
        self.mu_x, self.sd_x = X_clean.mean(0), X_clean.std(0) + 1e-9
        S = (X_clean - self.mu_x) / self.sd_x
        D = S.shape[1]
        self.p_idx = [j for j, c in enumerate(cols) if _P.match(c)]
        self.models = []
        for g, own in meter_groups(cols).items():
            p_own = [j for j in own if j in self.p_idx]
            if g == "HEAD" or not p_own:
                continue
            others = np.setdiff1d(np.arange(D), own)
            A = S[:, others]
            G = A.T @ A + self.ridge * len(A) / 1000.0 * np.eye(len(others))
            self.models.append((p_own, others, np.linalg.solve(G, A.T @ S[:, p_own])))
        self.err_sd = np.nanstd(self._errors_raw(X_clean), 0) + 1e-12
        return self

    def _errors_raw(self, X):
        S = (X - self.mu_x) / self.sd_x
        E = np.empty((len(X), len(self.p_idx)))
        pos = {j: k for k, j in enumerate(self.p_idx)}
        for p_own, others, B in self.models:
            E[:, [pos[j] for j in p_own]] = S[:, p_own] - S[:, others] @ B
        return E

    def channel_stats(self, X):
        return np.abs(self._errors_raw(X) / self.err_sd)

    def score(self, X):
        return np.nanmax(self.channel_stats(X), axis=1)

    def localize(self, X):
        return np.array(self.p_idx)[np.nanargmax(self.channel_stats(X), axis=1)]

    def calibrate(self, X_val, att_val, q=QUANTILE):
        s = self.score(X_val)
        self.thr = float(np.nanquantile(s[untouched(att_val, CAL_WINDOW)], q))
        return self.thr


def fit_default():
    Xtr, ltr, cols = load("train")
    Xva, lva, _ = load("val")
    det = PeerPowerDetector().fit(Xtr[ltr["attacked"].to_numpy() == 0], cols)
    det.calibrate(Xva, lva["attacked"].to_numpy())
    return det


def main():
    from validation.detect_robustness import ev
    from validation.a2_persist_eval import persist
    det = fit_default()
    for split in ("val", "test"):
        X, lab, _ = load(split)
        r = ev(persist(det.score(X) > det.thr, 3).astype(float), lab, 0.5)
        print(f"{split}: win {r['win']}/{r['n']} A1 {r['A1']} A2 {r['A2']} A3 {r['A3']} FAR {r['FAR']*100:.2f}%")


if __name__ == "__main__":
    main()
