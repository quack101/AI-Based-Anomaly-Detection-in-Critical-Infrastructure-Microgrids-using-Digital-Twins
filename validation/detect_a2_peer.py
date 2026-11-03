"""
A2 (single-meter bias) detector: per-meter peer-consistency test on voltage.

Why A2 is missed today
----------------------
An fdia_v1 A2 attack biases ONE meter's P, Q and V. Measured on the test
split, the bias is 0.6-2x the channel's minute-to-minute noise (V: 0.005-0.02
pu against ~0.005 pu meter noise). The AE-Mahalanobis score pools all 168
channels, so 3 biased channels are diluted; v4's per-channel AE CUSUM caught
A2 1/3 at ~5% test FAR.

Method
------
1. Peer prediction. For every meter, a ridge regression predicts its channels
   from all OTHER meters' channels, fitted on clean train rows (closed form,
   deterministic). A single-meter attack cannot move its own prediction, so
   the bias appears whole in the prediction error. Feeder-wide changes are
   predicted away.
2. Voltage channels only. The V peer error is white noise at the meter-noise
   level (lag-1 autocorrelation ~0.01; its sd matches the white-noise
   prediction for trailing means within ~3%). P/Q peer errors are strongly
   autocorrelated (0.6-0.8) and the A2 P/Q bias is only 0.2-0.7 of their sd,
   so they add false alarms, not A2 signal.
3. Spike cap. Each step's normalised V error is clipped to +/-CLIP before
   averaging, so an isolated large error (e.g. a tap change) cannot raise a
   window mean. A2's per-step bias (<=~2 sd) passes the cap unchanged.
4. Temporal averaging. A2 is a sustained bias (15-240 steps): the trailing
   mean over L steps shrinks white noise by 1/sqrt(L) and keeps the bias.
   Score_t = max over V channels and L in WINDOWS of
   |trailing mean - mu_L| / sd_L (mu_L, sd_L from clean train).
5. Leak-free calibration. A trailing window that ends on a clean row just
   after an attack still contains attacked rows; the threshold is therefore
   the QUANTILE of the score over clean val rows whose whole trailing window
   (max(WINDOWS) rows) is clean.

numpy only; no estimator runs; new file: no existing detector is changed.

Run: python -m validation.detect_a2_peer
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
FDIA = OUT.parent / "dataset" / "fdia_v1"
RIDGE = 1.0
CLIP = 3.0
WINDOWS = (5, 15, 30)
QUANTILE = 0.99          # pre-registered operating point (~1% FAR on clean val)
_METER = re.compile(r"^(M\d{3})_")


def load(split):
    z = pd.read_csv(FDIA / f"{split}_z.csv")
    lab = pd.read_csv(FDIA / f"{split}_labels.csv")
    cols = [c for c in z.columns if c != "timestamp"]
    return z[cols].to_numpy(float), lab, cols


def is_v(c):
    return c.endswith("_V") or "_V_" in c


def meter_groups(cols):
    groups = {}
    for j, c in enumerate(cols):
        m = _METER.match(c)
        groups.setdefault(m.group(1) if m else "HEAD", []).append(j)
    return groups


def trailing_mean(E, L):
    """Mean of rows t-L+1..t (NaN for the first L-1 rows)."""
    if L == 1:
        return E.copy()
    c = np.cumsum(np.nan_to_num(E), axis=0)
    out = np.full_like(E, np.nan)
    out[L - 1:] = c[L - 1:] - np.vstack([np.zeros((1, E.shape[1])), c[:-L]])
    return out / L


def untouched(att, L):
    """True where the row and the L-1 rows before it are all clean."""
    return np.convolve(np.asarray(att, float), np.ones(L), mode="full")[:len(att)] == 0


class PeerVoltageDetector:
    def __init__(self, ridge=RIDGE, clip=CLIP, windows=WINDOWS):
        self.ridge, self.clip, self.windows = ridge, clip, tuple(windows)

    def fit(self, X_clean, cols):
        self.cols = cols
        self.mu_x = X_clean.mean(0)
        self.sd_x = X_clean.std(0) + 1e-9
        S = (X_clean - self.mu_x) / self.sd_x
        D = S.shape[1]
        self.v_idx = [j for j, c in enumerate(cols) if c.startswith("M") and is_v(c)]
        self.models = []
        for g, own in meter_groups(cols).items():
            v_own = [j for j in own if j in self.v_idx]
            if g == "HEAD" or not v_own:
                continue
            others = np.setdiff1d(np.arange(D), own)
            A = S[:, others]
            G = A.T @ A + self.ridge * len(A) / 1000.0 * np.eye(len(others))
            self.models.append((v_own, others, np.linalg.solve(G, A.T @ S[:, v_own])))
        E = self._errors_raw(X_clean)
        self.err_sd = np.nanstd(E, 0) + 1e-12
        Zc = np.clip(E / self.err_sd, -self.clip, self.clip)
        self.mu_L, self.sd_L = {}, {}
        for L in self.windows:
            T = trailing_mean(Zc, L)[L - 1:]
            self.mu_L[L], self.sd_L[L] = T.mean(0), T.std(0) + 1e-12
        return self

    def _errors_raw(self, X):
        S = (X - self.mu_x) / self.sd_x
        E = np.empty((len(X), len(self.v_idx)))
        pos = {j: k for k, j in enumerate(self.v_idx)}
        for v_own, others, B in self.models:
            E[:, [pos[j] for j in v_own]] = S[:, v_own] - S[:, others] @ B
        return E

    def channel_stats(self, X):
        """{L: (T, n_V) normalised |trailing-mean error|}."""
        Zc = np.clip(self._errors_raw(X) / self.err_sd, -self.clip, self.clip)
        return {L: np.abs((trailing_mean(Zc, L) - self.mu_L[L]) / self.sd_L[L]) for L in self.windows}

    def score(self, X):
        st = self.channel_stats(X)
        return np.nanmax(np.stack([np.nanmax(st[L], axis=1) for L in self.windows]), axis=0)

    def localize(self, X):
        """Index into self.cols of the V channel with the largest statistic, per row."""
        st = self.channel_stats(X)
        stacked = np.stack([st[L] for L in self.windows])          # (nL, T, nV)
        best = np.nanargmax(np.nan_to_num(stacked.max(0), nan=-1), axis=1)
        return np.array(self.v_idx)[best]

    def calibrate(self, X_val, att_val, q=QUANTILE):
        s = self.score(X_val)
        self.thr = float(np.nanquantile(s[untouched(att_val, max(self.windows))], q))
        return self.thr


def fit_default():
    Xtr, ltr, cols = load("train")
    Xva, lva, _ = load("val")
    det = PeerVoltageDetector().fit(Xtr[ltr["attacked"].to_numpy() == 0], cols)
    det.calibrate(Xva, lva["attacked"].to_numpy())
    return det


def main():
    from validation.detect_robustness import ev
    det = fit_default()
    Xte, lte, cols = load("test")
    ate = lte["attacked"].to_numpy()
    st = det.score(Xte)
    r = ev(st, lte, det.thr)
    far_u = float(np.mean(st[untouched(ate, max(det.windows))] > det.thr))
    # localization: on detected A2 windows, is the top channel one of the targets?
    loc = det.localize(Xte)
    hits = tot = 0
    for wid, g in lte[lte["attacked"] == 1].groupby("window_id"):
        rows = g.index.to_numpy()
        alarm = rows[st[rows] > det.thr]
        if len(alarm):
            tot += 1
            top = pd.Series(loc[alarm]).mode().iloc[0]
            meter = cols[top].split("_")[0]
            hits += int(any(t.startswith(meter + "_") for t in str(g.iloc[0]["targets"]).split(";")))
    out = {"threshold": det.thr, "quantile": QUANTILE, "windows": det.windows, "clip": det.clip,
           "win": r["win"], "n": r["n"], "A1": r["A1"], "A2": r["A2"], "A3": r["A3"],
           "FAR_all_clean": r["FAR"], "FAR_untouched": far_u,
           "per_ts_AUC": r["per_ts_AUC"], "window_AUC": r["window_AUC"],
           "meter_localization": hits / tot if tot else None}
    print(json.dumps(out, indent=2, default=float))
    json.dump(out, open(OUT / "detect_a2_peer_summary.json", "w"), indent=2, default=float)
    np.savez_compressed(OUT / "detect_a2_peer_scores.npz", test=st, thr=det.thr)


if __name__ == "__main__":
    main()
