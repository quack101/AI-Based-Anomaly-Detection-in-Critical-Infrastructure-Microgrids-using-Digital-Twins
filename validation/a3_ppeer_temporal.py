"""
A3 follow-up: does the power peer check (validation/detect_peer_power.py)
gain from time averaging, the way the voltage check does?

Live demo run C showed A3 (10% zone redistribution) holding the P-peer score
just around its threshold for the whole hour -- a small, persistent shift.
Averaging over L steps shrinks noise by ~sqrt(L) but not the shift.
(An earlier test used L = 5 only, on the large fdia_v1 A3 windows: no gain.)

Variants of the P statistic, all on the same ridge peer predictions:
  P1     current: max_c |e_c,t| / sd_c                         (no averaging)
  PT     clipped (+-3) normalised errors, trailing means L = 1, 15, 30,
         each standardised on clean train; max over channels and L
         (exactly the V-peer design, applied to P channels)
  PTpair PT plus a "load moved" statistic: for each L, mean of the largest
         positive and the largest negative standardised trailing mean
         (A3 pushes one group up and another down by the same total)
Each is thresholded at 99% of clean val (untouched 30-row windows), alarm
after 3 consecutive steps, and combined as V | P* | AE. Scored on the
benchmarks of validation/overall_eval.py (tune = even ids, report = odd).

RESULT (reports/a3_improvement_attempt.md): at equal false-alarm rate none of
these adds A3 detection; the missed A3 windows sit below the meters'
measurement-noise floor.

Run: python -m validation.a3_ppeer_temporal
"""

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
from attacks.library import BiasContext                                    # noqa: E402
from validation import detect_a2_peer as V                                 # noqa: E402
from validation import detect_peer_power as Pp                             # noqa: E402
from validation.a2_persist_eval import persist                             # noqa: E402
from validation.detect_a2_peer import trailing_mean, untouched             # noqa: E402
from validation.eval_attack_bench import ae_maha                           # noqa: E402
from validation.overall_eval import bias_windows, bench_windows, CTX, SEED  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CLIP, WINDOWS = 3.0, (1, 15, 30)


class PeerPowerTemporal:
    """P-peer with clipped trailing means (pair=True adds the up/down statistic)."""

    def __init__(self, base, X_clean, pair=False, clip=CLIP, windows=WINDOWS):
        self.b, self.pair, self.clip, self.windows = base, pair, clip, tuple(windows)
        self.cols, self.p_idx = base.cols, base.p_idx
        Zc = self._z(X_clean)
        self.mu_L, self.sd_L = {}, {}
        for L in self.windows:
            T = trailing_mean(Zc, L)[L - 1:]
            self.mu_L[L], self.sd_L[L] = T.mean(0), T.std(0) + 1e-12
        if pair:
            self.pmu, self.psd = {}, {}
            for L in self.windows:
                pr = self._pair_raw(Zc, L)[L - 1:]
                self.pmu[L], self.psd[L] = pr.mean(), pr.std() + 1e-12

    def _z(self, X):
        return np.clip(self.b._errors_raw(X) / self.b.err_sd, -self.clip, self.clip)

    def _std(self, Zc, L):
        return (trailing_mean(Zc, L) - self.mu_L[L]) / self.sd_L[L]

    def _pair_raw(self, Zc, L):
        T = self._std(Zc, L)
        with np.errstate(all="ignore"):
            return 0.5 * (np.nanmax(T, 1) - np.nanmin(T, 1))

    def score(self, X):
        Zc = self._z(X)
        parts = [np.nanmax(np.abs(self._std(Zc, L)), 1) for L in self.windows]
        if self.pair:
            # put the pair statistic on the same scale as a single channel's |z|
            parts += [(self._pair_raw(Zc, L) - self.pmu[L]) / self.psd[L] for L in self.windows]
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanmax(np.stack(parts), 0))

    def calibrate(self, X_val, att_val, q=0.99):
        self.thr = float(np.nanquantile(self.score(X_val)[untouched(att_val, max(self.windows))], q))
        return self.thr


def main():
    t0 = time.time()
    vdet, pdet = V.fit_default(), Pp.fit_default()
    Xtr, ltr, cols = V.load("train")
    Xva, lva, _ = V.load("val")
    clean_tr = Xtr[ltr["attacked"].to_numpy() == 0]
    att_va = lva["attacked"].to_numpy()
    pT = PeerPowerTemporal(pdet, clean_tr); pT.calibrate(Xva, att_va)
    pTP = PeerPowerTemporal(pdet, clean_tr, pair=True); pTP.calibrate(Xva, att_va)
    pL = {L: PeerPowerTemporal(pdet, clean_tr, windows=(L,)) for L in (10, 15, 30, 60)}
    for d in pL.values():
        d.calibrate(Xva, att_va)
    ae_score, ae_thr = ae_maha()
    print(f"fitted ({time.time() - t0:.0f}s): P1 thr {pdet.thr:.3f}, PT thr {pT.thr:.3f}, PTpair thr {pTP.thr:.3f}",
          flush=True)

    def ratios(Xs):
        return {"V": np.nan_to_num(vdet.score(Xs)) / vdet.thr, "AE": ae_score(Xs) / ae_thr,
                "P1": np.nan_to_num(pdet.score(Xs)) / pdet.thr, "PT": pT.score(Xs) / pT.thr,
                "PTpair": pTP.score(Xs) / pTP.thr,
                **{f"PL{L}": d.score(Xs) / d.thr for L, d in pL.items()}}

    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    Rc = ratios(X)
    rng = np.random.default_rng(SEED)
    ctx = BiasContext(sigma_overrides={})
    wins = bias_windows("A1", rng, X, cols, ctx) + bias_windows("A2", rng, X, cols, ctx) + bench_windows(X, cols)
    Rw = [(meta, {k: v[off:] for k, v in ratios(seg).items()}) for meta, seg, off in wins]
    print(f"{len(wins)} windows scored ({time.time() - t0:.0f}s)", flush=True)

    systems = {"P only: P1 (current)": ["P1"], "P only: PT": ["PT"], "P only: PTpair": ["PTpair"],
               "V|P1|AE (current system)": ["V", "P1", "AE"], "V|PT|AE": ["V", "PT", "AE"],
               "V|PTpair|AE": ["V", "PTpair", "AE"]}
    rng2 = np.random.default_rng(1)
    summ, rows_all = {}, []
    for name, keys in systems.items():
        def alarm(R):
            return persist(np.any([R[k] > 1 for k in keys], axis=0), 3)
        al = alarm(Rc)
        far = float(al[CTX:].mean())
        chance = {d: round(100 * float(np.mean([al[i:i + d].any() for i in rng2.integers(CTX, len(X) - d, 2000)])), 1)
                  for d in (15, 60, 240)}
        rows = [{**meta, "half": "tune" if i % 2 == 0 else "report", "hit": bool(alarm(R).any()), "system": name}
                for i, (meta, R) in enumerate(Rw)]
        df = pd.DataFrame(rows); rows_all.append(df)
        rep = df[df.half == "report"]
        by_type = {t: round(100 * g.hit.mean(), 1) for t, g in rep.groupby("type")}
        a3 = rep[rep.type == "A3"]
        summ[name] = {"FAR_%": round(100 * far, 2), "chance_%": chance, "report_by_type_%": by_type,
                      "report_mean_%": round(float(np.mean(list(by_type.values()))), 1),
                      "tune_mean_%": round(100 * float(np.mean([g.hit.mean() for _, g in df[df.half == "tune"].groupby("type")])), 1),
                      "A3_report_by_level_%": {l: round(100 * g.hit.mean()) for l, g in a3.groupby("level")},
                      "A3_report_by_duration_%": {str(d): round(100 * g.hit.mean()) for d, g in a3.groupby("duration")},
                      "A3_report_by_access_%": {a: round(100 * g.hit.mean()) for a, g in a3.groupby("access")}}
        s = summ[name]
        print(f"{name:26s} FAR {s['FAR_%']:5.2f}%  chance15/60/240 {chance}  tune {s['tune_mean_%']:5.1f}  "
              f"report {s['report_mean_%']:5.1f}  {by_type}\n{'':26s} A3 level {s['A3_report_by_level_%']} "
              f"dur {s['A3_report_by_duration_%']} access {s['A3_report_by_access_%']}", flush=True)
    # matched false-alarm comparison: scale the extra P statistic's threshold so the
    # whole system's clean-test FAR equals the current system's
    target = summ["V|P1|AE (current system)"]["FAR_%"] / 100
    base = np.any([Rc[k] > 1 for k in ("V", "P1", "AE")], axis=0)
    summ["matched"] = {}
    for extra in ["PT"] + [f"PL{L}" for L in pL]:
        lo, hi = 1.0, 20.0
        for _ in range(40):
            m = (lo + hi) / 2
            far = persist(base | (Rc[extra] > m), 3)[CTX:].mean()
            lo, hi = (m, hi) if far > target else (lo, m)
        m = hi
        far = float(persist(base | (Rc[extra] > m), 3)[CTX:].mean())
        hits = [{**meta, "half": "tune" if i % 2 == 0 else "report",
                 "hit": bool(persist(np.any([R[k] > 1 for k in ("V", "P1", "AE")], 0) | (R[extra] > m), 3).any())}
                for i, (meta, R) in enumerate(Rw)]
        df = pd.DataFrame(hits); rep = df[df.half == "report"]; tun = df[df.half == "tune"]
        bt = {t: round(100 * g.hit.mean(), 1) for t, g in rep.groupby("type")}
        a3 = rep[rep.type == "A3"]
        summ["matched"][extra] = {"mult": m, "FAR_%": round(100 * far, 2), "report_by_type_%": bt,
                                  "tune_A3_%": round(100 * tun[tun.type == "A3"].hit.mean(), 1),
                                  "A3_level": {l: round(100 * g.hit.mean()) for l, g in a3.groupby("level")},
                                  "A3_dur": {str(d): round(100 * g.hit.mean()) for d, g in a3.groupby("duration")}}
        print(f"MATCHED V|P1|AE + {extra:6s} x{m:.2f}  FAR {100*far:.2f}%  report {bt}  tune A3 "
              f"{summ['matched'][extra]['tune_A3_%']}  A3 lvl {summ['matched'][extra]['A3_level']} "
              f"dur {summ['matched'][extra]['A3_dur']}", flush=True)
    json.dump(summ, open(OUT / "a3_ppeer_temporal_summary.json", "w"), indent=1)
    pd.concat(rows_all).to_csv(OUT / "a3_ppeer_temporal_windows.csv", index=False)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
