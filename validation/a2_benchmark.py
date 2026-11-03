"""
A2 benchmark: many single-meter attacks, so A2 detection can be measured
with more than the 3 A2 windows in the fdia_v1 test split.

Attacks are made by attacks/library.py's apply_bias_timestep (the same A2
code path as the fdia_v1 builder and the live injector; OpenDSS is not
needed for A1/A2), with the fdia_v1 levels, profiles and durations:
  levels    PQ_LEVELS = 0.05/0.10/0.20 (relative), V_LEVELS = 0.005/0.01/0.02 pu
  profiles  step / ramp / pulse;  durations 15 / 60 / 240 steps
Base series: the CLEAN test telemetry (simulation/phase6_corpus/test_telemetry.csv,
identical to fdia_v1 test_z on its clean rows). Each window is injected on
its own copy of a segment, so windows never overlap.

Variants (each REPS windows per level x profile x duration cell, random meter):
  A2        fdia_v1 A2: one meter's P, Q and V, one sign
  A2_V      voltage channel only (Zideh & Solanki 2023 attack one V channel)
  A2_PQ     P and Q only, V untouched (the voltage detector's blind spot)
  ZS        Zideh & Solanki-matched: one V channel, step, |alpha| = 1-5% of V

Scored detectors: the peer-voltage detector (validation/detect_a2_peer.py).
A window counts as detected if any row in it scores above the threshold
calibrated on clean val (q = 0.99). FAR = alarm rate on the whole clean test
series.

Run: python -m validation.a2_benchmark
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))  # A2 path never uses the circuit
from attacks.library import AttackSpec, BiasContext, apply_bias_timestep           # noqa: E402
from validation.detect_a2_peer import fit_default, untouched                       # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PQ_LEVELS = [0.05, 0.10, 0.20]
V_LEVELS = [0.005, 0.01, 0.02]
PROFILES = ["step", "ramp", "pulse"]
DURATIONS = [15, 60, 240]
REPS = 5
ZS_ALPHAS = [0.01, 0.02, 0.03, 0.05]
ZS_REPS = 20
CONTEXT = 30          # rows before the window the detector needs (max trailing window)
SEED0 = 20261104


def inject(seg, cols, spec, ctx, offset):
    """Apply spec to a (rows x channels) segment whose row 0 is stream index offset."""
    out = seg.copy()
    for i in range(len(seg)):
        row = dict(zip(cols, seg[i]))
        new, info = apply_bias_timestep(row, spec, offset + i, ctx)
        if info["attacked"]:
            out[i] = [new[c] for c in cols]
    return out


def run_cell(det, X, cols, spec_fn, n, rng, dur):
    hits = []
    for k in range(n):
        start = int(rng.integers(CONTEXT, len(X) - dur - 1))
        spec = spec_fn(start, k)
        seg = X[start - CONTEXT:start + dur]
        s = det.score(inject(seg, cols, spec, ctx_global, start - CONTEXT))
        hits.append(bool(np.nanmax(s[CONTEXT:]) > det.thr))
    return hits


def main():
    global ctx_global
    ctx_global = BiasContext(sigma_overrides={})   # level-mode A2 uses train caps, not sigmas
    det = fit_default()
    cols = det.cols
    X = pd.read_csv(BASE / "simulation" / "phase6_corpus" / "test_telemetry.csv")[cols].to_numpy(float)
    s_clean = det.score(X)
    far = float(np.mean(s_clean[CONTEXT:] > det.thr))
    print(f"threshold {det.thr:.3f}; FAR on clean test {far*100:.2f}%", flush=True)

    rows = []
    rng = np.random.default_rng(SEED0)
    variants = {"A2": ("P", "Q", "V"), "A2_V": ("V",), "A2_PQ": ("P", "Q")}
    for var, qty in variants.items():
        for li, (pq, v) in enumerate(zip(PQ_LEVELS, V_LEVELS)):
            for prof in PROFILES:
                for dur in DURATIONS:
                    fn = lambda start, k: AttackSpec("A2", start, dur, prof, seed=int(rng.integers(1 << 31)),
                                                     quantities=qty, pq_rel=pq, v_abs_pu=v)
                    hits = run_cell(det, X, cols, fn, REPS, rng, dur)
                    rows.append({"variant": var, "level": f"L{li}", "pq_rel": pq, "v_abs_pu": v,
                                 "profile": prof, "duration": dur, "n": REPS, "detected": int(sum(hits))})
            print(var, f"L{li}", "done", flush=True)

    # Zideh & Solanki-matched: one V channel, constant multiplicative bias alpha*|V| ~ alpha pu
    for a in ZS_ALPHAS:
        fn = lambda start, k: AttackSpec("A2", start, 15, "step", seed=int(rng.integers(1 << 31)),
                                         quantities=("V",), v_abs_pu=a * 1.0)
        for dur in (15, 60):
            fn2 = lambda start, k, d=dur: AttackSpec("A2", start, d, "step", seed=int(rng.integers(1 << 31)),
                                                     quantities=("V",), v_abs_pu=a * 1.0)
            hits = run_cell(det, X, cols, fn2, ZS_REPS, rng, dur)
            rows.append({"variant": "ZS", "level": f"{int(a*100)}%", "pq_rel": None, "v_abs_pu": a,
                         "profile": "step", "duration": dur, "n": ZS_REPS, "detected": int(sum(hits))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2_benchmark_cells.csv", index=False)

    summ = {"threshold": det.thr, "FAR_clean_test": far, "by_variant_level": {}, "by_variant_duration": {},
            "by_variant_profile": {}}
    for keys, name in [(["variant", "level"], "by_variant_level"), (["variant", "duration"], "by_variant_duration"),
                       (["variant", "profile"], "by_variant_profile")]:
        g = df.groupby(keys)[["detected", "n"]].sum()
        summ[name] = {" ".join(map(str, k)): f"{int(r.detected)}/{int(r.n)} ({100*r.detected/r.n:.0f}%)"
                      for k, r in g.iterrows()}
    tot = df[df.variant == "A2"][["detected", "n"]].sum()
    summ["A2_overall"] = f"{int(tot.detected)}/{int(tot.n)} ({100*tot.detected/tot.n:.0f}%)"
    print(json.dumps(summ, indent=2, default=float))
    json.dump(summ, open(OUT / "a2_benchmark_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
