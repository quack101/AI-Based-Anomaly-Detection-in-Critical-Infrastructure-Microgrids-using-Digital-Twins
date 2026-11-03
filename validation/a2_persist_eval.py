"""Persistence rule for the peer-voltage detector: alarm only when the score
stays above the threshold for K consecutive steps. Reports, per K: per-step
FAR on clean test, the chance that a CLEAN window of each duration alarms,
fdia_v1 test results, and the A2 benchmark (fresh run, same seeds)."""
import sys, types, json
import numpy as np, pandas as pd
sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
from attacks.library import AttackSpec, BiasContext
from validation.detect_a2_peer import fit_default, load, untouched
from validation.detect_robustness import ev
import validation.a2_benchmark as B

def persist(alarm, K):
    if K == 1: return alarm.copy()
    run = np.zeros(len(alarm), int); c = 0
    for i, a in enumerate(alarm):
        c = c + 1 if a else 0; run[i] = c
    return run >= K

def main():
    det = fit_default(); cols = det.cols
    X = pd.read_csv(B.BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    s_clean = det.score(X)
    Xte, lte, _ = load("test"); s_te = det.score(Xte)
    ctx = BiasContext(sigma_overrides={})
    out = {}
    for K in (1, 3, 5, 10):
        a_clean = persist(s_clean > det.thr, K)
        far = float(a_clean[30:].mean()); rng = np.random.default_rng(1); chance = {}
        for d in (15, 60, 240):
            st = rng.integers(30, len(X) - d, 4000); chance[d] = float(np.mean([a_clean[i:i+d].any() for i in st]))
        r = ev(persist(s_te > det.thr, K).astype(float), lte, 0.5)
        # benchmark
        rng = np.random.default_rng(B.SEED0); cells = []
        for var, qty in {"A2": ("P","Q","V"), "A2_PQ": ("P","Q")}.items():
            for li, (pq, v) in enumerate(zip(B.PQ_LEVELS, B.V_LEVELS)):
                for prof in B.PROFILES:
                    for dur in B.DURATIONS:
                        for k in range(B.REPS):
                            start = int(rng.integers(B.CONTEXT, len(X) - dur - 1))
                            spec = AttackSpec("A2", start, dur, prof, seed=int(rng.integers(1 << 31)),
                                              quantities=qty, pq_rel=pq, v_abs_pu=v)
                            seg = X[start - B.CONTEXT:start + dur]
                            sc = det.score(B.inject(seg, cols, spec, ctx, start - B.CONTEXT))
                            hit = bool(persist(sc > det.thr, K)[B.CONTEXT:].any())
                            cells.append((var, f"L{li}", dur, hit))
        df = pd.DataFrame(cells, columns=["variant", "level", "duration", "hit"])
        lv = df.groupby(["variant", "level"]).hit.mean().round(2).to_dict()
        du = df.groupby(["variant", "duration"]).hit.mean().round(2).to_dict()
        out[K] = {"FAR_step": far, "chance_window": chance,
                  "fdia_test": {k: r[k] for k in ("win", "A1", "A2", "A3", "window_AUC")},
                  "bench_level": {f"{a} {b}": v for (a, b), v in lv.items()},
                  "bench_duration": {f"{a} {b}": v for (a, b), v in du.items()}}
        print(f"K={K}: FAR/step {far*100:.2f}% chance(15/60/240) {[round(chance[d]*100) for d in (15,60,240)]}% | "
              f"fdia win {r['win']}/18 A1 {r['A1']} A2 {r['A2']} A3 {r['A3']}", flush=True)
        print("    level:", {f"{a} {b}": v for (a, b), v in lv.items()}, flush=True)
        print("    dur  :", {f"{a} {b}": v for (a, b), v in du.items()}, flush=True)
    json.dump(out, open(B.OUT / "a2_persist_summary.json", "w"), indent=2, default=float)

if __name__ == "__main__":
    main()
