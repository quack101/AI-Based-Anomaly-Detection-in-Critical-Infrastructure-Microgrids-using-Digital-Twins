"""Does adding Q, or a per-meter joint P/Q/V statistic, raise A2/A3 detection?
Tuning half only (even benchmark ids for A3; A2 windows from the overall_eval
generator, even positions). Same calibration (99% clean val, untouched rows),
K = 3, FAR on clean test."""
import re, sys, types
import numpy as np, pandas as pd
sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
from attacks.library import BiasContext
from validation import detect_a2_peer as V
from validation.detect_a2_peer import load, meter_groups, untouched
from validation.a2_persist_eval import persist
from validation.overall_eval import bias_windows, bench_windows, BASE, SEED

def main():
    Xtr, ltr, cols = load("train"); Xva, lva, _ = load("val")
    clean = ltr.attacked.values == 0
    mu, sd = Xtr[clean].mean(0), Xtr[clean].std(0) + 1e-9
    S = (Xtr[clean] - mu) / sd; D = S.shape[1]
    groups = {g: own for g, own in meter_groups(cols).items() if g != "HEAD"}
    models = {}
    for g, own in groups.items():
        oth = np.setdiff1d(np.arange(D), own); A = S[:, oth]
        models[g] = (own, oth, np.linalg.solve(A.T @ A + len(A) / 1000 * np.eye(len(oth)), A.T @ S[:, own]))
    def errs(X):
        Sx = (X - mu) / sd; E = np.full((len(X), D), np.nan)
        for own, oth, B in models.values(): E[:, own] = Sx[:, own] - Sx[:, oth] @ B
        return E
    esd = np.nanstd(errs(Xtr[clean]), 0) + 1e-12
    isP = np.array([bool(re.match(r"M\d{3}_P", c)) for c in cols]); isQ = np.array([bool(re.match(r"M\d{3}_Q", c)) for c in cols])
    vdet = V.fit_default()
    def scores(X):
        Z = errs(X) / esd
        out = {"P": np.nanmax(np.abs(Z[:, isP]), 1), "PQ": np.nanmax(np.abs(Z[:, isP | isQ]), 1)}
        # per-meter chi2 over that meter's P and Q (instantaneous) -> max over meters
        chi = np.stack([np.nansum(Z[:, [j for j in own if isP[j] or isQ[j]]] ** 2, 1) for own, _, _ in models.values()], 1)
        out["meterPQ"] = np.sqrt(np.nanmax(chi, 1))
        out["V"] = np.nan_to_num(vdet.score(X))
        return out
    unt = untouched(lva.attacked.values, 30)
    sv = scores(Xva); thr = {k: np.quantile(v[unt], 0.99) for k, v in sv.items()}
    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    sc = scores(X)
    rules = {"P": ["P"], "PQ": ["PQ"], "meterPQ": ["meterPQ"], "V|P": ["V", "P"], "V|PQ": ["V", "PQ"], "V|meterPQ": ["V", "meterPQ"]}
    def alarm(s, names): return np.any([s[n] > thr[n] for n in names], axis=0)
    rng = np.random.default_rng(SEED); ctx = BiasContext(sigma_overrides={})
    _ = bias_windows("A1", rng, X, cols, ctx)                       # keep the same rng stream as overall_eval
    wins = [w for i, w in enumerate(bias_windows("A2", rng, X, cols, ctx)) if i % 2 == 0]
    wins += [w for i, w in enumerate(bench_windows(X, cols)) if i % 2 == 0 and w[0]["type"] == "A3"]
    ws = [(m, scores(seg), off) for m, seg, off in wins]
    for name, names in rules.items():
        far = persist(alarm(sc, names), 3)[30:].mean()
        res = {}
        for m, s, off in ws:
            res.setdefault(m["type"], []).append(persist(alarm(s, names), 3)[off:].any())
        print(f"{name:10s} FAR {far*100:.2f}% | " + "  ".join(f"{t} {sum(v)}/{len(v)}" for t, v in res.items()), flush=True)

if __name__ == "__main__":
    main()
