"""Existing AE-Mahalanobis detector (saved v3 weights, detect_ae_weights.npz,
same scoring as detect_mahalanobis.py maha_full, threshold = 99% clean val)
on the SAME A2 benchmark windows as a2_persist_eval.py (same seeds), with the
same any-step and 3-step persistence rules. BDD cannot run here (needs the
estimator), so this compares against the learned half of the fusion only."""
import sys, types, json
import numpy as np, pandas as pd
sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
from attacks.library import AttackSpec, BiasContext
from validation.detect_mahalanobis import ae_errors, ledoit_wolf, PARAM_KEYS
from validation.detect_ae import W
from validation.detect_a2_peer import load
from validation.a2_persist_eval import persist
import validation.a2_benchmark as B

CTX = max(B.CONTEXT, W)

def main():
    wz = np.load(B.OUT / "detect_ae_weights.npz", allow_pickle=True)
    P = {k: wz[k] for k in PARAM_KEYS}; mu_ae, sd_ae, r_c = wz["mu"], wz["sd"], wz["r_c"]
    Xtr, _, cols = load("train"); Xva, lva, _ = load("val")
    Etr = ae_errors(P, mu_ae, sd_ae, r_c, Xtr); tr = Etr[np.isfinite(Etr[:, 0])]
    mu = tr.mean(0); Sig, _, _ = ledoit_wolf(tr - mu); Minv = np.linalg.inv(Sig)
    def score(X):
        E = ae_errors(P, mu_ae, sd_ae, r_c, X); s = np.full(len(E), np.nan); f = np.isfinite(E[:, 0])
        d = E[f] - mu; s[f] = np.einsum("ij,ij->i", d @ Minv, d); return s
    sv = score(Xva); thr = float(np.quantile(sv[(lva.attacked.values == 0) & np.isfinite(sv)], 0.99))
    X = pd.read_csv(B.BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    sc_clean = score(X); out = {"threshold": thr}
    for K in (1, 3):
        a = persist(np.nan_to_num(sc_clean) > thr, K); rng = np.random.default_rng(1)
        out[f"K{K}_FAR_step"] = float(a[CTX:].mean())
        out[f"K{K}_chance"] = {d: float(np.mean([a[i:i+d].any() for i in rng.integers(CTX, len(X)-d, 4000)])) for d in (15, 60, 240)}
    ctx = BiasContext(sigma_overrides={}); rng = np.random.default_rng(B.SEED0); cells = []
    for var, qty in {"A2": ("P", "Q", "V"), "A2_PQ": ("P", "Q")}.items():
        for li, (pq, v) in enumerate(zip(B.PQ_LEVELS, B.V_LEVELS)):
            for prof in B.PROFILES:
                for dur in B.DURATIONS:
                    for k in range(B.REPS):
                        start = int(rng.integers(B.CONTEXT, len(X) - dur - 1))
                        spec = AttackSpec("A2", start, dur, prof, seed=int(rng.integers(1 << 31)),
                                          quantities=qty, pq_rel=pq, v_abs_pu=v)
                        s0 = max(start - CTX, 0)
                        seg = B.inject(X[s0:start + dur], cols, spec, ctx, s0)
                        al = np.nan_to_num(score(seg)) > thr
                        cells.append((var, f"L{li}", dur, bool(al[start - s0:].any()),
                                      bool(persist(al, 3)[start - s0:].any())))
    df = pd.DataFrame(cells, columns=["variant", "level", "duration", "hit1", "hit3"])
    for col in ("hit1", "hit3"):
        out[col + "_level"] = {f"{a} {b}": round(v, 2) for (a, b), v in df.groupby(["variant", "level"])[col].mean().items()}
        out[col + "_duration"] = {f"{a} {b}": round(v, 2) for (a, b), v in df.groupby(["variant", "duration"])[col].mean().items()}
    print(json.dumps(out, indent=1, default=float))
    json.dump(out, open(B.OUT / "a2_baseline_summary.json", "w"), indent=2, default=float)

if __name__ == "__main__":
    main()
