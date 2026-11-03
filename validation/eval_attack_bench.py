"""
Evaluate detectors on dataset/attack_bench (A3 / A4 / A5 windows made by
validation/gen_attack_bench.py with the library's attack() and OpenDSS).

Each window: its attack vectors a (duration x 168) are added to the clean
test telemetry rows they were computed on; the detector sees CONTEXT clean
rows before the window (its trailing history) and the attacked rows.
Detected = any alarm inside the window. Rules: any step (K=1) and K=3
consecutive steps (the recommended rule). Chance = probability that a CLEAN
window of the same length alarms with the same rule.

Detectors:
  V-peer     validation/detect_a2_peer.py   (voltage, A2 detector)
  P-peer     validation/detect_peer_power.py (active power, A3 detector)
  V or P     alarm if either alarms (each at its own threshold)
  AE-Maha    existing AE-Mahalanobis (saved v3 weights, detect_mahalanobis.py),
             the learned half of the current fusion. BDD itself needs the
             estimator and is not re-run here; for A4/A5 the generator's
             literature (and phase8 reports) show it cannot see them by design.

Run: python -m validation.eval_attack_bench
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation import detect_a2_peer as V
from validation import detect_peer_power as Pp
from validation.a2_persist_eval import persist
from validation.detect_ae import W
from validation.detect_mahalanobis import ae_errors, ledoit_wolf, PARAM_KEYS

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
BENCH = BASE / "dataset" / "attack_bench"
CONTEXT = 30


def ae_maha():
    wz = np.load(OUT / "detect_ae_weights.npz", allow_pickle=True)
    P = {k: wz[k] for k in PARAM_KEYS}; mu_ae, sd_ae, r_c = wz["mu"], wz["sd"], wz["r_c"]
    Xtr, _, cols = V.load("train"); Xva, lva, _ = V.load("val")
    tr = ae_errors(P, mu_ae, sd_ae, r_c, Xtr); tr = tr[np.isfinite(tr[:, 0])]
    mu = tr.mean(0); Sig, _, _ = ledoit_wolf(tr - mu); Minv = np.linalg.inv(Sig)

    def score(X):
        E = ae_errors(P, mu_ae, sd_ae, r_c, X); s = np.full(len(E), np.nan); f = np.isfinite(E[:, 0])
        d = E[f] - mu; s[f] = np.einsum("ij,ij->i", d @ Minv, d); return np.nan_to_num(s)
    sv = score(Xva)
    thr = float(np.quantile(sv[(lva.attacked.values == 0)], 0.99))
    return score, thr


def main():
    vdet, pdet = V.fit_default(), Pp.fit_default()
    ae_score, ae_thr = ae_maha()
    cols = vdet.cols
    bench_cols = json.load(open(BENCH / "channels.json"))
    order = [bench_cols.index(c) for c in cols]
    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)

    def alarms(Xs):
        return {"V-peer": vdet.score(Xs) > vdet.thr, "P-peer": pdet.score(Xs) > pdet.thr,
                "AE-Maha": ae_score(Xs) > ae_thr}

    # chance + FAR on the clean test series
    clean = alarms(X); clean["V or P"] = clean["V-peer"] | clean["P-peer"]
    chance, far = {}, {}
    rng = np.random.default_rng(1)
    for name, a in clean.items():
        for K in (1, 3):
            al = persist(a, K)
            far[f"{name} K{K}"] = float(al[CONTEXT:].mean())
            for d in (15, 60, 240):
                st = rng.integers(CONTEXT, len(X) - d, 3000)
                chance[f"{name} K{K} d{d}"] = float(np.mean([al[i:i + d].any() for i in st]))

    rows = []
    wins = [json.loads(l) for l in open(BENCH / "windows.jsonl")]
    wins = {w["id"]: w for w in wins if w.get("ok")}
    for wid, w in sorted(wins.items()):
        a = np.load(BENCH / f"a_{wid}.npy")[:, order]
        s, d = w["start"], w["duration"]
        s0 = max(0, s - max(CONTEXT, W))
        seg = X[s0:s + d].copy(); seg[s - s0:] += a
        al = alarms(seg); al["V or P"] = al["V-peer"] | al["P-peer"]
        row = {k: w[k] for k in ("id", "type", "magnitude", "access", "profile", "duration")}
        for name, x in al.items():
            for K in (1, 3):
                row[f"{name} K{K}"] = bool(persist(x, K)[s - s0:].any())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "attack_bench_eval.csv", index=False)

    dets = ["V-peer", "P-peer", "V or P", "AE-Maha"]
    summ = {"n_windows": len(df), "FAR_clean_test": far, "chance": chance, "by": {}}
    for t, g in df.groupby("type"):
        for key in (["magnitude"], ["duration"], ["access"] if t == "A3" else ["magnitude"]):
            for val, gg in g.groupby(key[0]):
                summ["by"][f"{t} {key[0]}={val}"] = {f"{d} K{K}": f"{int(gg[f'{d} K{K}'].sum())}/{len(gg)}"
                                                    for d in dets for K in (1, 3)} | {"n": len(gg)}
        summ["by"][f"{t} ALL"] = {f"{d} K{K}": f"{int(g[f'{d} K{K}'].sum())}/{len(g)}" for d in dets for K in (1, 3)}
    json.dump(summ, open(OUT / "attack_bench_eval_summary.json", "w"), indent=1, default=float)
    print(f"windows evaluated: {len(df)}  ({df.type.value_counts().to_dict()})")
    print("FAR per step (clean test):", {k: f"{v*100:.2f}%" for k, v in far.items()})
    print("chance (clean window alarms):", {k: f"{v*100:.0f}%" for k, v in chance.items() if "K3" in k})
    for k, v in summ["by"].items():
        print(f"{k:28s}", "  ".join(f"{d}: {v[d + ' K3']}" for d in dets))


if __name__ == "__main__":
    main()
