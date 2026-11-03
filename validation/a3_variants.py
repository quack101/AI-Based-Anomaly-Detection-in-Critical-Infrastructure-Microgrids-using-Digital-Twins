"""A3 detector variants, tuned on EVEN benchmark ids, reported on ODD ids.
Base: P-peer per-channel |z| (detect_peer_power). Variants add temporal
averaging of the signed z (A3 is a sustained shift) and/or a group statistic
(A3 moves 8 meters). All thresholds = 99% of clean val (untouched rows);
K = 3 persistence; chance and FAR measured on the clean test series."""
import json
import numpy as np, pandas as pd
from validation import detect_peer_power as Pp
from validation.detect_a2_peer import load, untouched, trailing_mean
from validation.a2_persist_eval import persist
from validation.eval_attack_bench import BASE, BENCH

def main():
    p = Pp.fit_default(); cols = p.cols
    order = [json.load(open(BENCH / "channels.json")).index(c) for c in cols]
    Xtr, ltr, _ = load("train"); Xva, lva, _ = load("val")
    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    Ztr = p._errors_raw(Xtr[ltr.attacked.values == 0]) / p.err_sd
    stats = {}
    for L in (5, 15):
        T = trailing_mean(Ztr, L)[L - 1:]; stats[L] = (T.mean(0), T.std(0) + 1e-12)

    def score(Xs, v):
        Z = p._errors_raw(Xs) / p.err_sd
        out = [np.nanmax(np.abs(Z), 1)] if v["inst"] else []
        for L in v["L"]:
            m, s = stats[L]; out.append(np.nanmax(np.abs((trailing_mean(Z, L) - m) / s), 1))
        if v.get("topk"):
            q = np.sort(np.nan_to_num(Z ** 2), 1)[:, ::-1][:, :v["topk"]].sum(1); out.append(np.sqrt(q))
        return np.nanmax(np.stack(out), 0)
    variants = {"base (max|z|)": {"inst": True, "L": ()}, "+L5": {"inst": True, "L": (5,)},
                "+L15": {"inst": True, "L": (15,)}, "+L5+L15": {"inst": True, "L": (5, 15)},
                "L5 only": {"inst": False, "L": (5,)}, "top4": {"inst": False, "L": (), "topk": 4}}
    wins = [json.loads(l) for l in open(BENCH / "windows.jsonl")]
    wins = [w for w in wins if w.get("ok") and w["type"] == "A3"]
    rng = np.random.default_rng(1); res = {}
    for name, v in variants.items():
        thr = float(np.nanquantile(score(Xva, v)[untouched(lva.attacked.values, 30)], 0.99))
        al = persist(score(X, v) > thr, 3); far = float(al[30:].mean())
        ch = {d: float(np.mean([al[i:i + d].any() for i in rng.integers(30, len(X) - d, 3000)])) for d in (15, 60, 240)}
        hit = {}
        for w in wins:
            a = np.load(BENCH / f"a_{w['id']}.npy")[:, order]; s, d = w["start"], w["duration"]
            seg = X[s - 30:s + d].copy(); seg[30:] += a
            hit[w["id"]] = (w["magnitude"], w["duration"], bool(persist(score(seg, v) > thr, 3)[30:].any()))
        def rate(par, cond=lambda m, d: True):
            h = [x[2] for i, x in hit.items() if i % 2 == par and cond(x[0], x[1])]; return f"{sum(h)}/{len(h)}"
        res[name] = {"FAR": far, "chance": ch, "tune(even)": rate(0), "report(odd)": rate(1),
                     "odd 2%": rate(1, lambda m, d: m == 0.02), "odd 5%": rate(1, lambda m, d: m == 0.05),
                     "odd 10%": rate(1, lambda m, d: m == 0.1), "odd d15": rate(1, lambda m, d: d == 15)}
        print(f"{name:14s} FAR {far*100:.2f}% chance15/60/240 {[round(ch[d]*100) for d in (15,60,240)]} | "
              f"EVEN {res[name]['tune(even)']} | ODD {res[name]['report(odd)']} (2% {res[name]['odd 2%']}, "
              f"5% {res[name]['odd 5%']}, 10% {res[name]['odd 10%']}, d15 {res[name]['odd d15']})", flush=True)
    json.dump(res, open(BASE / "validation" / "a3_variants_summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
