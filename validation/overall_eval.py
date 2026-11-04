"""
Overall catch rate of the detection system on every attack type, and
combination rules for the per-type detectors.

Benchmarks (one window at a time on the clean test telemetry, 30+ rows of
clean history before each window):
  A1  random multi-channel bias, fdia_v1 configs (class mixed/V/PQ, k 5/20,
      levels L0-L2) x step/ramp/pulse x 15/60/240, made here with the
      library's apply_bias_timestep (no OpenDSS needed)
  A2  single-meter bias (P, Q and V), same generator, fdia_v1 levels
  A3  dataset/attack_bench (OpenDSS, validation/gen_attack_bench.py)
  A4/A5 dataset/attack_bench at the library sizes (0.01-0.05 pu)
Each benchmark is split by window id: EVEN ids are the "tune" half (used to
choose a rule), ODD ids the "report" half.

Per-step scores, each divided by its own threshold (99% of clean val):
  V   validation/detect_a2_peer.py   (voltage peer test, trailing means)
  P   validation/detect_peer_power.py (active-power peer test)
  AE  existing AE-Mahalanobis (saved v3 weights)
Rules (all with K consecutive steps):
  V|P        alarm if V or P above its own threshold (current system)
  V|P|AE     adds the existing detector
  max-ratio  r_t = max(V/tV, P/tP, AE/tAE), one threshold re-calibrated so
             that the COMBINED rule is at 99% of clean val
Detection vs false-alarm rate is reported at several operating points.

Run: python -m validation.overall_eval
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
from attacks.library import AttackSpec, BiasContext, apply_bias_timestep  # noqa: E402
from validation import detect_a2_peer as V                                 # noqa: E402
from validation import detect_peer_power as Pp                             # noqa: E402
from validation.a2_persist_eval import persist                             # noqa: E402
from validation.detect_ae import W                                         # noqa: E402
from validation.eval_attack_bench import ae_maha, BENCH                    # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
CTX = max(30, W)
PQ_LEVELS, V_LEVELS = [0.05, 0.10, 0.20], [0.005, 0.01, 0.02]
PROFILES, DURATIONS = ["step", "ramp", "pulse"], [15, 60, 240]
SEED = 20261107


def bias_windows(kind, rng, X, cols, ctx):
    """A1 / A2 windows: list of (meta, segment with attack applied, offset of window in segment)."""
    out = []
    configs = []
    if kind == "A1":
        for cls in ("mixed", "V", "PQ"):
            for k in (5, 20):
                for lvl in range(3):
                    configs.append({"class": cls, "k": k, "level": lvl})
    else:
        for lvl in range(3):
            configs += [{"level": lvl}] * 6          # same count per level as A1 has per level (18)
    i = 0
    for cfg in configs:
        for prof in PROFILES:
            dur = DURATIONS[(i // 3) % 3]
            i += 1
            start = int(rng.integers(CTX, len(X) - dur - 1))
            lvl = cfg["level"]
            kw = {}
            if kind == "A1":
                kw = {"k": cfg["k"], "channel_class": cfg["class"]}
                if cfg["class"] in ("mixed", "PQ"):
                    kw["pq_rel"] = PQ_LEVELS[lvl]
                if cfg["class"] in ("mixed", "V"):
                    kw["v_abs_pu"] = V_LEVELS[lvl]
            else:
                kw = {"pq_rel": PQ_LEVELS[lvl], "v_abs_pu": V_LEVELS[lvl]}
            spec = AttackSpec(kind, start, dur, prof, seed=int(rng.integers(1 << 31)), **kw)
            s0 = start - CTX
            seg = X[s0:start + dur].copy()
            for r in range(len(seg)):
                row = dict(zip(cols, seg[r]))
                new, info = apply_bias_timestep(row, spec, s0 + r, ctx)
                if info["attacked"]:
                    seg[r] = [new[c] for c in cols]
            out.append(({"type": kind, "level": f"L{lvl}", "duration": dur, "profile": prof,
                         **({"class": cfg["class"], "k": cfg["k"]} if kind == "A1" else {})}, seg, CTX))
    return out


def bench_windows(X, cols):
    order = [json.load(open(BENCH / "channels.json")).index(c) for c in cols]
    out = []
    for line in open(BENCH / "windows.jsonl"):
        w = json.loads(line)
        if not w.get("ok"):
            continue
        a = np.load(BENCH / f"a_{w['id']}.npy")[:, order]
        s, d = w["start"], w["duration"]
        seg = X[s - CTX:s + d].copy()
        seg[CTX:] += a
        lvl = f"{int(round(w['magnitude'] * 100))}%" if w["type"] == "A3" else f"{w['magnitude']} pu"
        out.append(({"type": w["type"], "level": lvl, "duration": d, "profile": w["profile"],
                     "access": w.get("access", "")}, seg, CTX))
    return out


def main():
    vdet, pdet = V.fit_default(), Pp.fit_default()
    ae_score, ae_thr = ae_maha()
    cols = vdet.cols
    Xva, lva, _ = V.load("val")
    unt = V.untouched(lva["attacked"].to_numpy(), 30)

    def ratios(Xs):
        return np.stack([np.nan_to_num(vdet.score(Xs)) / vdet.thr, np.nan_to_num(pdet.score(Xs)) / pdet.thr,
                         ae_score(Xs) / ae_thr], axis=1)

    Rva = ratios(Xva)[unt]
    X = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")[cols].to_numpy(float)
    Rclean = ratios(X)
    print("clean ratios done", flush=True)

    rng = np.random.default_rng(SEED)
    ctx = BiasContext(sigma_overrides={})
    windows = bias_windows("A1", rng, X, cols, ctx) + bias_windows("A2", rng, X, cols, ctx) + bench_windows(X, cols)
    print(f"{len(windows)} windows built", flush=True)
    Rw = []
    for meta, seg, off in windows:
        Rw.append((meta, ratios(seg)[off:]))
    print("window ratios done", flush=True)

    def mk_rules(q):
        """name -> function(R) -> boolean alarms per step, thresholds at val quantile q."""
        t = [np.quantile(Rva[:, j], q) / np.quantile(Rva[:, j], 0.99) for j in range(3)]
        mx = np.quantile(Rva.max(1), q)
        mx2 = np.quantile(Rva[:, :2].max(1), q)
        return {
            "V|P": lambda R: (R[:, 0] > t[0]) | (R[:, 1] > t[1]),
            "V|P|AE": lambda R: (R[:, 0] > t[0]) | (R[:, 1] > t[1]) | (R[:, 2] > t[2]),
            "max-ratio(V,P)": lambda R: R[:, :2].max(1) > mx2,
            "max-ratio(V,P,AE)": lambda R: R.max(1) > mx,
            "AE only (existing)": lambda R: R[:, 2] > t[2],
        }

    results = {}
    rng2 = np.random.default_rng(1)
    for q in (0.99, 0.995, 0.98, 0.95):
        rules = mk_rules(q)
        for K in (1, 3):
            for name, fn in rules.items():
                al = persist(fn(Rclean), K)
                far = float(al[CTX:].mean())
                chance = {d: float(np.mean([al[i:i + d].any() for i in rng2.integers(CTX, len(X) - d, 2000)]))
                          for d in (15, 60, 240)}
                rows = []
                for wid, (meta, R) in enumerate(Rw):
                    rows.append({**meta, "half": "tune" if wid % 2 == 0 else "report",
                                 "hit": bool(persist(fn(R), K).any())})
                df = pd.DataFrame(rows)
                key = f"q{q} K{K} {name}"
                results[key] = {"FAR": far, "chance": chance, "df": df}
    # summaries
    summ = {}
    for key, r in results.items():
        df = r["df"]
        rep = df[df.half == "report"]
        by_type = {t: round(100 * g.hit.mean(), 1) for t, g in rep.groupby("type")}
        all_types = round(float(np.mean(list(by_type.values()))), 1)
        summ[key] = {"FAR_%": round(100 * r["FAR"], 2), "chance_%": {d: round(100 * v) for d, v in r["chance"].items()},
                     "report_by_type_%": by_type, "report_mean_over_types_%": all_types,
                     "tune_mean_over_types_%": round(100 * np.mean([g.hit.mean() for _, g in df[df.half == "tune"].groupby("type")]), 1),
                     "report_by_type_level_%": {f"{t} {l}": round(100 * g.hit.mean()) for (t, l), g in rep.groupby(["type", "level"])},
                     "report_by_type_duration_%": {f"{t} {d}": round(100 * g.hit.mean()) for (t, d), g in rep.groupby(["type", "duration"])}}
        if key.startswith("q0.99 K3") or key.endswith("max-ratio(V,P,AE)"):
            print(f"{key:32s} FAR {summ[key]['FAR_%']:5.2f}% | tune {summ[key]['tune_mean_over_types_%']:5.1f}% "
                  f"report {summ[key]['report_mean_over_types_%']:5.1f}% | {by_type}", flush=True)
    json.dump(summ, open(OUT / "overall_eval_summary.json", "w"), indent=1, default=float)
    pd.concat([r["df"].assign(rule=k) for k, r in results.items() if k.startswith("q0.99 K3")]).to_csv(
        OUT / "overall_eval_windows.csv", index=False)


if __name__ == "__main__":
    main()
