"""Replay demo runB_live (A1 at 81780, A2 at 81900, both 60-step steps) on the
corpus test telemetry (the producer's REPLAY input: same loads and noise
level, different noise draw from the live run), apply the SAME attack specs
with the library's per-timestep path (as streaming/injector.py does), and
score with the peer-voltage detector (K=3). Writes a timeline figure."""
import sys, types, json
import numpy as np, pandas as pd
sys.modules.setdefault("opendssdirect", types.ModuleType("opendssdirect"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from attacks.library import AttackSpec, BiasContext, apply_bias_timestep
from validation.detect_a2_peer import fit_default
from validation.a2_persist_eval import persist
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUN = BASE / "streaming" / "demo_runs" / "runB_live"
K = 3

def main():
    run = json.load(open(RUN / "run.json")); specs = [AttackSpec(**s) for s in run["specs"]]
    det = fit_default(); cols = det.cols
    tel = pd.read_csv(BASE / "simulation/phase6_corpus/test_telemetry.csv")
    ts = tel["timestamp"].to_numpy(); X = tel[cols].to_numpy(float)
    i0 = int(np.searchsorted(ts, run["start"])) - 30                 # 30 rows of history
    i1 = int(np.searchsorted(ts, run["start"] + run["steps"]))
    seg, T = X[i0:i1].copy(), ts[i0:i1]
    ctx = BiasContext(sigma_overrides={}); targets = {}
    for i in range(len(seg)):
        row = dict(zip(cols, seg[i]))
        for sp in specs:
            row, info = apply_bias_timestep(row, sp, int(T[i]), ctx)
            if info["attacked"]: targets[sp.type] = info["targets"]
        seg[i] = [row[c] for c in cols]
    s = det.score(seg); al = persist(s > det.thr, K); loc = det.localize(seg)
    out = {"threshold": det.thr, "K": K, "targets": targets, "windows": {}}
    inwin = np.zeros(len(T), bool)
    for sp in specs:
        w = (T >= sp.start) & (T < sp.start + sp.duration); inwin |= (T >= sp.start) & (T < sp.start + sp.duration + 30)
        hit = np.flatnonzero(al & w)
        top = pd.Series([cols[j] for j in loc[hit]]).mode().iloc[0] if len(hit) else None
        out["windows"][sp.type] = {"detected": bool(len(hit)), "delay_steps": int(T[hit[0]] - sp.start) if len(hit) else None,
                                   "alarm_steps": int(len(hit)), "of": int(w.sum()), "top_channel": top,
                                   "top_on_attacked_meter": bool(top and any(t.split('_')[0] == top.split('_')[0] for t in targets[sp.type]))}
    clean = (T >= run["start"]) & ~inwin
    out["clean_alarm_rate"] = float(al[clean].mean()); out["clean_steps"] = int(clean.sum())
    print(json.dumps(out, indent=1, default=float))
    json.dump(out, open(RUN / "peer_detector_replay.json", "w"), indent=1, default=float)
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    m = T >= run["start"]
    ax.plot(T[m], s[m], color="#2a78d6", lw=1.4, label="peer-voltage score")
    ax.axhline(det.thr, color="#6b6a64", ls="--", lw=1, label=f"threshold ({det.thr:.2f})")
    for sp, c in zip(specs, ("#eb6834", "#1baf7a")):
        ax.axvspan(sp.start, sp.start + sp.duration, color=c, alpha=0.15, label=f"{sp.type} attack")
    ax.scatter(T[m & al], s[m & al], s=12, color="#c0392b", zorder=5, label=f"alarm ({K} steps in a row)")
    ax.set_xlabel("timestep"); ax.set_ylabel("score"); ax.set_title("Demo run B replay: peer-voltage detector", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(ncol=5, fontsize=8, loc="upper left", frameon=False)
    fig.savefig(RUN / "peer_detector_replay.png", dpi=160)

if __name__ == "__main__":
    main()
