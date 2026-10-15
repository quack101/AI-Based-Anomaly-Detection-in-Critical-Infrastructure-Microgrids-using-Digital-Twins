"""
Build dataset/fdia_v1 -- a labeled benchmark of normal and anomalous
meter telemetry for the simulated IEEE-123 feeder (corpus A, P0). The
clean telemetry is simulation/phase6_corpus (never modified). TRAIN is
copied through unchanged; VAL and TEST get synthetic anomaly windows
built with attacks/library.py (A1 bias, A2 single-meter bias, A3 zone
load redistribution). This is the detector benchmark -- z_delivered =
z_clean + a at Layer 2 only; no ground-truth column is ever written
into a z file, and nothing reads x_true.

Windows: 18 per split, durations cycling 15/60/240 (6 each), profiles
cycling step/ramp/pulse decoupled from duration (6 each, all 9
duration x profile combos covered), attack configs cycling a rotated
25-entry list (A1: 3 classes x 2 k x 3 levels = 18; A2: 3 levels; A3:
3 full + 1 partial). Each window sits in its own 1/18 segment of the
split with a seeded offset, so consecutive windows are >= 60 clean
apart. ~14.6% of each split's timesteps are attacked.

Levels: P/Q bias = r*|z_t|, r in {0.05, 0.10, 0.20}; V bias in
{0.005, 0.01, 0.02} pu. A3 dP_total in {2, 5, 10}% of the current
total metered P.

Run: python -m validation.build_fdia_v1
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.library import attack, AttackSpec, AttackContext

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS = BASE_DIR / "simulation" / "phase6_corpus"
OUT_DIR = BASE_DIR / "dataset" / "fdia_v1"
REPORTS = BASE_DIR / "reports"

N_WINDOWS = 18
DURATIONS = [15, 60, 240]
PROFILES = ["step", "ramp", "pulse"]
PQ_LEVELS = [0.05, 0.10, 0.20]
V_LEVELS = [0.005, 0.01, 0.02]
A3_LEVELS = [0.02, 0.05, 0.10]
SPLIT_SEED = {"val": 20260101, "test": 20260202}
SPLIT_OFFSET = {"val": 0, "test": 12}  # rotate the config list so val+test together cover all 25


def build_configs():
    cfg = []
    for cls in ["mixed", "V", "PQ"]:
        for k in [5, 20]:
            for lvl in [0, 1, 2]:
                cfg.append({"type": "A1", "class": cls, "k": k, "level": lvl})
    for lvl in [0, 1, 2]:
        cfg.append({"type": "A2", "class": "meter", "level": lvl})
    for lvl in [0, 1, 2]:
        cfg.append({"type": "A3", "class": "zone", "level": lvl, "access": "full"})
    cfg.append({"type": "A3", "class": "zone", "level": 1, "access": "partial"})  # 5% partial
    return cfg


def spec_for(cfg, start, duration, profile, seed):
    t, lvl = cfg["type"], cfg["level"]
    if t == "A1":
        kw = {"k": cfg["k"], "channel_class": cfg["class"]}
        if cfg["class"] in ("mixed", "PQ"):
            kw["pq_rel"] = PQ_LEVELS[lvl]
        if cfg["class"] in ("mixed", "V"):
            kw["v_abs_pu"] = V_LEVELS[lvl]
        return AttackSpec("A1", start, duration, profile, seed=seed, **kw)
    if t == "A2":
        return AttackSpec("A2", start, duration, profile, seed=seed,
                          pq_rel=PQ_LEVELS[lvl], v_abs_pu=V_LEVELS[lvl])
    return AttackSpec("A3", start, duration, profile, seed=seed, zone=True, group_size=4,
                      magnitude=A3_LEVELS[lvl], access=cfg["access"])


def level_label(cfg):
    if cfg["type"] == "A3":
        return f"{int(A3_LEVELS[cfg['level']] * 100)}pct_{cfg['access']}"
    return f"L{cfg['level']}(pq={PQ_LEVELS[cfg['level']]},v={V_LEVELS[cfg['level']]}pu)"


def class_label(cfg):
    return cfg["class"] + (f"_k{cfg['k']}" if cfg["type"] == "A1" else "")


TYPE_PATTERN = ["A1"] * 12 + ["A2"] * 3 + ["A3"] * 3   # 18 windows: A1 x12, A2 x3, A3 x3
A1_OFFSET = {"val": 0, "test": 9}
A3_OFFSET = {"val": 0, "test": 2}


def plan_windows(n_rows, split):
    """Duration and profile are a shuffled balanced multiset (6 of each,
    all 9 duration x profile combos present twice); type is a separately
    shuffled multiset (A1 x12, A2 x3, A3 x3). Shuffling the two
    independently decouples type from duration/profile. Each type cycles
    its own sub-config list from a per-split offset so val and test
    together cover every config."""
    rng = np.random.default_rng(SPLIT_SEED[split])
    all_cfg = build_configs()
    a1 = [c for c in all_cfg if c["type"] == "A1"]
    a2 = [c for c in all_cfg if c["type"] == "A2"]
    a3 = [c for c in all_cfg if c["type"] == "A3"]

    dur_prof = [(d, p) for p in PROFILES for d in DURATIONS] * 2  # 18, 6 each duration & profile
    rng.shuffle(dur_prof)
    types = list(TYPE_PATTERN)
    rng.shuffle(types)

    counters = {"A1": 0, "A2": 0, "A3": 0}
    sub = {"A1": (a1, A1_OFFSET[split]), "A2": (a2, 0), "A3": (a3, A3_OFFSET[split])}
    seg = n_rows // N_WINDOWS
    plan = []
    for i in range(N_WINDOWS):
        dur, prof = dur_prof[i]
        t = types[i]
        lst, off = sub[t]
        cfg = lst[(off + counters[t]) % len(lst)]
        counters[t] += 1
        jit = int(rng.integers(0, max(1, seg - dur - 60)))
        start = i * seg + jit
        plan.append({"window_id": i, "start": start, "duration": dur, "profile": prof, "config": cfg,
                     "seed": SPLIT_SEED[split] + i})
    return plan


def build_split(split, df, df_str, ctx):
    """df: parsed telemetry (used to compute attacks + labels). df_str:
    the SAME file read as raw strings, so clean cells are written back
    byte-identical to the corpus (only attacked cells are reformatted).
    Returns a string DataFrame [timestamp + 168 channels]."""
    chan = [c for c in df.columns if c not in ("timestamp", "real_timestamp")]
    z_clean = df[chan].reset_index(drop=True)
    ts = df["real_timestamp"].tolist()
    z_str = df_str[["timestamp"] + chan].reset_index(drop=True).copy()
    col_of = {c: z_str.columns.get_loc(c) for c in chan}
    labels = pd.DataFrame({"timestamp": df["timestamp"].values, "attacked": 0, "window_id": -1,
                           "type": "", "class": "", "level": "", "profile": "", "duration": 0, "targets": ""})
    plan = plan_windows(len(df), split)
    used = np.zeros(len(df), dtype=bool)
    for w in plan:
        cfg, s, d = w["config"], w["start"], w["duration"]
        assert not used[s:s + d].any(), "window overlap"
        used[s:s + d] = True
        spec = spec_for(cfg, s, d, w["profile"], w["seed"])
        za, lb = attack(z_clean, ts, spec, ctx)
        active = lb["attacked"].values.astype(bool)
        idx = np.arange(s, s + d)[active[s:s + d]]
        labels.loc[idx, ["attacked", "window_id", "type", "class", "level", "profile", "duration"]] = [
            1, w["window_id"], cfg["type"], class_label(cfg), level_label(cfg), w["profile"], d]
        for r in idx:
            tgt = lb.iloc[r]["target_ids"]
            labels.at[r, "targets"] = tgt
            for c in tgt.split(";"):
                if c:                                   # only attacked cells reformatted
                    z_str.iat[r, col_of[c]] = repr(float(za.iloc[r][c]))
    return z_str, labels, plan, chan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
    ctx = AttackContext()

    # train: copied through unchanged (clean), corpus strings verbatim
    train_str = pd.read_csv(CORPUS / "train_telemetry.csv", dtype=str, keep_default_na=False)
    chan = [c for c in train_str.columns if c not in ("timestamp", "real_timestamp")]
    train_str[["timestamp"] + chan].to_csv(OUT_DIR / "train_z.csv", index=False)
    pd.DataFrame({"timestamp": train_str["timestamp"], "attacked": 0, "window_id": -1, "type": "", "class": "",
                  "level": "", "profile": "", "duration": 0, "targets": ""}).to_csv(
        OUT_DIR / "train_labels.csv", index=False)

    manifest = {"dataset": "fdia_v1", "code_commit": commit, "clean_source": "simulation/phase6_corpus",
                "time_resolution": "1 minute", "n_windows_per_split": N_WINDOWS,
                "levels": {"pq_rel": PQ_LEVELS, "v_abs_pu": V_LEVELS, "a3_dP_frac": A3_LEVELS},
                "splits": {}, "windows": {}, "sha256": {}}
    counts = []
    for split in ["val", "test"]:
        df = pd.read_csv(CORPUS / f"{split}_telemetry.csv")
        df_str = pd.read_csv(CORPUS / f"{split}_telemetry.csv", dtype=str, keep_default_na=False)
        z_df, labels, plan, chan = build_split(split, df, df_str, ctx)
        assert not any(c for c in z_df.columns if c in ("real_timestamp", "converged", "iterations", "solve_time_s")
                       or "x_true" in c), "ground-truth column in z"
        z_df.to_csv(OUT_DIR / f"{split}_z.csv", index=False)
        labels.to_csv(OUT_DIR / f"{split}_labels.csv", index=False)
        att = labels["attacked"].sum()
        manifest["splits"][split] = {"n_timesteps": len(df), "n_attacked": int(att),
                                     "attacked_fraction": float(att / len(df)), "seed": SPLIT_SEED[split],
                                     "a1_offset": A1_OFFSET[split], "a3_offset": A3_OFFSET[split],
                                     "n_windows": len(plan)}
        manifest["windows"][split] = [
            {"window_id": w["window_id"], "start": int(w["start"]), "duration": w["duration"],
             "profile": w["profile"], "seed": w["seed"], "type": w["config"]["type"],
             "class": class_label(w["config"]), "level": level_label(w["config"])} for w in plan]
        aw = labels[labels["attacked"] == 1].copy()
        aw["type_class"] = aw["type"] + ":" + aw["class"]
        for dim in ["type", "type_class", "level", "profile", "duration"]:
            g = aw.groupby(dim).size().reset_index(name="n_attacked_timesteps").rename(columns={dim: "value"})
            g.insert(0, "dimension", dim)
            g.insert(0, "split", split)
            counts.append(g)
        # window counts (number of windows, not timesteps) per dimension
        wp = pd.DataFrame([{"type": w["config"]["type"], "type_class": w["config"]["type"] + ":" + class_label(w["config"]),
                            "level": level_label(w["config"]), "profile": w["profile"],
                            "duration": w["duration"]} for w in plan])
        for dim in ["type", "type_class", "level", "profile", "duration"]:
            g = wp.groupby(dim).size().reset_index(name="n_windows").rename(columns={dim: "value"})
            g.insert(0, "dimension", dim)
            g.insert(0, "split", split)
            counts.append(g)

    for f in ["train_z.csv", "train_labels.csv", "val_z.csv", "val_labels.csv", "test_z.csv", "test_labels.csv"]:
        manifest["sha256"][f] = sha256(OUT_DIR / f)
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    counts_df = pd.concat(counts, ignore_index=True)
    counts_df = counts_df[["split", "dimension", "value", "n_attacked_timesteps", "n_windows"]]
    counts_df.to_csv(REPORTS / "fdia_v1_counts.csv", index=False)
    with open(REPORTS / "fdia_v1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("built dataset/fdia_v1  commit", commit[:10])
    for split in ["val", "test"]:
        s = manifest["splits"][split]
        print(f"  {split}: {s['n_attacked']}/{s['n_timesteps']} attacked ({s['attacked_fraction']*100:.2f}%)")
    pd.set_option("display.width", 200)
    for dim in ["type", "profile", "duration"]:
        t = counts_df[counts_df["dimension"] == dim]
        print(f"\n{dim}: windows " +
              ", ".join(f"{v}={int(t[t.value.astype(str) == str(v)]['n_windows'].sum())}"
                        for v in sorted(t["value"].astype(str).unique())) +
              " | attacked-timesteps " +
              ", ".join(f"{v}={int(t[t.value.astype(str) == str(v)]['n_attacked_timesteps'].sum())}"
                        for v in sorted(t["value"].astype(str).unique())))
    tc = counts_df[(counts_df["dimension"] == "type_class")]
    print("\ntype_class coverage across both splits:", sorted(tc["value"].unique()))
    print("saved manifest + counts to reports/")


if __name__ == "__main__":
    main()
