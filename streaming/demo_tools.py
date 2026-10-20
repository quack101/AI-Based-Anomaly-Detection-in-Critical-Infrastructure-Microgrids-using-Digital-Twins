"""
streaming/demo_tools.py -- helpers for the live Kafka demo runs. Not part of
the detector: these only move data, check it, and summarise it.

  python -m streaming.demo_tools reset
        delete + recreate telemetry.raw / telemetry.delivered / labels / alarms
  python -m streaming.demo_tools dump <run_dir>
        consume alarms + labels from the broker into <run_dir>/alarms.csv and
        labels.csv (consumes from the beginning until the topic goes idle)
  python -m streaming.demo_tools gate <run_dir> [n_scored]
        REAL-KAFKA GATE: consume telemetry.delivered through the real broker,
        assemble the 168-vector exactly as DetectorConsumer does, compare to the
        producer's truth vector (<run_dir>/truth.csv) and to
        DetectorCore.score_vector on the truth window
  python -m streaming.demo_tools analyze <run_dir> <run_label> [--functional]
        per-window detection table + timeline PNG (<run_dir>/summary.json,
        timeline.png). --functional: pipeline test only; no rates reported.
  python -m streaming.demo_tools diagnose <run_dir>
        live clean scores vs the offline fusion score at the same test rows,
        and live-vs-corpus telemetry differences (start time / load profile /
        noise scale)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TOPICS = ["telemetry.raw", "telemetry.delivered", "labels", "alarms"]
BOOT = "localhost:9092"
TEST_ROW0 = 73440          # stream index of test-split row 0
W = 10


def reset():
    from kafka.admin import KafkaAdminClient, NewTopic
    a = KafkaAdminClient(bootstrap_servers=BOOT)
    try:
        a.delete_topics(TOPICS)
    except Exception:
        pass
    time.sleep(6)
    for _ in range(15):
        try:
            a.create_topics([NewTopic(t, 1, 1) for t in TOPICS])
            break
        except Exception:
            time.sleep(2)
    print("topics:", sorted(t for t in a.list_topics() if not t.startswith("__")))


def _read_topic(topic, idle_ms=5000):
    from kafka import KafkaConsumer, TopicPartition
    c = KafkaConsumer(bootstrap_servers=BOOT, auto_offset_reset="earliest", consumer_timeout_ms=idle_ms,
                      value_deserializer=lambda m: json.loads(m.decode("utf-8")))
    tp = TopicPartition(topic, 0)
    c.assign([tp])
    c.seek_to_beginning(tp)
    rows = [r.value for r in c]
    c.close()
    return rows


def dump(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    al = pd.DataFrame(_read_topic("alarms")).sort_values("timestamp")
    al.to_csv(run_dir / "alarms.csv", index=False)
    lb = pd.DataFrame(_read_topic("labels")).sort_values("timestamp")
    lb["attacked_meters"] = lb["attacked_meters"].map(lambda v: ";".join(v))
    lb["targets"] = lb["targets"].map(lambda v: ";".join(v))
    lb.to_csv(run_dir / "labels.csv", index=False)
    print(f"alarms: {len(al)} rows ({int(al['alarm'].sum())} alarm=True); labels: {len(lb)} rows "
          f"({int(lb['attacked'].sum())} attacked)")


def gate(run_dir, n_scored=20):
    from streaming.detector_core import DetectorCore, AE_W
    from streaming.detector_consumer import DetectorConsumer
    run_dir = Path(run_dir)
    truth = pd.read_csv(run_dir / "truth.csv").set_index("timestamp")
    labels = pd.read_csv(run_dir / "labels.csv").set_index("timestamp")
    core = DetectorCore(registry_variant="multiphase")
    dc = DetectorConsumer(core, window_timeout_s=1e9)
    from kafka import KafkaConsumer, TopicPartition
    c = KafkaConsumer(bootstrap_servers=BOOT, auto_offset_reset="earliest", consumer_timeout_ms=8000,
                      value_deserializer=lambda m: json.loads(m.decode("utf-8")))
    tp = TopicPartition("telemetry.delivered", 0)
    c.assign([tp])
    c.seek_to_beginning(tp)
    n_msgs, rows = 0, []
    hist = []
    for rec in c:                                   # real-broker consumption, as stored
        n_msgs += 1
        r = dc.ingest(rec.value)
        if r is None or not r["scored"]:
            continue
        t = r["timestamp"]
        if labels.loc[t, "attacked"] == 1:
            continue                                # gate on clean timesteps only
        # truth window = the producer's own vectors for t-9..t (all clean for this gate)
        tw = [t - k for k in range(AE_W - 1, -1, -1)]
        if any(labels.loc[x, "attacked"] == 1 for x in tw if x in labels.index):
            continue
        window = truth.loc[tw, core.cols].to_numpy(float)
        direct = core.score_vector(window, str(r["real_timestamp"]))
        rows.append({"timestamp": t, "dvec": float(np.max(np.abs(r["current_vec"] - window[-1]))),
                     "dfus": abs(r["fusion_score"] - direct["fusion_score"]),
                     "alarm_match": bool(r["alarm"] == direct["alarm"]),
                     "dJ_rel": abs(r["J"] - direct["J"]) / max(abs(direct["J"]), 1e-12)})
        if len(rows) >= n_scored:
            break
    c.close()
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "gate_real_kafka.csv", index=False)
    print(f"consumed {n_msgs} messages from telemetry.delivered via the real broker; "
          f"{len(df)} clean scored timesteps compared")
    print(f"  max |vector_from_kafka - producer_truth_vector| = {df['dvec'].max():.3e}")
    print(f"  max |fusion_from_kafka - direct score_vector|   = {df['dfus'].max():.3e}")
    print(f"  max rel diff in J                               = {df['dJ_rel'].max():.3e}")
    print(f"  alarms identical                                = {bool(df['alarm_match'].all())}")
    ok = len(df) >= n_scored and df["dvec"].max() <= 1e-6 and df["dfus"].max() <= 1e-6 and df["alarm_match"].all()
    print(f"REAL-KAFKA GATE: {'PASS' if ok else 'FAIL'}")
    return ok


def _windows(labels):
    att = labels["attacked"].to_numpy().astype(bool)
    ts = labels["timestamp"].to_numpy()
    wins, i = [], 0
    while i < len(att):
        if att[i]:
            j = i
            while j + 1 < len(att) and att[j + 1] and ts[j + 1] == ts[j] + 1:
                j += 1
            wins.append((int(ts[i]), int(ts[j]), labels.iloc[i]["attack_class"],
                         set(";".join(labels.iloc[i:j + 1]["targets"].fillna("")).split(";")) - {""}))
            i = j + 1
        else:
            i += 1
    return wins


def analyze(run_dir, label, functional=False):
    run_dir = Path(run_dir)
    al = pd.read_csv(run_dir / "alarms.csv").sort_values("timestamp").reset_index(drop=True)
    lb = pd.read_csv(run_dir / "labels.csv").sort_values("timestamp").reset_index(drop=True)
    thr = float(np.load(BASE / "streaming" / "detector_fit_cache.npz")["thr_f"])
    wins = _windows(lb)
    attacked_ts = set(lb.loc[lb["attacked"] == 1, "timestamp"])
    spill = set()
    for s, e, _, _ in wins:
        spill |= set(range(e + 1, e + W))                 # detector window still holds attacked steps
    al["in_window"] = al["timestamp"].isin(attacked_ts)
    al["spill"] = al["timestamp"].isin(spill)
    clean = al[~al["in_window"] & ~al["spill"]]
    summary = {"run": label, "functional_only": bool(functional), "n_scored": int(len(al)),
               "n_alarm_total": int(al["alarm"].sum()), "fusion_threshold": thr,
               "windows": []}
    if not functional:
        summary["clean_stretch"] = {"n_steps": int(len(clean)), "n_alarms": int(clean["alarm"].sum()),
                                    "alarm_rate": float(clean["alarm"].mean()) if len(clean) else None,
                                    "definition": "scored steps outside attack windows and not within the "
                                                  f"{W - 1} steps after one (detector history spillover)"}
    for s, e, cls, tg in wins:
        sub = al[(al["timestamp"] >= s) & (al["timestamp"] <= e)]
        hit = sub[sub["alarm"]]
        w = {"class": cls, "start": s, "end": e, "n_targets": len(tg),
             "targets_sample": sorted(tg)[:6]}
        if not functional:
            if len(hit):
                first = hit.iloc[0]
                w.update({"detected": True, "delay": int(first["timestamp"] - s),
                          "first_alarm_top_channel": first["top_channel"],
                          "top_channel_matches_attacked": bool(first["top_channel"] in tg),
                          "alarm_steps": int(len(hit)),
                          "frac_alarm_steps_top_in_targets": float(np.mean([c in tg for c in hit["top_channel"]]))})
            else:
                w.update({"detected": False, "delay": None, "top_channel_matches_attacked": None})
        summary["windows"].append(w)
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2, default=str)

    # ---- timeline PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f2933", "#6b7785", "#e3e7ec"
    hues = {"A1": "#d9822b", "A2": "#2b7bb9"}
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=140)
    for s, e, cls, _ in wins:
        ax.axvspan(s - 0.5, e + 0.5, color=hues.get(cls, "#999999"), alpha=0.16, lw=0)
        ax.text((s + e) / 2, 1.03, cls, ha="center", va="bottom", fontsize=9, color=ink,
                transform=ax.get_xaxis_transform())
    ax.plot(al["timestamp"], al["fusion_score"], color=ink, lw=1.2, label="fusion score")
    fired = al[al["alarm"]]
    if len(fired):
        ax.plot(fired["timestamp"], fired["fusion_score"], "o", ms=3.5, color="#c0392b", label="alarm")
    ax.axhline(thr, color=muted, lw=1, ls=(0, (4, 3)), label=f"threshold ({thr:.3f})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("stream index (1 step = 1 simulated minute)", fontsize=9, color=muted)
    ax.set_ylabel("fusion score", fontsize=9, color=muted)
    ax.grid(True, color=grid, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    ax.set_title(label, fontsize=10, color=ink, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(run_dir / "timeline.png")
    plt.close(fig)

    print(f"[{label}] scored {summary['n_scored']} steps, alarm=True on {summary['n_alarm_total']}")
    if functional:
        print("  (functional run: pipeline test only -- no rates reported)")
    else:
        cs = summary["clean_stretch"]
        print(f"  clean-stretch: {cs['n_alarms']}/{cs['n_steps']} alarms = {cs['alarm_rate']*100:.2f}%")
    for w in summary["windows"]:
        if functional:
            print(f"  window {w['class']} t={w['start']}..{w['end']} targets={w['n_targets']} {w['targets_sample']}")
        elif w["detected"]:
            print(f"  window {w['class']} t={w['start']}..{w['end']}: DETECTED delay {w['delay']} steps, "
                  f"first-alarm top={w['first_alarm_top_channel']} matches_attacked={w['top_channel_matches_attacked']}, "
                  f"alarm steps {w['alarm_steps']}, top-in-targets {w['frac_alarm_steps_top_in_targets']*100:.0f}%")
        else:
            print(f"  window {w['class']} t={w['start']}..{w['end']}: not detected")


def diagnose(run_dir):
    """Live clean scores vs the offline fusion at the same test rows, then the
    three suspects in order: (a) start time, (b) load profile, (c) noise scale."""
    from streaming.detector_core import DetectorCore
    from validation.detect_mahalanobis import ae_errors
    run_dir = Path(run_dir)
    al = pd.read_csv(run_dir / "alarms.csv").sort_values("timestamp").reset_index(drop=True)
    lb = pd.read_csv(run_dir / "labels.csv")
    attacked_ts = set(lb.loc[lb["attacked"] == 1, "timestamp"])
    spill = set()
    for s, e, _, _ in _windows(lb):
        spill |= set(range(e + 1, e + W))
    live = al[~al["timestamp"].isin(attacked_ts) & ~al["timestamp"].isin(spill)].copy()
    live["row"] = live["timestamp"] - TEST_ROW0

    core = DetectorCore(registry_variant="multiphase")
    fd = BASE / "dataset" / "fdia_v1"
    tz = pd.read_csv(fd / "test_z.csv")
    X = tz[core.cols].to_numpy(float)
    comm = pd.read_csv(BASE / "validation" / "fdia_v1_bdd_test_v5_pertimestep.csv").set_index("row")
    tl = pd.read_csv(fd / "test_labels.csv")["attacked"].to_numpy()
    rows = live["row"].to_numpy()
    ok = np.array([0 <= r < len(tl) and tl[r] == 0 for r in rows])
    live, rows = live[ok].reset_index(drop=True), rows[ok]
    Ete = ae_errors(core.P, core.mu_ae, core.sd_ae, core.r_c, X)
    d = Ete[rows] - core.mu
    off_maha = np.einsum("ij,ij->i", d @ core.Minv, d)
    off_J = comm.loc[rows, "J"].to_numpy(); off_L = comm.loc[rows, "max_lnr"].to_numpy()
    off_bdd = np.maximum(off_J / core.J99, off_L / core.LNR99)
    zb = (off_bdd - core.muf[0]) / core.sdf[0]; zm = (off_maha - core.muf[1]) / core.sdf[1]
    off_fus = 1.0 / (1.0 + np.exp(-(zb * core.w_bdd + zm * core.w_maha + core.w_b)))

    def q(x):
        x = np.asarray(x, float)
        return f"median {np.median(x):9.3f}  p95 {np.percentile(x, 95):9.3f}"
    print(f"live clean steps compared: {len(live)} (offline rows {rows.min()}..{rows.max()})")
    print(f"  fusion   live  {q(live['fusion_score'])}   | offline {q(off_fus)}")
    print(f"  J        live  {q(live['J'])}   | offline {q(off_J)}")
    print(f"  maxLNR   live  {q(live['maxLNR'])}   | offline {q(off_L)}")
    print(f"  bdd      live  {q(live['bdd_score'])}   | offline {q(off_bdd)}")
    print(f"  maha     live  {q(live['ae_mahalanobis'])}   | offline {q(off_maha)}")
    res = {"n": int(len(live)), "fusion": [float(np.median(live['fusion_score'])), float(np.percentile(live['fusion_score'], 95)),
                                           float(np.median(off_fus)), float(np.percentile(off_fus, 95))],
           "J": [float(np.median(live['J'])), float(np.percentile(live['J'], 95)),
                 float(np.median(off_J)), float(np.percentile(off_J, 95))],
           "bdd": [float(np.median(live['bdd_score'])), float(np.percentile(live['bdd_score'], 95)),
                   float(np.median(off_bdd)), float(np.percentile(off_bdd, 95))],
           "maha": [float(np.median(live['ae_mahalanobis'])), float(np.percentile(live['ae_mahalanobis'], 95)),
                    float(np.median(off_maha)), float(np.percentile(off_maha, 95))]}

    # (a)/(b)/(c): live published telemetry (truth.csv) vs the corpus at the same rows
    tp = run_dir / "truth.csv"
    if tp.exists():
        tr = pd.read_csv(tp).set_index("timestamp")
        corp = pd.read_csv(BASE / "simulation" / "phase6_corpus" / "test_telemetry.csv")
        corp.index = corp["timestamp"]
        common = [t for t in tr.index if t in corp.index and tl[t - TEST_ROW0] == 0]
        if common:
            diff = tr.loc[common, core.cols].to_numpy(float) - corp.loc[common, core.cols].to_numpy(float)
            sig = {e["id"]: e["sigma"] for e in core.model.meas_entries}
            from observability.estimator_model import default_sigma_overrides
            s1 = json.load(open(BASE / "config" / "sigma_overrides_s1.json"))["sigma"]
            from sensors.registry import load_registry, enabled_entries
            reg = {e["id"]: e["sigma"] for e in enabled_entries(load_registry(variant="multiphase"))}
            sig_noise = np.array([s1.get(c, reg[c]) for c in core.cols])
            bias = diff.mean(0) / sig_noise
            ratio = diff.std(0) / (np.sqrt(2.0) * sig_noise)
            order = np.argsort(-np.abs(bias))[:6]
            print(f"\n(live published telemetry vs corpus test telemetry at the same {len(common)} rows)")
            print(f"  systematic offset |mean diff|/sigma: median {np.median(np.abs(bias)):.3f}, "
                  f"p95 {np.percentile(np.abs(bias), 95):.3f}, max {np.abs(bias).max():.2f}")
            print(f"  noise-scale ratio std(diff)/(sqrt2*sigma): median {np.median(ratio):.3f}, p5 {np.percentile(ratio, 5):.3f}, "
                  f"p95 {np.percentile(ratio, 95):.3f}")
            print("  channels with largest systematic offset (in sigma):",
                  [(core.cols[i], round(float(bias[i]), 2)) for i in order])
            res["offset_sigma_median"] = float(np.median(np.abs(bias)))
            res["offset_sigma_p95"] = float(np.percentile(np.abs(bias), 95))
            res["noise_ratio_median"] = float(np.median(ratio))
            res["noise_ratio_p5_p95"] = [float(np.percentile(ratio, 5)), float(np.percentile(ratio, 95))]
            res["top_offsets"] = [(core.cols[i], float(bias[i])) for i in order]
    json.dump(res, open(run_dir / "diagnosis.json", "w"), indent=2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "reset":
        reset()
    elif cmd == "dump":
        dump(sys.argv[2])
    elif cmd == "gate":
        gate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3], functional="--functional" in sys.argv)
    elif cmd == "diagnose":
        diagnose(sys.argv[2])
    else:
        raise SystemExit(f"unknown command {cmd!r}")
