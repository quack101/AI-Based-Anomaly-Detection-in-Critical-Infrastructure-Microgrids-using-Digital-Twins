"""
Every figure of the Phase-3 report, drawn with matplotlib from the
repository's own result files (no hand-drawn or generated images).

Run from latest-code:   python reports/figures/make_report_figures.py
Output:                 reports/figures/fig_*.png  (200 dpi)

Data sources
  fig8_1  reports/placement_quality_vs_baselines.txt (sweep run 20260822T092205Z, independent construction)
  fig8_2/8_3  Table 8.4 of the report (validation/detect_*_summary.json) + detect_robustness_summary.json
  fig8_4  validation/detect_robustness_summary.json (five AE seeds)
  fig8_5  clean validation MSE per bottleneck size (Section 6.7.1 / 8.5.5)
  fig8_6  streaming/demo_runs/runB_realistic/{alarms,labels}.csv
  fig8_7  validation/overall_eval_summary.json (held-out half, q0.99, K=3)
  fig8_8  validation/a2_persist_summary.json, validation/a2_baseline_summary.json
  fig8_9  validation/overall_eval_summary.json (A3 by level and duration)
  fig8_10 streaming/demo_runs/runC_all/{alarms,labels}.csv (live A1-A5 run)
  fig8_11 streaming/demo_runs/runC_failover/{alarms,labels}.csv, broker_kill.json (containers, two broker kills)
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NAVY, GREY, ORANGE, GOLD, PURPLE = "#1f3a5f", "#6b7280", "#c0622d", "#9a7b16", "#6a3fb5"
FILL, FILL_O, FILL_Y, EDGE_G = "#e8edf4", "#fbeee6", "#fdf3d1", "#8a8f99"
HUES = {"A1": "#f3d9c4", "A2": "#c9d7ea", "A3": "#c4e6d6", "A4": "#f2cfdd", "A5": "#ddd0dc"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


# ----------------------------------------------------------------- helpers
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="plain", size=9.5, bold=False, align="center"):
    fc, ec, lw = {"plain": ("white", NAVY, 1.3), "fill": (FILL, NAVY, 1.3), "new": (FILL_O, ORANGE, 1.6),
                  "kafka": (FILL_Y, GOLD, 1.4), "band": ("#f4f4f5", EDGE_G, 1.2),
                  "dash": ("none", EDGE_G, 1.2)}[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw,
                       ls="--" if kind == "dash" else "-")
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if align == "center" else x + 0.12
        ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color="black",
                fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=NAVY, text=None, tx=0, ty=0.12, size=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls="--" if dashed else "-",
                                shrinkA=2, shrinkB=2, mutation_scale=11))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", va="bottom", fontsize=size,
                color=GREY, style="italic")


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def jload(rel):
    return json.load(open(BASE / rel))


# ----------------------------------------------------------- design figures
def fig5_1():
    fig, ax = canvas(9, 5.0)
    layers = [(3.45, "Layer 1: physical", "OpenDSS power flow on IEEE-123. Produces the true state.",
               ["Feeder model\nand loads", "True state x_true", "Never modified"], "fill", NAVY),
              (1.75, "Layer 2: cyber", "Meters report z = h(x_true) + noise; messages travel over Kafka.",
               ["Meters (49 + head)", "Wire: z_delivered = z + a", "Attacks A1-A5 added\nhere only"], "new", ORANGE),
              (0.05, "Layer 3: inference", "Control centre estimates the state and looks for attacks.",
               ["Fusion detector\n(BDD + GRU autoencoder)", "Peer checks (new)\nV-peer and P-peer",
                "Alarm if any\ndetector fires"], "fill", NAVY)]
    for y, title, sub, boxes, kind, col in layers:
        box(ax, 0.1, y, 8.8, 1.55, "", "band" if kind == "fill" else "new")
        ax.text(0.3, y + 1.32, title, fontsize=12, fontweight="bold", color=col, va="center")
        ax.text(0.3, y + 1.05, sub, fontsize=9.5, va="center")
        for i, t in enumerate(boxes):
            k = "plain" if (i == 2 and y > 3) else ("new" if (y < 1 and i == 1) or (1 < y < 3 and i == 1) else "fill")
            box(ax, 0.4 + i * 2.85, y + 0.15, 2.55, 0.68, t, k, size=9.5, bold=(y < 1 and i == 2))
    arrow(ax, 4.5, 3.45, 4.5, 3.32); arrow(ax, 4.5, 1.75, 4.5, 1.62)
    save(fig, "fig5_1_layers.png")


def fig5_2():
    fig, ax = canvas(10, 6.0)
    # upstream physics
    for i, t in enumerate(["IDEAL load\nprofiles (91 nodes)", "OpenDSS\nIEEE-123 solver", "Meter model\nz = h(x) + noise\n(168 channels)"]):
        box(ax, 0.2 + i * 2.2, 4.75, 1.9, 0.95, t, "fill", size=9)
        if i: arrow(ax, 0.2 + i * 2.2 - 0.3, 5.22, 0.2 + i * 2.2, 5.22)
    box(ax, 0.2, 3.1, 1.9, 0.95, "Producer\n(one message per\nmeter per minute)", "fill", size=9)
    arrow(ax, 5.45, 4.75, 2.1, 3.85)
    # attack planner (new)
    box(ax, 7.4, 4.75, 2.4, 0.95, "Attack planner (new)\noffline, with OpenDSS\nA3/A4/A5 plans", "new", size=9)
    box(ax, 4.0, 3.1, 1.9, 0.95, "Injector\nA1/A2 bias live;\nadds stored A3-A5", "new", size=9)
    arrow(ax, 7.4, 5.0, 5.9, 3.9, dashed=True, text="plans (.npz)", tx=0.35, ty=0.05)
    arrow(ax, 2.1, 3.57, 4.0, 3.57, text="telemetry.raw")
    # detector container
    box(ax, 0.2, 0.15, 7.2, 2.35, "", "band")
    ax.text(0.35, 2.28, "Detector consumer (10-step buffer)", fontsize=10, fontweight="bold", va="center")
    box(ax, 0.4, 1.05, 2.1, 0.95, "Fusion (unchanged)\nBDD + GRU-AE\np > 0.3934", "fill", size=9)
    box(ax, 2.7, 1.05, 2.1, 0.95, "V-peer (new)\nvoltage vs other\nmeters, 5/15/30 min", "new", size=9)
    box(ax, 5.0, 1.05, 2.1, 0.95, "P-peer (new)\npower vs other\nmeters, per minute", "new", size=9)
    box(ax, 2.7, 0.27, 2.1, 0.55, "Alarm if any fires", "new", size=9.5, bold=True)
    for x in (1.45, 3.75, 6.05):
        arrow(ax, x, 1.05, 3.75 + (x - 3.75) * 0.25, 0.82)
    arrow(ax, 4.95, 3.1, 4.4, 2.5, text="telemetry.delivered", tx=0.75, ty=-0.02)
    # kafka + view
    box(ax, 7.9, 2.35, 1.9, 1.7, "Kafka cluster\n(3 brokers, Docker)\n\ntelemetry.raw\ntelemetry.delivered\nlabels · alarms", "kafka", size=8.5)
    box(ax, 7.9, 0.27, 1.9, 1.25, "Live view\n3 score lines,\n5 attack classes", "fill", size=9)
    arrow(ax, 4.8, 0.55, 7.9, 0.85, text="alarms", tx=0.2, ty=0.05)
    arrow(ax, 5.9, 3.57, 7.9, 3.4, dashed=True, color=GREY)
    ax.text(0.2, 5.82, "All stage-to-stage messages go through the Kafka topics shown on the right; arrows show the logical flow.",
            fontsize=8.5, color=GREY, style="italic")
    save(fig, "fig5_2_components.png")


def fig5_3():
    fig, ax = canvas(10, 5.35)
    box(ax, 0.1, 0.1, 7.3, 5.15, "", "dash")
    ax.text(0.25, 5.02, "Docker Compose project (docker compose up --build). One container per stage.",
            fontsize=9, color=GREY)
    # three-broker cluster
    box(ax, 0.3, 3.15, 6.9, 1.6, "", "band")
    ax.text(0.45, 4.55, "Kafka cluster, apache/kafka:4.3.2 pinned by digest, KRaft", fontsize=10.5, fontweight="bold", va="center")
    for i, port in enumerate((9092, 9094, 9096)):
        box(ax, 0.5 + i * 2.25, 3.3, 2.05, 1.0, f"kafka-{i + 1}\nbroker + controller\nhost port {port}", "kafka", size=8.5)
    ax.text(3.75, 2.98, "every partition on all three brokers; a write needs 2 copies (min.insync.replicas = 2)",
            fontsize=9, color=GREY, style="italic", ha="center", va="center")
    # stage containers
    stages = [("producer\n(OpenDSS)", "fill"), ("injector\n(A1-A5 on\nthe wire)", "new"), ("detector\n(fusion,\npeer\nchecks)", "new"),
              ("consumer\n(operator\ntwin)", "fill"), ("viewer\n(records\na GIF)", "fill"), ("report\n(writes the\nresult files)", "fill")]
    for i, (t, k) in enumerate(stages):
        box(ax, 0.3 + i * 1.17, 1.15, 1.05, 1.45, t, k, size=9)
        arrow(ax, 0.82 + i * 1.17, 2.6, 0.82 + i * 1.17, 2.83, color=GREY)
    box(ax, 0.3, 0.25, 6.9, 0.6, "kafka-init creates the topics; the stages start once all three brokers are healthy",
        "plain", size=9)
    # host side
    ax.text(7.65, 5.02, "Host machine", fontsize=9, color=GREY)
    box(ax, 7.6, 3.55, 2.3, 1.2, "live_view.py\nlive window, reads the\ncluster on localhost", "fill", size=9.5)
    box(ax, 7.6, 2.0, 2.3, 1.2, "data folders\nmounted read-only\ninto the stages", "plain", size=9.5)
    box(ax, 7.6, 0.25, 2.3, 1.4, "gen_demo_attacks.py\nrun once, offline\n-> demo_attacks/*.npz\n(read by the injector)", "new", size=9)
    arrow(ax, 7.6, 4.15, 7.2, 4.0, color=GREY)
    arrow(ax, 7.6, 2.6, 7.0, 2.2, dashed=True, color=GREY)
    ax.plot([7.6, 2.0], [1.0, 1.0], color=ORANGE, lw=1.3, ls="--")
    arrow(ax, 2.0, 1.0, 2.0, 1.17, color=ORANGE)
    save(fig, "fig5_3_deployment.png")


def fig5_4():
    fig, ax = canvas(10, 3.6)
    steps = ["Delivered\nmeasurements z\n(165 weighted\nchannels)", "Add pseudo\nmeasurements (84)\nand regulator-side\nrows (112)",
             "Start from\npower-flow\nsolution\n(pseudo_pf)", "Gauss-Newton\nnull-space step,\ntruncated SVD,\ntrust region", "Residuals r\nJ, LNR\nbdd_score"]
    for i, t in enumerate(steps):
        box(ax, 0.1 + i * 2.0, 1.75, 1.75, 1.4, t, "fill", size=9)
        if i: arrow(ax, 0.1 + i * 2.0 - 0.25, 2.45, 0.1 + i * 2.0, 2.45)
    box(ax, 4.1, 0.2, 2.7, 0.9, "246 exact zero-injection constraints\nenforced by the null-space method", "plain", size=9)
    box(ax, 7.0, 0.2, 2.9, 0.9, "bdd_score = max(J / 402.36, LNR / 5.664)\nlimits = 99th percentile of clean val", "plain", size=9)
    arrow(ax, 5.45, 1.1, 6.95, 1.75, dashed=True, color=GREY)
    arrow(ax, 8.95, 1.75, 8.95, 1.1, dashed=True, color=GREY)
    save(fig, "fig5_4_estimator.png")


def fig5_5():
    fig, ax = canvas(10, 3.7)
    steps = ["Input window\n10 steps x\n168 channels\n(scaled)", "GRU encoder\nhidden size 96\n(3 gate blocks)", "Bottleneck\ntanh, d = 32\n(3,104 weights)",
             "Linear decoder\nrebuilds the\ncurrent step\n(168 values)", "Error\n|z - z_hat| / r_c\nper channel"]
    for i, t in enumerate(steps):
        box(ax, 0.1 + i * 2.0, 2.0, 1.75, 1.3, t, "fill", size=9)
        if i: arrow(ax, 0.1 + i * 2.0 - 0.25, 2.65, 0.1 + i * 2.0, 2.65)
    box(ax, 3.2, 0.25, 3.6, 0.95, "Mahalanobis distance of the error vector\nLedoit-Wolf shrinkage, lambda = 0.0007", "plain", size=9)
    box(ax, 7.2, 0.25, 2.7, 0.95, "Score s_maha\n(mean 162.93, std 71.44\non clean validation)", "plain", size=9)
    arrow(ax, 8.9, 2.0, 5.0, 1.2, dashed=True, color=GREY); arrow(ax, 6.8, 0.72, 7.2, 0.72)
    ax.text(0.15, 0.7, "84,968 weights.\nTrained on clean\ndata only.", fontsize=9, va="center")
    save(fig, "fig5_5_autoencoder.png")


def fig5_6():
    fig, ax = canvas(10, 3.9)
    box(ax, 0.1, 2.4, 1.9, 0.95, "bdd_score\nfrom state estimator", "fill", size=9)
    box(ax, 0.1, 0.75, 1.9, 0.95, "s_maha\nfrom GRU autoencoder", "fill", size=9)
    box(ax, 2.5, 2.4, 2.0, 0.95, "z_bdd = (bdd - 0.652)\n/ 0.140", "plain", size=9)
    box(ax, 2.5, 0.75, 2.0, 0.95, "z_maha = (s - 162.93)\n/ 71.44", "plain", size=9)
    box(ax, 5.0, 1.5, 2.1, 1.05, "logit = 0.366 z_bdd\n+ 0.395 z_maha\n- 2.464", "fill", size=9)
    box(ax, 7.6, 1.5, 2.3, 1.05, "p = sigmoid(logit)\nfusion alarm if\np > 0.3934", "fill", size=9.5, bold=True)
    for y in (2.87, 1.22):
        arrow(ax, 2.0, y, 2.5, y)
    arrow(ax, 4.5, 2.87, 5.0, 2.3); arrow(ax, 4.5, 1.22, 5.0, 1.75); arrow(ax, 7.1, 2.02, 7.6, 2.02)
    ax.text(7.6, 1.2, "This alarm is one of the three inputs\nof the combined alarm (Figure 5.7).", fontsize=8.5,
            color=ORANGE, style="italic", va="top")
    save(fig, "fig5_6_fusion.png")


def fig5_7():
    fig, ax = canvas(10, 4.7)
    box(ax, 0.1, 3.3, 2.3, 1.0, "Readings of all OTHER\nmeters at minute t\n(meter m left out)", "fill", size=9)
    box(ax, 2.9, 3.3, 2.3, 1.0, "Ridge model for meter m\n(fitted on clean\ntraining data only)", "fill", size=9)
    box(ax, 5.7, 3.3, 1.9, 1.0, "Predicted\nreading of m", "fill", size=9)
    box(ax, 5.7, 1.9, 1.9, 0.9, "Actual reading\nof meter m", "plain", size=9)
    box(ax, 8.0, 2.55, 1.9, 1.15, "error =\n(actual - predicted)\n/ normal error size", "plain", size=9)
    arrow(ax, 2.4, 3.8, 2.9, 3.8); arrow(ax, 5.2, 3.8, 5.7, 3.8)
    arrow(ax, 7.6, 3.8, 8.0, 3.35); arrow(ax, 7.6, 2.35, 8.0, 2.85)
    box(ax, 4.2, 0.2, 2.6, 1.15, "V-peer (voltage)\nclip to +/-3, mean over\n5, 15, 30 min; thr 3.979", "new", size=9)
    box(ax, 7.1, 0.2, 2.8, 1.15, "P-peer (active power)\nsingle-minute error;\nthr 5.298", "new", size=9)
    arrow(ax, 8.6, 2.55, 5.6, 1.35); arrow(ax, 9.0, 2.55, 8.5, 1.35)
    box(ax, 0.1, 0.2, 3.6, 1.6, "Combined alarm\nalarm = fusion OR V-peer OR P-peer\npeer checks need 3 minutes in a row;\nthe meter with the largest error\nis named as the suspect", "fill", size=9)
    arrow(ax, 4.2, 0.78, 3.7, 0.9)
    ax.text(0.1, 4.5, "A tampered meter cannot change what its untouched neighbours imply, so the tampering appears whole as error.",
            fontsize=9, style="italic", color=GREY)
    save(fig, "fig5_7_peer_check.png")


def fig6_1():
    fig, ax = canvas(9.5, 4.3)
    for i, t in enumerate(["IDEAL homes\n119 homes\n1-minute data", "25 homes averaged\nper node ->\n91 node profiles",
                           "OpenDSS power\nflow, 86,400\nsteps", "Meter model\n+ noise (seed 42)\n-> Corpus A"]):
        box(ax, 0.1 + i * 2.4, 2.9, 2.0, 1.15, t, "fill", size=9.5)
        if i: arrow(ax, 0.1 + i * 2.4 - 0.4, 3.47, 0.1 + i * 2.4, 3.47)
    box(ax, 0.1, 1.65, 9.2, 0.6, "Split by calendar day: 42 days / 9 days / 9 days", "plain", size=10, bold=True)
    arrow(ax, 8.3, 2.9, 8.3, 2.25)
    for i, t in enumerate(["Train\n60,480 rows\n(clean)", "Validation\n12,960 rows", "Test\n12,960 rows"]):
        box(ax, 0.6 + i * 3.0, 0.1, 2.3, 1.05, t, "fill", size=10)
        arrow(ax, 1.75 + i * 3.0, 1.65, 1.75 + i * 3.0, 1.15)
    save(fig, "fig6_1_corpus.png")


def fig6_2():
    t = np.arange(61); s, e = 10, 50
    step = ((t >= s) & (t < e)).astype(float)
    ramp = np.where((t >= s) & (t < e), (t - s + 1) / (e - s), 0.0)
    pulse = np.where((t >= s) & (t < e) & (((t - s) // 5) % 2 == 0), 1.0, 0.0)
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for a, (name, y) in zip(axs, [("pulse", pulse), ("ramp", ramp), ("step", step)]):
        a.step(t, y, where="post", color=ORANGE, lw=1.8) if name != "ramp" else a.plot(t, y, color=ORANGE, lw=1.8)
        a.set_title(name, fontsize=10); a.set_xlabel("step"); a.set_ylim(-0.08, 1.15)
    axs[0].set_ylabel("attack size / level")
    fig.tight_layout(); save(fig, "fig6_2_shapes.png")


# ------------------------------------------------------------ result charts
def fig8_1():
    txt = open(BASE / "reports/placement_quality_vs_baselines.txt").read()
    a1 = re.search(r"--- A1\. branch_diversity vs Random ---\n.*?\n((?:\s+\d+ \|.*\n)+)", txt).group(1)
    a4 = re.search(r"--- A4\. Hybrid vs CommunityDetection ---\nbranch_diversity:\n.*?\n((?:\s+\d+ \|.*\n)+)", txt).group(1)
    r = np.array([[float(v) for v in l.split("|")[:4]] for l in a1.strip().splitlines()])
    c = np.array([[float(l.split("|")[i]) for i in (0, 2, 5)] for l in a4.strip().splitlines()])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(r[:, 0], r[:, 1], "-o", color=NAVY, lw=1.8, label="Hybrid (used)")
    ax.errorbar(r[:, 0], r[:, 2], yerr=r[:, 3], fmt="--s", color=GREY, lw=1.6, capsize=3, label="Random (mean ± sd, 10 seeds)")
    ax.plot(c[:, 0], c[:, 1], ":^", color=ORANGE, lw=1.6, label="Community detection, 12 groups")
    ax.plot(c[:, 0], c[:, 2], ":v", color=GOLD, lw=1.6, label="Community detection, merged to 4")
    ax.axvline(49, color=GREY, ls=":", lw=1); ax.text(49.7, 12, "49 meters", color=GREY, fontsize=9)
    ax.set_xlabel("Number of meters"); ax.set_ylabel("Branches covered (of 53)"); ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig8_1_placement.png")


TABLE_8_4 = {  # Table 8.4 of the report (test split, FAR near 1%)
    "Physics\n(any step)": (10, 0.607, 0.784), "AE max over\nchannels": (7, 0.753, None),
    "AE trailing\nmean": (7, 0.867, 0.746), "AE Mahalanobis\n(seed 1234)": (11, 0.874, 0.770),
    "Fusion\n(seed 1234)": (12, 0.873, 0.811)}


def fig8_2():
    rob = jload("validation/detect_robustness_summary.json")
    m, s = rob["fusion"]["win"]
    names = [k for k in TABLE_8_4 if "max" not in k] + ["Fusion\n(5 seeds)"]
    vals = [TABLE_8_4[k][0] for k in names[:-1]] + [m]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(names, vals, color=[GREY] * 4 + [NAVY], width=0.6)
    ax.errorbar([4], [m], yerr=[s], color="black", capsize=5, lw=1.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (s + 0.25 if v == m else 0.25), f"{v:g}", ha="center", fontsize=9.5)
    ax.axhline(18, color=ORANGE, ls="--", lw=1.2); ax.text(4.3, 18.3, "18 attack windows in test", color=ORANGE, ha="right", fontsize=9)
    ax.set_ylim(0, 20); ax.set_ylabel("Windows detected (of 18)")
    save(fig, "fig8_2_windows.png")


def fig8_3():
    names = list(TABLE_8_4)
    step = [TABLE_8_4[k][1] for k in names]; win = [TABLE_8_4[k][2] for k in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    b1 = ax.bar(x - w / 2, step, w, color=NAVY, label="Step AUC")
    b2 = ax.bar(x + w / 2, [v or 0 for v in win], w, color=ORANGE, label="Window AUC")
    for b, v in list(zip(b1, step)) + list(zip(b2, win)):
        if v: ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, color=GREY, ls=":", lw=1); ax.set_ylim(0.4, 1.0); ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("AUC (test split)"); ax.legend(frameon=False, loc="upper left")
    save(fig, "fig8_3_auc.png")


def fig8_4():
    rob = jload("validation/detect_robustness_summary.json")
    maha = [r["win"] for r in rob["maha_seeds"]]
    fus = [r["win"] for r in rob.get("fusion_seeds", [])] or [12, 10, 10, 12, 9]
    runs = np.arange(1, len(maha) + 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.scatter(runs - 0.06, maha, s=90, color=GREY, label=f"AE Mahalanobis (mean {np.mean(maha):.1f})", zorder=3)
    ax.scatter(runs + 0.06, fus, s=90, marker="s", color=NAVY, label=f"Fusion (mean {np.mean(fus):.1f})", zorder=3)
    ax.set_xticks(runs); ax.set_ylim(8, 13); ax.set_xlabel("Training run (AE seed)"); ax.set_ylabel("Windows detected (of 18)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig8_4_seeds.png")


def fig8_5():
    d, mse = [8, 16, 32, 48], [0.3473, 0.2703, 0.1770, 0.116]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    bars = ax.bar([str(v) for v in d], mse, color=[GREY, GREY, NAVY, ORANGE], width=0.55)
    for b, v in zip(bars, mse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9.5)
    ax.text(2, 0.215, "used", ha="center", color=NAVY, fontsize=9.5)
    ax.text(3, 0.17, "better fit,\nworse detector\n(AUC 0.743 vs 0.753)", ha="center", color=ORANGE, fontsize=9)
    ax.set_ylim(0, 0.4); ax.set_xlabel("Bottleneck size d"); ax.set_ylabel("Clean validation MSE")
    save(fig, "fig8_5_bottleneck.png")


def windows(lb):
    att = lb.sort_values("timestamp")
    out, cur = [], None
    for t, a, c in zip(att.timestamp, att.attacked, att.attack_class.fillna("")):
        if a and cur and cur[2] == c and t == cur[1] + 1:
            cur[1] = t
        elif a:
            cur = [t, t, c]; out.append(cur)
        else:
            cur = None
    return out


def fig8_6():
    rd = BASE / "streaming/demo_runs/runB_realistic"
    al = pd.read_csv(rd / "alarms.csv").sort_values("timestamp"); lb = pd.read_csv(rd / "labels.csv")
    thr = float(np.load(BASE / "streaming/detector_fit_cache.npz")["thr_f"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for s, e, c in windows(lb):
        ax.axvspan(s - 0.5, e + 0.5, color=HUES.get(c, "#eee"), alpha=0.7, lw=0)
        ax.text((s + e) / 2, 1.03, c, ha="center", fontsize=9)
    ax.plot(al.timestamp, al.fusion_score, color="black", lw=0.9, label="fusion score")
    f = al[al.alarm.astype(bool)]
    ax.plot(f.timestamp, f.fusion_score, "o", ms=3, color="#c0392b", label="alarm")
    ax.axhline(thr, color=GREY, ls="--", lw=1, label=f"threshold ({thr:.3f})")
    ax.set_ylim(0, 1.1); ax.set_xlabel("stream index (1 step = 1 simulated minute)"); ax.set_ylabel("fusion score")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    save(fig, "fig8_6_runB.png")


def fig8_7():
    d = jload("validation/overall_eval_summary.json")
    rows = [("Existing learned detector (AE Mahalanobis, any minute)", "q0.99 K1 AE only (existing)", GREY),
            ("Peer checks only (V | P)", "q0.99 K3 V|P", ORANGE), ("Combined (V | P | fusion)", "q0.99 K3 V|P|AE", NAVY)]
    types = ["A1", "A2", "A3", "A4", "A5"]; x = np.arange(len(types)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for i, (lab, key, col) in enumerate(rows):
        v = [d[key]["report_by_type_%"][t] for t in types]
        b = ax.bar(x + (i - 1) * w, v, w, color=col, label=f"{lab}, FAR {d[key]['FAR_%']:.2f}%")
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 1.2, f"{vv:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["A1 random\nbias (27)", "A2 one meter\n(27)", "A3 load\nredistribution (58)", "A4 stealthy\n(6)", "A5 stealthy\n(9)"], fontsize=9)
    ax.set_ylim(0, 125); ax.set_ylabel("Held-out windows detected (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig8_7_catch_rates.png")


def fig8_8():
    new = jload("validation/a2_persist_summary.json")["3"]; old = jload("validation/a2_baseline_summary.json")
    durs = ["15", "60", "240"]; x = np.arange(3); w = 0.2
    vals = [("Existing (AE Mahalanobis, any minute)", [old["hit1_duration"][f"A2 {d}"] for d in durs], GREY),
            ("Chance, existing", [old["K1_chance"][d] for d in durs], "#c9ccd2"),
            ("V-peer (new)", [new["bench_duration"][f"A2 {d}"] for d in durs], NAVY),
            ("Chance, V-peer", [new["chance_window"][d] for d in durs], "#9fb0c8")]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for i, (lab, v, col) in enumerate(vals):
        b = ax.bar(x + (i - 1.5) * w, [100 * a for a in v], w, color=col, label=lab)
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, 100 * vv + 1.2, f"{100 * vv:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{d} minutes" for d in durs]); ax.set_ylim(0, 112)
    ax.set_ylabel("A2 windows detected (%)"); ax.set_xlabel("Attack duration (135-window A2 benchmark)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "fig8_8_a2_benchmark.png")


def fig8_9():
    d = jload("validation/overall_eval_summary.json")
    new, old = d["q0.99 K3 V|P|AE"], d["q0.99 K1 AE only (existing)"]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, keys, title in [(axs[0], ["A3 2%", "A3 5%", "A3 10%"], "by size (share of metered load moved)"),
                            (axs[1], ["A3 15", "A3 60", "A3 240"], "by duration (minutes)")]:
        src = "report_by_type_level_%" if "%" in keys[0] else "report_by_type_duration_%"
        x = np.arange(3); w = 0.36
        for i, (lab, dd, col) in enumerate([("Existing (AE Mahalanobis), FAR 0.95%", old, GREY), ("Combined system, FAR 1.13%", new, NAVY)]):
            v = [dd[src].get(k, np.nan) for k in keys]
            b = ax.bar(x + (i - 0.5) * w, v, w, color=col, label=lab)
            for bb, vv in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, [k.split()[1] for k in keys]); ax.set_title(title, fontsize=10); ax.set_ylim(0, 118)
    axs[0].set_ylabel("A3 windows detected (%)"); axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); save(fig, "fig8_9_a3.png")


def fig8_10():
    rd = BASE / "streaming/demo_runs/runC_all"
    al = pd.read_csv(rd / "alarms.csv").sort_values("timestamp"); lb = pd.read_csv(rd / "labels.csv")
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    for s, e, c in windows(lb):
        ax.axvspan(s - 0.5, e + 0.5, color=HUES.get(c, "#eee"), alpha=0.75, lw=0)
        ax.text((s + e) / 2, 4.08, c, ha="center", fontsize=9.5)
    ymax = 4.0
    for col, lab, colr in [("fusion_ratio", "fusion (BDD + AE)", "black"), ("v_ratio", "voltage peer check", PURPLE),
                           ("p_ratio", "power peer check", GOLD)]:
        ax.plot(al.timestamp, np.clip(al[col], 0, ymax), color=colr, lw=1.0, label=lab)
    top = np.clip(al[["fusion_ratio", "v_ratio", "p_ratio"]].max(1), 0, ymax)
    m = al.alarm.astype(bool)
    ax.plot(al.timestamp[m], top[m], "o", ms=2.6, color="#c0392b", label="alarm (any detector)")
    ax.axhline(1.0, color=GREY, ls="--", lw=1, label="threshold (each detector)")
    ax.set_ylim(0, 4.3); ax.set_ylabel("score / own threshold (clipped at 4)")
    ax.set_xlabel("stream index (1 step = 1 simulated minute)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig8_10_runC.png")


def fig8_11():
    rd = BASE / "streaming/demo_runs/runC_failover"
    al = pd.read_csv(rd / "alarms.csv").sort_values("timestamp").reset_index(drop=True)
    lb = pd.read_csv(rd / "labels.csv")
    kills = jload("streaming/demo_runs/runC_failover/broker_kill.json")["kills"]
    start = int(lb.timestamp.min())
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    for s, e, c in windows(lb):
        ax.axvspan(s - 0.5, e + 0.5, color=HUES.get(c, "#eee"), alpha=0.75, lw=0)
        ax.text((s + e) / 2, 4.08, c, ha="center", fontsize=9.5)
    ymax = 4.0
    for col, lab, colr in [("fusion_ratio", "fusion (BDD + AE)", "black"), ("v_ratio", "voltage peer check", PURPLE),
                           ("p_ratio", "power peer check", GOLD)]:
        ax.plot(al.timestamp, np.clip(al[col], 0, ymax), color=colr, lw=1.0, label=lab)
    top = np.clip(al[["fusion_ratio", "v_ratio", "p_ratio"]].max(1), 0, ymax)
    m = al.alarm.astype(bool)
    ax.plot(al.timestamp[m], top[m], "o", ms=2.6, color="#c0392b", label="alarm (any detector)")
    ax.axhline(1.0, color=GREY, ls="--", lw=1, label="threshold (each detector)")
    # steps the detector scored while a broker was down (alarm rows at kill and at restart)
    for k in kills:
        a, b = al.timestamp.iloc[k["alarms_at_kill"]], al.timestamp.iloc[min(k["alarms_at_restart"], len(al) - 1)]
        ax.axvspan(a - 0.5, b + 0.5, facecolor="none", edgecolor=GREY, hatch="///", lw=0)
        ax.text((a + b) / 2, 3.55, f"{k['killed']}\ndown", ha="center", fontsize=8.5, color=GREY,
                bbox=dict(fc="white", ec="none", pad=1.0))
    # the first kill as the producer saw it
    k = kills[0]
    p0, p1 = start + k["raw_steps_at_kill"], start + k["raw_steps_at_restart"]
    ax.plot([p0, p1], [3.0, 3.0], color=ORANGE, lw=2.2)
    ax.text((p0 + p1) / 2, 3.08, f"{k['killed']} down while the producer published these steps", ha="center",
            va="bottom", fontsize=8.5, color=ORANGE, bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_ylim(0, 4.3); ax.set_ylabel("score / own threshold (clipped at 4)")
    ax.set_xlabel("stream index (1 step = 1 simulated minute)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig8_11_failover.png")


ALL = [fig5_1, fig5_2, fig5_3, fig5_4, fig5_5, fig5_6, fig5_7, fig6_1, fig6_2,
       fig8_1, fig8_2, fig8_3, fig8_4, fig8_5, fig8_6, fig8_7, fig8_8, fig8_9, fig8_10, fig8_11]

if __name__ == "__main__":
    for f in ALL:
        f()
