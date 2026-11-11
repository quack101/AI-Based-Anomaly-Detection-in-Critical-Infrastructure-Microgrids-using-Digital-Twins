"""
streaming/live_view.py -- live matplotlib viewer for the Kafka detector demo.
Consumes "alarms" (one record per scored step from detector_consumer) and
"labels" (the injector's ground truth, used ONLY to shade attacked steps and
to colour the top-channel bars) and redraws about every 0.5 s.

Read-only: it never publishes, and nothing it shows feeds back into the
detector. It runs in its own consumer group ("live_view" + KAFKA_GROUP_SUFFIX)
and never commits offsets, so it cannot take messages from detector_consumer
or shift any other reader's position.

  python -m streaming.live_view [--xmin T0 --xmax T1] [--stop-at T] [--idle-exit S]
                                [--save-gif [PATH]] [--headless] [--title TEXT]
  python -m streaming.live_view --replay streaming/demo_runs/runB_realistic --save-gif
        replay a recorded run's alarms.csv / labels.csv (no broker needed);
        the figure is titled as a REPLAY, never as live

Top panel: fusion score vs stream index, the frozen fusion threshold
(detector_fit_cache.npz thr_f = 0.3934), red dots where alarm=True, attacked
steps shaded by class. Bottom panel: how often each channel was the top
channel over the last 20 scored steps -- top_channel is the channel with the
largest contribution to the AE-Mahalanobis distance (streaming/detector_core.py
score_vector), not a state-estimation residual. Text box: current step, alarm,
attack label, running alarm count.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

BASE = Path(__file__).resolve().parent.parent
DEFAULT_GIF = BASE / "streaming" / "demo_runs" / "live_demo.gif"
TOP_N = 20

INK, MUTED, GRID, SURFACE = "#1f2933", "#6b7785", "#e3e7ec", "#fcfcfb"
ALARM = "#c0392b"
HUES = {"A1": "#d9822b", "A2": "#2b7bb9"}      # same as demo_tools.analyze timeline.png
OTHER_HUE = "#8e6c8a"
BAR_IDLE = "#c5ccd4"


def load_threshold():
    return float(np.load(BASE / "streaming" / "detector_fit_cache.npz")["thr_f"])


def _as_list(v):
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return []
    return str(v).split(";")


class LiveState:
    """Everything the figure shows, built only from alarms/labels records."""

    def __init__(self):
        self.alarms = {}       # t -> alarm record
        self.labels = {}       # t -> label record

    def add(self, topic, rec):
        t = int(rec["timestamp"])
        if topic == "alarms":
            self.alarms[t] = rec
        elif topic == "labels":
            self.labels[t] = dict(rec, targets=_as_list(rec.get("targets")),
                                  attacked_meters=_as_list(rec.get("attacked_meters")))

    def series(self):
        ts = sorted(self.alarms)
        return (np.array(ts, dtype=float),
                np.array([float(self.alarms[t]["fusion_score"]) for t in ts]),
                np.array([bool(self.alarms[t]["alarm"]) for t in ts], dtype=bool))

    def attack_spans(self, upto=None):
        """Contiguous attacked label runs -> [(start, end, class)], only up to step
        `upto`: the injector runs ahead of the detector, so labels for steps not yet
        scored must not be drawn (it would show an attack before the detector sees it)."""
        spans = []
        for t in sorted(self.labels):
            if upto is not None and t > upto:
                break
            lb = self.labels[t]
            if not int(lb.get("attacked", 0)):
                continue
            cls = lb.get("attack_class") or "attack"
            if spans and spans[-1][1] == t - 1 and spans[-1][2] == cls:
                spans[-1] = (spans[-1][0], t, cls)
            else:
                spans.append((t, t, cls))
        return spans

    def recent_top(self, n=TOP_N):
        """[(channel, count, attack_class)] over the last n scored steps, most frequent
        first; attack_class = the label's class at the latest of those steps where the
        channel was top AND in that step's label targets, else None (never targeted)."""
        ts = sorted(self.alarms)[-n:]
        counts = Counter(self.alarms[t]["top_channel"] for t in ts)
        targeted = {}
        for t in ts:
            ch = self.alarms[t]["top_channel"]
            if t in self.labels and ch in self.labels[t]["targets"]:
                targeted[ch] = self.labels[t].get("attack_class") or "attack"
        return [(ch, c, targeted.get(ch)) for ch, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def status(self):
        if not self.alarms:
            return None
        t = max(self.alarms)
        rec = self.alarms[t]
        lb = self.labels.get(t)
        return {"t": t, "alarm": bool(rec["alarm"]), "fusion": float(rec["fusion_score"]),
                "top": rec["top_channel"],
                "attacked": None if lb is None else bool(int(lb.get("attacked", 0))),
                "attack_class": None if lb is None else lb.get("attack_class"),
                "n_alarm": int(sum(bool(r["alarm"]) for r in self.alarms.values())),
                "n_scored": len(self.alarms)}


class LiveView:
    def __init__(self, state, thr, title, xlim=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        self.plt, self.state, self.thr, self.xlim = plt, state, thr, xlim
        self.fig = plt.figure(figsize=(12, 7), facecolor=SURFACE)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[4, 1.25],
                                   hspace=0.45, wspace=0.08, left=0.09, right=0.98, top=0.85, bottom=0.08)
        self.ax = self.fig.add_subplot(gs[0, :])
        self.axb = self.fig.add_subplot(gs[1, 0])
        self.axt = self.fig.add_subplot(gs[1, 1])
        self.fig.suptitle(title, x=0.09, ha="left", fontsize=11, color=INK)
        try:
            self.fig.canvas.manager.set_window_title("FDI digital twin -- live detector view")
        except Exception:
            pass

        ax = self.ax
        ax.set_facecolor(SURFACE)
        self.line, = ax.plot([], [], color=INK, lw=1.4, label="fusion score")
        self.dots, = ax.plot([], [], "o", ms=5, color=ALARM, mec=SURFACE, mew=1, ls="none", label="alarm")
        ax.axhline(thr, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(1.0, thr, f" threshold {thr:.3f}", transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                fontsize=8, color=MUTED)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("stream index (1 step = 1 simulated minute)", fontsize=9, color=MUTED)
        ax.set_ylabel("fusion score", fontsize=9, color=MUTED)
        self._style(ax)
        ax.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="fusion score"),
                           Line2D([], [], marker="o", ms=5, color=ALARM, ls="none", label="alarm"),
                           Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="threshold"),
                           Patch(color=HUES["A1"], alpha=0.25, label="A1 attacked (label)"),
                           Patch(color=HUES["A2"], alpha=0.25, label="A2 attacked (label)")],
                  loc="lower left", fontsize=8, frameon=False, ncol=5, bbox_to_anchor=(0, 1.07))
        self._span_artists = []
        self.axt.axis("off")
        self.txt = self.axt.text(0.04, 0.95, "waiting for alarms ...", va="top", ha="left", family="monospace",
                                 fontsize=10.5, color=INK, transform=self.axt.transAxes,
                                 bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRID))
        self.note = None

    @staticmethod
    def _style(ax):
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    def redraw(self):
        st = self.state
        ts, fus, alarm = st.series()
        ax = self.ax
        self.line.set_data(ts, fus)
        self.dots.set_data(ts[alarm], fus[alarm])

        for a in self._span_artists:
            a.remove()
        self._span_artists = []
        for s, e, cls in st.attack_spans(upto=max(st.alarms) if st.alarms else -1):
            hue = HUES.get(cls, OTHER_HUE)
            self._span_artists.append(ax.axvspan(s - 0.5, e + 0.5, color=hue, alpha=0.16, lw=0, zorder=0))
            self._span_artists.append(ax.text((s + e) / 2, 1.01, cls, ha="center", va="bottom", fontsize=9,
                                              color=INK, transform=ax.get_xaxis_transform()))
        if self.xlim:
            ax.set_xlim(*self.xlim)
        elif len(ts):
            ax.set_xlim(ts[0] - 2, max(ts[-1], ts[0] + 60) + 5)

        axb = self.axb
        axb.cla()
        self._style(axb)
        axb.grid(True, axis="x", color=GRID, lw=0.7)
        axb.grid(False, axis="y")
        rows = st.recent_top()[:10]
        cur = st.status()
        if rows:
            names = [r[0] for r in rows][::-1]
            counts = [r[1] for r in rows][::-1]
            colors = [HUES.get(r[2].split("+")[0], OTHER_HUE) if r[2] else BAR_IDLE for r in rows][::-1]
            y = np.arange(len(names))
            axb.barh(y, counts, height=0.72, color=colors, edgecolor=SURFACE, linewidth=2)
            axb.set_yticks(y, [("> " if cur and n == cur["top"] else "") + n for n in names],
                           fontsize=8, color=INK)
            for lbl, n in zip(axb.get_yticklabels(), names):
                if cur and n == cur["top"]:
                    lbl.set_fontweight("bold")
            for yi, c in zip(y, counts):
                axb.text(c, yi, f" {c}", va="center", fontsize=8, color=MUTED)
            n_last = min(TOP_N, len(st.alarms))
            axb.set_xlim(0, max(n_last, 1) * 1.08)
            axb.xaxis.set_major_locator(MaxNLocator(integer=True))
        axb.set_title(f"top channel (largest AE-Mahalanobis contribution), last {TOP_N} scored steps   "
                      f"(> = current;  coloured = attack target in the label, by class)",
                      fontsize=9, color=INK, loc="left")
        axb.set_xlabel("steps as top channel", fontsize=9, color=MUTED)

        if cur:
            att = "n/a" if cur["attacked"] is None else ("YES " + (cur["attack_class"] or "")).strip() \
                if cur["attacked"] else "no"
            self.txt.set_text(f"step     {cur['t']}\n"
                              f"fusion   {cur['fusion']:.3f}\n"
                              f"ALARM    {'YES' if cur['alarm'] else 'no'}\n"
                              f"attack   {att}\n"
                              f"top ch   {cur['top']}\n"
                              f"alarms   {cur['n_alarm']} / {cur['n_scored']} scored")
            self.txt.get_bbox_patch().set_edgecolor(ALARM if cur["alarm"] else GRID)
            self.txt.get_bbox_patch().set_linewidth(2.5 if cur["alarm"] else 1)
            self.txt.get_bbox_patch().set_facecolor("#fbeceb" if cur["alarm"] else "white")
        self.fig.canvas.draw_idle()

    def set_note(self, text):
        if self.note is None:
            self.note = self.axt.text(0.04, 0.02, "", va="bottom", ha="left", fontsize=8.5, color=MUTED,
                                      transform=self.axt.transAxes)
        self.note.set_text(text)
        self.fig.canvas.draw_idle()


class GifRecorder:
    """Frames are quantised to a palette image as they are grabbed (1 byte/pixel),
    so a 400-step run stays well under a few hundred MB in memory."""

    def __init__(self, fig, path, dpi=72, fps=8):
        self.fig, self.path, self.dpi, self.fps = fig, Path(path), dpi, fps
        self.frames = []

    def grab(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=self.dpi, facecolor=self.fig.get_facecolor())
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    def finish(self, hold_last_s=3.0):
        if not self.frames:
            print("live_view: no frames recorded, GIF not written", flush=True)
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dur = [int(1000 / self.fps)] * len(self.frames)
        dur[-1] = int(hold_last_s * 1000)
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=dur, loop=0)
        try:
            shown = self.path.resolve().relative_to(BASE)    # repo path is non-ASCII; keep cp1252 consoles happy
        except ValueError:
            shown = self.path
        print(f"live_view: wrote {len(self.frames)} frames to {shown} "
              f"({self.path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return self.path


def kafka_source(group_id):
    from kafka import KafkaConsumer
    from .kafka_config import BOOTSTRAP_SERVERS, TOPIC_ALARMS, TOPIC_LABELS
    c = KafkaConsumer(TOPIC_ALARMS, TOPIC_LABELS, bootstrap_servers=BOOTSTRAP_SERVERS, group_id=group_id,
                      auto_offset_reset="earliest", enable_auto_commit=False,
                      value_deserializer=lambda m: json.loads(m.decode("utf-8")))

    def poll():
        out = []
        for tp, recs in c.poll(timeout_ms=100).items():
            out += [(tp.topic, r.value) for r in recs]
        return out
    return poll, c.close


def replay_source(run_dir, steps_per_poll=2):
    """Recorded run -> the same (topic, record) stream, steps_per_poll scored steps
    per redraw (labels up to that step go first). Returns None when exhausted."""
    import pandas as pd
    run_dir = Path(run_dir)
    al = pd.read_csv(run_dir / "alarms.csv").sort_values("timestamp")
    lb = pd.read_csv(run_dir / "labels.csv").sort_values("timestamp")
    al_rows = al.to_dict("records")
    lb_rows = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
               for r in lb.to_dict("records")]
    pos = {"a": 0, "l": 0}

    def poll():
        if pos["a"] >= len(al_rows):
            return None
        batch = al_rows[pos["a"]:pos["a"] + steps_per_poll]
        pos["a"] += len(batch)
        t_hi = batch[-1]["timestamp"]
        out = []
        while pos["l"] < len(lb_rows) and lb_rows[pos["l"]]["timestamp"] <= t_hi:
            out.append(("labels", lb_rows[pos["l"]]))
            pos["l"] += 1
        return out + [("alarms", r) for r in batch]
    return poll, lambda: None


def run(view, poll, stop_at=None, idle_exit=None, recorder=None, headless=False, interval=0.5,
        close_on_end=False):
    """Poll -> redraw loop. Ends on stop_at (an alarm at or past that step has been
    drawn), idle_exit seconds without a record once data has arrived, an exhausted
    replay, or the window being closed. Returns the reason."""
    plt = view.plt
    last_rx, reason = time.time(), None
    while reason is None:
        recs = poll()
        if recs is None:
            reason = "replay finished"
            break
        for topic, rec in recs:
            view.state.add(topic, rec)
        if recs:
            last_rx = time.time()
            view.redraw()
            if recorder:
                recorder.grab()
        if stop_at is not None and view.state.alarms and max(view.state.alarms) >= stop_at:
            reason = f"reached step {stop_at}"
        elif idle_exit and view.state.alarms and time.time() - last_rx > idle_exit:
            reason = f"no new records for {idle_exit:.0f} s"
        elif not headless and not plt.fignum_exists(view.fig.number):
            reason = "window closed"
        elif headless:
            time.sleep(interval)
        else:
            plt.pause(interval)
    print(f"live_view: stopped ({reason}); {len(view.state.alarms)} scored steps, "
          f"{sum(bool(r['alarm']) for r in view.state.alarms.values())} alarms shown", flush=True)
    if recorder:
        recorder.finish()
    if not headless and reason != "window closed" and not close_on_end:
        view.set_note(f"stream ended ({reason}) -- close the window to exit")
        plt.show()
    return reason


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", default=None, help="recorded run dir (alarms.csv + labels.csv); no broker")
    ap.add_argument("--replay-rate", type=int, default=2, help="scored steps per redraw in --replay")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--stop-at", type=int, default=None, help="stop once this stream index is scored")
    ap.add_argument("--idle-exit", type=float, default=None, help="stop after S s with no new record")
    ap.add_argument("--interval", type=float, default=0.5, help="redraw period, seconds")
    ap.add_argument("--save-gif", nargs="?", const=str(DEFAULT_GIF), default=None,
                    help=f"record the plot to a GIF (default {DEFAULT_GIF.relative_to(BASE)})")
    ap.add_argument("--gif-fps", type=float, default=8)
    ap.add_argument("--headless", action="store_true", help="no window (Agg); use with --save-gif")
    ap.add_argument("--close-on-end", action="store_true")
    ap.add_argument("--title", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold line (default: detector_fit_cache.npz thr_f)")
    a = ap.parse_args()

    import matplotlib
    if a.headless:
        matplotlib.use("Agg")
    thr = a.threshold if a.threshold is not None else load_threshold()
    if a.replay:
        title = a.title or (f"REPLAY of recorded Kafka run {Path(a.replay).name} (not live) -- "
                            f"fusion detector, threshold {thr:.3f}")
        poll, close = replay_source(a.replay, a.replay_rate)
    else:
        group = "live_view" + os.environ.get("KAFKA_GROUP_SUFFIX", "")
        title = a.title or f"LIVE -- fusion detector on topic 'alarms' (consumer group {group})"
        poll, close = kafka_source(group)
        print(f"live_view: consuming alarms + labels as group {group!r}", flush=True)
    xlim = (a.xmin, a.xmax) if a.xmin is not None and a.xmax is not None else None
    view = LiveView(LiveState(), thr, title, xlim)
    rec = GifRecorder(view.fig, a.save_gif, fps=a.gif_fps) if a.save_gif else None
    if not a.headless:
        view.plt.show(block=False)
    try:
        run(view, poll, stop_at=a.stop_at, idle_exit=a.idle_exit, recorder=rec, headless=a.headless,
            interval=a.interval, close_on_end=a.close_on_end)
    finally:
        close()


if __name__ == "__main__":
    main()
