"""
streaming/live_view.py (no Kafka broker needed): state built from alarms/labels
records, the replay source against the recorded Run B, headless GIF recording,
and the run_demo --run B preset keeping Run B's settings.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest

from streaming.live_view import LiveState, LiveView, GifRecorder, replay_source, run, load_threshold

BASE = Path(__file__).resolve().parent.parent
RUN_B = BASE / "streaming" / "demo_runs" / "runB_realistic"


def _alarm(t, f, alarm, top):
    return {"timestamp": t, "fusion_score": f, "alarm": alarm, "top_channel": top}


def _label(t, cls=None, targets=()):
    return {"timestamp": t, "attacked": 1 if cls else 0, "attack_class": cls,
            "targets": list(targets), "attacked_meters": []}


def _state():
    st = LiveState()
    for t in range(100, 130):
        cls = "A1" if 110 <= t < 120 else ("A2" if 122 <= t < 125 else None)
        st.add("labels", _label(t, cls, ["M001_P"] if cls else []))
        st.add("alarms", _alarm(t, 0.9 if cls else 0.1, bool(cls), "M001_P" if cls else "M002_Q"))
    return st


def test_attack_spans_split_by_class_and_gap():
    assert _state().attack_spans() == [(110, 119, "A1"), (122, 124, "A2")]


def test_labels_ahead_of_the_detector_are_not_shaded():
    st = _state()                               # alarms scored up to 129
    for t in range(130, 160):                   # injector already labelled an attack at 140..149
        st.add("labels", _label(t, "A2" if 140 <= t < 150 else None, ["M003_P"]))
    assert st.attack_spans(upto=max(st.alarms)) == [(110, 119, "A1"), (122, 124, "A2")]
    st.add("alarms", _alarm(145, 0.2, False, "M003_P"))
    assert st.attack_spans(upto=max(st.alarms))[-1] == (140, 145, "A2")


def test_recent_top_counts_last_20_and_target_class():
    rows = dict((ch, (c, cls)) for ch, c, cls in _state().recent_top(20))
    # steps 110..129: 13 attacked (top M001_P, targeted; last under A2), 7 clean (top M002_Q)
    assert rows == {"M001_P": (13, "A2"), "M002_Q": (7, None)}


def test_targeted_bar_keeps_its_class_after_the_attack_ends():
    st = LiveState()
    for t in range(0, 30):
        cls = "A1" if t < 10 else None
        st.add("labels", _label(t, cls, ["M001_P"] if cls else []))
        st.add("alarms", _alarm(t, 0.9 if cls else 0.1, bool(cls), "M001_P" if t < 15 else "M002_Q"))
    rows = dict((ch, (c, cls)) for ch, c, cls in st.recent_top(20))
    # steps 10..29 are clean; M001_P was top at 10..14 but only targeted before 10 -> not coloured
    assert rows == {"M001_P": (5, None), "M002_Q": (15, None)}
    rows = dict((ch, (c, cls)) for ch, c, cls in st.recent_top(25))
    assert rows["M001_P"] == (10, "A1")       # still A1-coloured while the current step is clean


def test_status_reports_current_step_and_running_alarm_count():
    s = _state().status()
    assert s["t"] == 129 and s["alarm"] is False and s["attacked"] is False
    assert s["n_alarm"] == 13 and s["n_scored"] == 30


def test_label_strings_from_csv_are_split():
    st = LiveState()
    st.add("labels", {"timestamp": 5, "attacked": 1, "attack_class": "A2", "targets": "M042_P;M042_Q",
                      "attacked_meters": None})
    assert st.labels[5]["targets"] == ["M042_P", "M042_Q"] and st.labels[5]["attacked_meters"] == []


@pytest.mark.skipif(not (RUN_B / "alarms.csv").exists(), reason="recorded Run B artifacts absent")
def test_replay_of_run_b_matches_its_summary(tmp_path):
    summary = json.load(open(RUN_B / "summary.json"))
    st = LiveState()
    view = LiveView(st, load_threshold(), "test")
    poll, _ = replay_source(RUN_B, steps_per_poll=50)
    gif = tmp_path / "x.gif"
    reason = run(view, poll, recorder=GifRecorder(view.fig, gif), headless=True, interval=0)
    assert reason == "replay finished"
    s = st.status()
    assert s["n_scored"] == summary["n_scored"] and s["n_alarm"] == summary["n_alarm_total"]
    spans = st.attack_spans()
    assert [(a, b, c) for a, b, c in spans] == [(w["start"], w["end"], w["class"]) for w in summary["windows"]]
    from PIL import Image
    assert Image.open(gif).n_frames == -(-summary["n_scored"] // 50)


def test_threshold_is_the_frozen_fusion_threshold():
    assert abs(load_threshold() - 0.3934018284216307) < 1e-12


def test_run_b_preset_keeps_run_b_settings():
    from streaming.run_demo import RUN_PRESETS
    rb = json.load(open(RUN_B / "run.json")) if (RUN_B / "run.json").exists() else None
    p = RUN_PRESETS["B"]
    assert p["run_name"] != "runB_realistic"           # never overwrites the recorded run
    if rb:
        assert (p["start"], p["noise"], p["warmup"], p["sleep"]) == (rb["start"], rb["noise"], rb["warmup"],
                                                                     rb["sleep_s"])
