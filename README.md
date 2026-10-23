# Cyber-Physical Digital Twin for FDI Detection on an IEEE-123 Feeder

A streaming digital twin of an unbalanced AC distribution feeder, built to study
False Data Injection (FDI) attack detection on smart-meter telemetry.

Synthetic household load profiles are streamed through Kafka into an OpenDSS
IEEE-123 model, which solves one AC power flow per timestamp. Attacks are injected
on the wire between producer and consumer; detection runs on the resulting telemetry.

The research and architecture specification is **`RES.md`** — it is the single source
of truth for design decisions, the attack taxonomy, and the phased roadmap.
`CLAUDE.md` holds the agent operating rules for implementation.

## Why this setup

Most FDI literature targets balanced DC transmission systems, evaluated offline.
This project sits in three gaps at once: **unbalanced three-phase AC**, **distribution
rather than transmission**, and **streaming rather than batch**.

The novelty is the combination and the setting, not the individual components.

## Architecture

```
load profiles → producer → Kafka → consumer → OpenDSS AC solve → x_true
                              ↑                                     ↓
                      attack injector                        sensor layer
                                                                    ↓
                                                    detection: temporal + physics
```

Three layers are kept strictly separate:

| Layer | What it is |
|---|---|
| **L1 physical** | OpenDSS ground-truth state — never modified by an attacker |
| **L2 cyber** | Sampled, noised, transmitted sensor readings — the only attack surface |
| **L3 inference** | Estimation, residuals, anomaly scores |

## Measured facts

Established by Phase 0 against the compiled circuit, not assumed:

| | |
|---|---|
| Buses | 132 |
| Phase-nodes | 278 |
| State dimension `dim(x)` | 555 |
| Load objects | 91 (49 metered, 42 unmetered) |
| Current measurements `m` | 98 |

`m` = 98 against `n` = 555 — the feeder is far from observable, so classical state
estimation is not yet possible. This is a real property of distribution systems,
not a design defect, and it is why the primary detector supplies *temporal*
redundancy rather than spatial.

## Setup

What a fresh clone needs before anything in `streaming/` or `tests/` runs:

1. **Python packages:** `python -m pip install -r requirements.txt`. Keep `kafka-python`
   below 3; 3.0.x crashes on Windows.
2. **Docker Desktop** (for the Kafka broker in `streaming/docker-compose.yml`). Start it and
   wait for "Engine running". Only the live demo needs it; the tests and `--replay` do not.
3. **Data folders.** They are not in git (`.gitignore`: `dataset/`, `*.csv`,
   `simulation/phase6_corpus/`), so get them from the team and place them exactly here:

   | Folder | Contents | Size |
   |---|---|---|
   | `dataset/node_loads_91/` | 91 synthetic load profiles `node_<load>.csv` (`timestamp, P_kW, Q_kvar`), 60 days at 1 min | ~255 MB |
   | `dataset/fdia_v1/` | `train/val/test_z.csv`, `*_labels.csv`, `manifest.json` | ~260 MB |
   | `simulation/phase6_corpus/` | `train/val/test` `_telemetry.csv`, `_labels.csv`, `_ground_truth.csv`, `_x_true.npz` | ~590 MB |

   When unzipping, check that the files sit directly in these folders, not in a nested
   folder of the same name. Quick check: `python -c "from streaming.producer import load_inputs; load_inputs()"`
   raises if `node_loads_91` is missing or misplaced.
4. **Check:** `python -m pytest tests -q`. The three `test_placement_regression.py` tests
   also need a local `artifacts/` placement sweep, which is not in git; those may error on a
   fresh clone.

## Status

**Phase 0 — complete.** Twin instrumented and validated: 1440/1440 timestamps
converged, `V_pu` ∈ [0.980, 1.046], losses 1.89% of feeder-head load, max solve
1.5 ms. Validated end-to-end against a live Kafka broker.

Phase 0 also confirmed the known data-layer defect: metered loads run ~14.7× light
(household kW substituted for feeder-scale spot loads), masked at feeder aggregate
by static unmetered load.

**Phase 1 — complete.** Data layer: `telemetry.raw` carries registry channels only
(`tests/test_streaming.py` fails if an unregistered field — e.g. household
`Voltage`/`Global_intensity`/`Sub_metering_*` — is published), and the 42 unmetered
loads enter the estimator as explicit pseudo-measurements with a σ estimated from
held-out data.

Phases 2–13 (measurement model, observability, state estimator, attack injector,
detectors, evaluation) are specified in `RES.md` Deliverable 10.

### Live detector demo (Kafka)

The frozen estimator (`estimator-v5-hfix`) and the adopted detector (physics BDD +
AE-Mahalanobis fusion) run on the stream as `streaming/detector_consumer.py`; the
injector applies A1/A2 additive-bias attacks (`attacks/library.py`
`apply_bias_timestep`). Every stage runs its own OpenDSS process with
`REGISTRY_VARIANT=multiphase`.

```
docker compose -f streaming/docker-compose.yml up -d
python -m streaming.run_demo runB_realistic --start 81640 --noise 1 --warmup 120
python -m streaming.run_demo runA_functional --start 0 --noise 0 --functional
docker compose -f streaming/docker-compose.yml down
```

Artifacts land in `streaming/demo_runs/<run>/` (alarms, labels, summary, timeline PNG).
See `reports/live_kafka_demo.md`.

### Live viewer (presentation demo)

One command runs Run B with a live window. It needs `kafka-python` 2.x
(`python -m pip install "kafka-python<3"`): 3.0.x crashes on Windows when a consumer closes
(`WinError 10038`).

```
python -m streaming.run_demo --run B              # full 400 steps (~7 min on the Run B machine, 13-25 min on a slower laptop)
python -m streaming.run_demo --run B --steps 150  # shorter; A1 starts at step 140, A2 needs >= 261
python -m streaming.run_demo --run B --save-gif   # also record streaming/demo_runs/live_demo.gif
```

`--run B` uses Run B's settings unchanged (multiphase, S1 corpus noise, start 81640 inside
the test split, 120-step warm-up, sleep 0.2 s, A1 at +140 and A2 at +260 for 60 steps
each). It checks for a broker at `localhost:9092` first, running
`docker compose -f streaming/docker-compose.yml up -d` if Docker is on PATH and no broker
answers. Then it starts the producer, injector, detector_consumer and
`streaming/live_view.py`. Output goes to `streaming/demo_runs/runB_live/`, so the recorded
`runB_realistic` is never overwritten. When the stream ends the window stays open until
you close it.

`streaming/live_view.py` reads `alarms` and `labels` in its own consumer group
(`live_view-<run>`, no offset commits), so it never takes messages from detector_consumer.
It redraws every 0.5 s and shows:

- **Top panel:** fusion score over time, the frozen threshold (0.393, from
  `detector_fit_cache.npz`), red dots where `alarm=True`, and attacked steps shaded by
  class from the labels.
- **Bottom panel:** how many of the last 20 scored steps each channel was the top channel, i.e.
  the channel contributing most to the AE-Mahalanobis distance (not a state-estimation residual).
  `>` marks the current one; a coloured bar means that channel was an attack target in
  the label, coloured by attack class.
- **Text box:** the current step, fusion score, alarm yes/no, attack label yes/no, top
  channel, and the running alarm count.

To run the viewer on its own against a running pipeline, use `python -m streaming.live_view`.

**No Docker on the demo machine:** play `streaming/demo_runs/live_demo.gif` (a live recording
of the full Run B made with `--save-gif`), or re-draw the recorded run without a broker:

```
python -m streaming.live_view --replay streaming/demo_runs/runB_realistic
python -m streaming.live_view --replay streaming/demo_runs/runB_realistic --headless --interval 0 --save-gif
```

A replay is titled "REPLAY of recorded Kafka run … (not live)" so nobody mistakes it for a
live run.

## Stack

OpenDSS via `opendssdirect.py` · Apache Kafka · Python · PyTorch (planned) ·
FastAPI + Three.js dashboard (planned)

## Layout

```
123Bus/                  IEEE-123 OpenDSS model
streaming/               producer, injector, detector, live viewer, run_demo launcher, docker-compose
streaming/demo_runs/     recorded demo runs + live_demo.gif
simulation/              OpenDSS driver, instrumentation, topology snapshot
reports/                 validation reports
requirements.txt         Python dependencies
RES.md                   research + architecture specification
CLAUDE.md                agent operating rules
```
