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

## Status

**Phase 0 — complete.** Twin instrumented and validated: 1440/1440 timestamps
converged, `V_pu` ∈ [0.980, 1.046], losses 1.89% of feeder-head load, max solve
1.5 ms. Validated end-to-end against a live Kafka broker.

Phase 0 also confirmed the known data-layer defect: metered loads run ~14.7× light
(household kW substituted for feeder-scale spot loads), masked at feeder aggregate
by static unmetered load.

**Phase 1 — in progress.** Fixing the data layer: rescale P to nodal nominal, derive
Q from nodal power factor, reject household `Voltage`/`Global_intensity`/`Sub_metering_*`
at the producer schema level, diversify load profiles, declare the 42 unmetered
loads as explicit pseudo-measurements.

Phases 2–13 (measurement model, observability, state estimator, attack injector,
detectors, evaluation) are specified in `RES.md` Deliverable 10.

## Stack

OpenDSS via `opendssdirect.py` · Apache Kafka · Python · PyTorch (planned) ·
FastAPI + Three.js dashboard (planned)

## Layout

```
123Bus/                  IEEE-123 OpenDSS model
streaming/               producer, consumer, docker-compose
simulation/              OpenDSS driver, instrumentation, topology snapshot
reports/                 validation reports
RES.md                   research + architecture specification
CLAUDE.md                agent operating rules
```
