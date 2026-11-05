# Full live demo (preset C): A1–A5 with the combined detector

## What changed

Before this, the live demo (Run B) showed only A1 and A2. The fusion detector caught A1 but missed A2. The full demo now runs all five attack classes in one live Kafka run. Its detector is the combined system from `reports/overall_detection.md`:

**alarm = fusion (BDD + GRU-AE) OR V-peer OR P-peer**

Each detector uses its own threshold, set at 99% of clean validation data. The two peer checks only alarm after 3 consecutive steps above threshold.

| piece | file |
|---|---|
| attack schedule (shared) | `streaming/demo_schedule.py` |
| A3/A4/A5 plans, built offline with OpenDSS | `validation/gen_demo_attacks.py` → `streaming/demo_attacks/C_80440_A*.npz` |
| injector adds the stored vectors in transit (no circuit access) | `attacks/precomputed.py`, `streaming/injector.py` |
| peer detectors on the live stream | `streaming/peer_monitor.py`, `streaming/detector_consumer.py` |
| live view: three score/threshold lines, all five classes | `streaming/live_view.py` |
| analysis: per-detector clean alarm rate, `detected_by` per window | `streaming/demo_tools.py` |
| launcher | `streaming/run_demo.py --run C`, or double-click `RUN_DEMO_C.bat` |

### How A3/A4/A5 enter a live stream

The injector sits on the wire (Layer 2) and must never solve a power flow. So A3, A4 and A5 are planned **offline**, the way a real attacker would plan them with their own grid model. Planning uses the library's own `attacks.library.attack()` on the clean test telemetry for the same stream indices. During the run, the injector only adds the stored vector: `z_delivered = z_true + a_t`. The import-boundary tests still pass.

## Schedule (S = 80440; 1 step = 1 simulated minute)

| steps | attack | size |
|---|---|---|
| 0–80 | clean | – |
| 80–140 | A1: 20 random channels | P/Q ±10% of the reading, V ±0.01 pu |
| 180–240 | A2: one meter (M042 P, Q, V) | same |
| 280–340 | A3: zone load redistribution, power-flow consistent, full access | 10% of metered P (≈5% of feeder load) moved between two groups of 4 meters |
| 380–410 | A4: kept-subspace stealthy (passes BDD by construction) | 1e-4 pu state shift |
| 450–480 | A5: Newton-corrected A4 | 1e-4 pu |
| 480–520 | clean | – |

### Why S = 80440, not 81640 (Run B's start)

At 81640 the clean corpus itself drives V-peer over threshold for about 20 steps (rows +412…+436, meter M044_V_C). That would show up as a false alarm between A4 and A5. Over the whole test corpus this happens 0.6% of the time, which is the design rate. I scanned every 520-step start: 80440 (2018-05-29, 20:40) has no V-peer, P-peer or AE alarm on clean data, with every score at most 0.99× its threshold. The live producer uses the same loads with a different noise draw, so a stray clean alarm is still possible but unlikely.

### Why A3 at 10%

I replayed this exact schedule offline using the nine bench A3 plans (zone, full access, 60 steps):

- At 5% of metered P, the combined system caught **5/9**.
- At 10%, it caught **9/9**.

The demo uses 10%. **For the paper, report the bench numbers** (`reports/a3_a4_a5_detection.md`, `reports/overall_detection.md`) rather than the demo. The demo is an illustration, not the evaluation.

## Pre-flight check done in the VM (no OpenDSS / Kafka there)

I ran the real `DetectorConsumer` + `PeerMonitor` + `PrecomputedAttack` + `Injector` + `demo_tools analyze` + `live_view` over this schedule. Inputs were the clean corpus rows plus:

- A1/A2 built to the same spec;
- A3 from a 10% bench plan;
- A4/A5 from the 0.01 pu bench vectors scaled to 1e-4 pu.

Fusion was replaced by a stub, so in this check the fusion line is flat and only the peer checks fire. (The pre-flight outputs were removed once the real live runs below replaced them.)

| window | detected | delay (steps) | by |
|---|---|---|---|
| A1 | yes | 2 | V-peer, P-peer (fusion also catches A1 live: Run B 60/60 steps) |
| A2 | yes | 4 | V-peer; localised to M042_V |
| A3 | yes | 2 | V-peer, P-peer |
| A4 | yes | 2 | V-peer, P-peer |
| A5 | yes | 2 | V-peer, P-peer |
| clean (135 steps) | 0 alarms | | |

The injector check confirmed the delivered change equals the plan exactly. The window boundaries are correct and `stealth_flag` is set. `tests/test_live_view.py` and `tests/test_import_boundaries.py` pass.

Note: V-peer uses trailing means of up to 30 steps, so it keeps alarming for up to ~30 steps after an attack ends. `analyze` excludes those spill-over steps from the clean false-alarm rate. Mention this if someone asks about the alarm dots after A4/A5.

## Live result (Windows, 5 Nov 2026, `streaming/demo_runs/runC_all/`)

Setup: live OpenDSS producer, corpus noise S1, real Kafka, real fusion core, 511 scored steps, pipeline 899 s. The A3/A4/A5 plans were built by `gen_demo_attacks` in about 5 min.

| window | detected | delay | alarm steps | first-alarm channel (in targets?) | by |
|---|---|---|---|---|---|
| A1 | yes | 0 | 59/60 | M039_Q (yes) | fusion, V-peer |
| A2 | yes | 14 | 46/60 | M042_V (yes) | V-peer |
| A3 | yes | 3 | 9/60 | M038_P (yes) | P-peer |
| A4 | yes | 0 | 30/30 | M044_P_B (yes) | fusion, V-peer, P-peer |
| A5 | yes | 0 | 30/30 | M048_P (yes) | fusion, V-peer, P-peer |
| clean (126 steps) | **0 alarms** (fusion 0%, V-peer 0%, P-peer 0%) | | | | |

Compared with the corpus, the live clean telemetry matches: noise ratio median 1.00, and the largest systematic offset is 0.21σ.

A3 is the weakest case. P-peer sits right at its threshold, so it alarms in bursts (9 of 60 steps) rather than continuously. That is consistent with the bench result for A3 (64–71%).

To run again, double-click `RUN_DEMO_C.bat` (about 15 min, since the plans are already built). Close the live window at the end.
