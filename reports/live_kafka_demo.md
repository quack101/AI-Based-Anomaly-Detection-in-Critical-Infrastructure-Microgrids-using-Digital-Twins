# Live Kafka integration — estimator + adopted detector on the stream

Branch `streaming-integration` (from `detectors`). The offline benchmark is
unchanged: no file under `estimation/`, `observability/`, `config/`, the saved
AE weights or the fusion weights was modified, and `attacks/library.py` only
gained code (+89 / −0 lines).

## What was added

| piece | where |
|---|---|
| scoring core (loads the frozen estimator, AE, Mahalanobis fit, fusion) | `streaming/detector_core.py` |
| Kafka detector stage: buffer → 168-vector → score → `alarms` | `streaming/detector_consumer.py` |
| head channels on the wire (multiphase only) | `streaming/producer.py` (`head_entries`, `HEAD` message) |
| `alarms` topic | `streaming/kafka_config.py` |
| per-timestep A1/A2 bias + `BiasContext` | `attacks/library.py` (`apply_bias_timestep`) |
| injector A1/A2 path (A3/A4/A5 still `NotImplementedError`) | `streaming/injector.py` |
| run / dump / gate / analyse / diagnose tooling | `streaming/run_demo.py`, `streaming/demo_tools.py` |

Every live stage is started with `REGISTRY_VARIANT=multiphase` and its own
OpenDSS process; default variants and `tests/test_streaming.py` are unchanged.
The detector's Mahalanobis μ/Σ and fusion standardisation were never saved as
files, so they are recomputed deterministically from the frozen AE and the
committed data and cached in `streaming/detector_fit_cache.npz` (verified in
Phase 1; this is not a retrain).

## Gates

- **Phase 1 (scoring core vs offline), PASS.** 20 test timesteps: J within
  1.3e-7 relative of the committed log, bdd_score 1.0e-7, fusion 1.3e-8, alarms
  identical; the full test split reproduces `detect_final_summary.json` exactly
  (point Mahalanobis 11/18, fusion 12/18, same AUCs and per-window decisions).
- **Phase 2 in-process equivalence, PASS** (vector 0.0, fusion 2.3e-8).
- **Real-Kafka gate (Run A), PASS.** 1,450 messages read back from the real
  broker; 20 clean scored timesteps: vector vs the producer's own truth vector
  2.3e-13, fusion vs direct `score_vector` 3.8e-9, J 1.3e-7 relative, alarms
  identical. (The 20 steps are 20 timesteps of 50 messages each.)
- **Injection check (Run A).** Over all 400 steps the delivered messages differ
  from the producer's output on exactly the labelled target channels and nowhere
  else (120 attacked steps; A1 20 channels, A2 3 channels).

## Run A — pipeline functional test, not detector performance

Noise-free, start at t = 0. It exists to show the plumbing and the real-broker
gate above. **No false-alarm or detection rate is reported from it.**
Artifacts: `streaming/demo_runs/runA_functional/`.

## Run B — live OpenDSS producer, corpus noise, test-split start

Producer: live OpenDSS, noise drawn with the corpus (ae0fcfb / `sigma_overrides_s1`)
sensor sigmas, start at stream index 81640 (= test row 8200, 16:40) after a
120-step unpublished warm-up. 400 steps, producer sleep 0.2 s. A1: 20 channels,
mixed class, step, P/Q bias 10 % of |z|, V bias 0.01 pu, 60 steps. A2: one meter
(M042: P, Q, V), step, same levels, 60 steps. One run, one noise draw. Threshold:
the frozen fusion threshold 0.3934.

| item | result |
|---|---|
| steps scored | 391 (first 9 are the W = 10 warm-up) |
| clean-stretch alarm rate | **3 / 253 = 1.19 %** (steps outside windows and the 9 steps after each) |
| clean-stretch false alarms | t = 81676, 81677, 81678 (one 3-step excursion, fusion up to 0.62, top channel M047_P) |
| A1 (t = 81780..81839) | **detected**, delay 0, alarms on 60 / 60 steps; top channel at first alarm M026_P ∈ attacked targets; 77 % of alarm steps have a top channel in the attacked set |
| A2 (t = 81900..81959) | **not detected**, 0 / 60 alarm steps; fusion median 0.096, max 0.249 (< 0.393) |

Artifacts: `streaming/demo_runs/runB_realistic/` (`alarms.csv`, `labels.csv`,
`summary.json`, `timeline.png`, `diagnosis.json`).

This is a demonstration of the pipeline with one A1 and one A2 window, not a
performance estimate; the offline benchmark (`reports/detector_robustness.md`)
is the performance result. The A2 miss is in line with it: A2 was the hardest
family offline (0/3 for the point score, 1/3 for the fusion, seed dependent).

Run C (replay of the test corpus through Kafka) was the fallback for a clean-stretch
alarm rate above 3 % that could not be explained; it was **not needed** (1.19 %).

## Why the early runs over-alarmed

Two separate causes, both found:

1. **Registry sigmas used as noise scale (first Kafka smoke test).** The
   producer's sensor layer draws noise from the registry sigmas, which commit
   81ff01a widened ~100–200× for estimator *weighting*. Live noise was ~100× the
   corpus (J ≈ 2e8, every step alarmed). Fix: the producer takes the corpus noise
   sigmas (`sigma_overrides_s1`) for the multiphase live run.
2. **Producer and detector sharing one OpenDSS instance (in-process test only).**
   `DetectorCore`'s power-flow start rewrites all 91 loads and resets the taps in
   the circuit it lives in; a producer in the same process then has its 42 undriven
   loads at pseudo values instead of the authored nominal ones. Same producer
   settings: interleaved in one process 10/21 alarms (test-split start) and 9/21
   (t = 0); producer vectors generated first and scored afterwards 0/21 and 0/21
   (`streaming/diagnose_inprocess.py`, output in
   `runB_realistic/diagnosis_inprocess.txt`). Over Kafka each stage is its own
   process, so the live path is not affected.

Checked and ruled out, with the numbers:

- **(a) start time / hour of day:** offline, the corpus's own early-morning rows
  alarm 0/231 (test rows 0–239) and 0/31 (day-1 train rows 9–39); the Run-B span
  offline alarms 3/391 (0.8 %).
- **(b) load profile:** the Kafka producer drives the same 49 loads and leaves
  the same 42 at nominal as the corpus generator; the contamination above is the
  only way it differed.
- **(c) noise scale (with S1 sigmas):** live published telemetry vs the corpus at
  the same 400 rows: systematic offset median 0.053 σ (p95 0.15 σ, max 0.21 σ),
  noise-scale ratio std(diff)/(√2·σ) median 0.989 (5th–95th pct 0.94–1.05).

Live clean scores vs the offline scores at the same test rows (253 clean steps,
offline rows 8209–8599): fusion median 0.086 vs 0.086, p95 0.175 vs 0.188; J
median 188 vs 192; Mahalanobis median 160 vs 162.

## Known limits

- The producer's noise generator is unseeded in `main()` (system entropy), so a
  rerun draws different noise; the committed artifacts are the record of these runs.
- One A1 and one A2 window in one run: no confidence interval, no rate.
- The 3-step clean excursion near t = 81677 (top channel M047_P, a row the
  estimator down-weights for a structural reason) has not been investigated;
  cause **unknown**.
- A3/A4/A5 are not on the stream: they need circuit or estimator access that the
  injector layer must not have.
- `detector_core` needs 10 consecutive complete timesteps; an incomplete step is
  logged "not scored" and skipped, never filled.
