# FDIA v1 detector comparison — residual baseline (BDD) vs GRU

Simulated IEEE-123 feeder, dataset `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix`. Detectors read telemetry `z` only; labels are used
only to pick clean timesteps for calibration and to score. Operating
point: **1% per-timestep false-alarm on clean val** (each detector score
at its 99th percentile over clean val; applied unchanged to test).

## Operating point / thresholds

- **BDD (recalibrated, `config/bdd_thresholds_val1pct.json`):** J99 376.15 → **402.36**,
  LNR99 5.269 → **5.664** (99th percentile of J / max-LNR over the 11,380 clean
  val timesteps; estimator pass 10,954 s, 100% converged). The OR rule
  (flag if J > J99 **or** LNR > LNR99) with each score at its own 1% point
  gives a **1.67% clean-val FAR**, so this detector sits slightly above a true
  1% point by construction.
- **GRU (`validation/detect_gru_weights.npz`):** threshold 7.508 (99th
  percentile of the single score over clean val). 1-layer GRU, hidden 96,
  W=10, 92,616 params; 21 epochs (early-stopped), best clean-val MSE 0.2917;
  training 255 s; gradient check 3.2e-6.

## Headline (test split)

| metric | BDD (val-1%) | GRU |
|---|---|---|
| clean test FAR | 1.76% | 1.04% |
| attacked-timestep detection | 6.52% | 2.47% |
| windows detected (of 18) | 13 | 8 |
| window detection rate | 0.722 | 0.444 |
| median detection delay (steps) | 11 | 14 |
| clean 60-step blocks flagged (of 254) | 82 | 61 |

The two detectors are **not at an identical clean FAR**: the BDD OR rule runs
at ~1.76% test FAR, the GRU at ~1.04%. The BDD detects more at a higher
false-alarm cost.

## Window detection by group (detection rate)

| dimension | value | n | BDD | GRU |
|---|---|---|---|---|
| type | A1 | 12 | 0.833 | 0.500 |
| type | A2 | 3 | 0.667 | 0.000 |
| type | A3 | 3 | 0.333 | 0.667 |
| class | PQ_k20 | 3 | 0.667 | 0.000 |
| class | PQ_k5 | 3 | 1.000 | 0.667 |
| class | V_k20 | 3 | 1.000 | 1.000 |
| class | meter (A2) | 3 | 0.667 | 0.000 |
| class | mixed_k5 | 3 | 0.667 | 0.333 |
| class | zone (A3) | 3 | 0.333 | 0.667 |
| level | L0 (pq .05 / v .005) | 5 | 0.600 | 0.400 |
| level | L1 (pq .10 / v .01) | 5 | 0.800 | 0.200 |
| level | L2 (pq .20 / v .02) | 5 | 1.000 | 0.600 |
| level | A3 2% full | 1 | 0.000 | 0.000 |
| level | A3 10% full | 1 | 1.000 | 1.000 |
| level | A3 5% partial | 1 | 0.000 | 1.000 |
| profile | step | 6 | 0.667 | 0.500 |
| profile | ramp | 6 | 1.000 | 0.667 |
| profile | pulse | 6 | 0.500 | 0.167 |
| duration | 15 | 6 | 0.500 | 0.333 |
| duration | 60 | 6 | 0.667 | 0.333 |
| duration | 240 | 6 | 1.000 | 0.667 |

## Where the GRU beats the baseline, and where it does not

- **GRU higher:** A3 zone load-redistribution (0.667 vs 0.333) and the A3 5%
  partial-access window (detected vs missed). The GRU also runs at a lower
  clean FAR (1.04% vs 1.76%) and fewer clean-block false alarms (61 vs 82).
- **GRU lower everywhere else:** overall window detection (0.444 vs 0.722),
  attacked-timestep detection (2.47% vs 6.52%), A1 (0.500 vs 0.833), A2
  single-meter bias (0.000 vs 0.667), PQ_k20 and the meter class (0.000 vs
  0.667), every level and profile except where tied, and all three
  durations. Its median delay is also longer (14 vs 11).
- **Tied:** V_k20 (both 1.000) and A3 10% full (both 1.000); A3 2% full missed
  by both.

Numbers only; no claim about any attack's detectability beyond these rates.

## Suspicious / unknown

- Neither detector reaches a true 1% test FAR cleanly: the BDD is ~1.76%
  (OR of two 1% thresholds), and the GRU's 1.04% is close but the clean
  test residuals run hotter than clean val, as already seen in Part C of the
  prior task.
- Per-timestep detection is low for both (≤6.5%); window-level detection is
  carried by long windows (240-step windows detected best). Short 15-step
  windows are the weak point for both.
- The GRU's next-step predictor adapts to a sustained shift within ~W steps,
  so step/ramp attacks mostly register only at onset — consistent with its
  longer delay and low per-timestep rate. Whether a longer window or an
  autoregressive-residual variant would help is unknown (not tested).
