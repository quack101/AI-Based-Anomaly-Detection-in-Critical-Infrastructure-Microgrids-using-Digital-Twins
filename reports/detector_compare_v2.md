# FDIA v1 detector comparison v2 — physics (BDD) vs reconstruction AE vs fusion

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix`; **not re-run** (physics scores reuse the saved
clean-val residuals and the saved test residual log). Detectors read
telemetry `z` only. Operating point: threshold = 99th percentile of the
score over clean val (1% val FAR), applied unchanged to test.

## Detectors

- **BDD (physics, single score):** `bdd_score = max(J/J99_val,
  maxLNR/LNR99_val)` with J99_val 402.36, LNR99_val 5.664. Threshold
  1.0389 (99th pct of bdd_score over clean val).
- **AE (reconstruction):** GRU encoder over W=10 steps → tanh bottleneck
  d → linear decode of the current step z_t (all 168 channels). d chosen
  from {8,16,32} by clean-val reconstruction MSE: **d=32** (MSE 0.177 vs
  0.270 at 16, 0.347 at 8), 84,968 params, 40 epochs, gradient check
  4.3e-6. Score **s_max = max_c |z−ẑ|/r_c** (val AUC 0.726 vs 0.690 for
  s_mean). Threshold 6.282. Each d-model trained ~365 s.

## Part C — ROC / AUC (test, per-timestep, attacked vs clean)

| detector | per-timestep AUC |
|---|---|
| BDD | 0.6065 |
| AE | 0.7529 |

ROC and window-detection-vs-window-FA curves saved as
`validation/detect_roc_{bdd,ae}.csv` and `detect_wincurve_{bdd,ae}.csv`.

## AE sustained-attack check (the old next-step GRU's failure mode)

On the 240-step **step-profile** attacked window (window 8): AE flagged
5.4% of its steps; onset block 0%, **mid-window block 20%**, end block 6%.
The reconstruction error does **not** decay to zero after onset (mid > onset),
confirming the AE keeps reacting through a sustained attack — unlike a
next-step predictor that adapts away the error. The absolute per-step rate
is still low.

## Part D — fusion and the matched-1% comparison

Thresholds at each detector's own 1% clean-val point: bdd 1.0389, ae 6.282.
Table over the rows scored by both detectors (t ≥ W−1):

| detector | clean test FAR | attacked-step detection | windows (of 18) | window rate | median delay | per-ts AUC | clean 60-blocks flagged |
|---|---|---|---|---|---|---|---|
| BDD | 1.13% | 4.68% | 10 | 0.556 | 12 | 0.606 | 63 |
| AE | 0.88% | 4.94% | 7 | 0.389 | 37 | 0.753 | 36 |
| fusion-OR | 2.00% | 8.73% | 12 | 0.667 | 12 | 0.689 | 79 |
| fusion-LR | — | — | — | — | — | — | — |

**fusion-LR (Rule 2) was not computed.** It trains a logistic regression on
val with features [bdd_score, ae_score] and label=attacked, which needs
`bdd_score` on the 1,580 **attacked** val timesteps. `val_residuals.csv`
holds **clean** val only, and the hard rule forbids re-running the
estimator, so the attacked-val physics score is unavailable. Reported as
blocked rather than fabricated or obtained by re-running the estimator.

## Window detection by attack family (detection rate at the 1% point)

| family | BDD | AE | fusion-OR |
|---|---|---|---|
| A1 (random bias) | 0.667 | 0.417 | — |
| A2 (single-meter bias) | 0.333 | 0.000 | — |
| A3 (zone redistribution) | 0.333 | 0.667 | — |
| class PQ_k5 | 1.000 | 0.667 | — |
| class V_k20 | 1.000 | 0.333 | — |
| class zone | 0.333 | 0.667 | — |

(Full per-group breakdowns: `validation/detect_compare_v2_breakdown_*.csv`,
`detect_phys_breakdown.csv`, `detect_ae_breakdown.csv`.)

## Which detector wins

- **By per-timestep AUC (threshold-free ranking): AE wins** (0.753 vs BDD
  0.606 vs fusion-OR 0.689), and at the lowest clean FAR (0.88%).
- **By window detection at the fixed 1% point: fusion-OR wins** (12/18),
  then BDD (10/18), then AE (7/18) — but fusion-OR runs at ~2% test FAR
  (the OR of two 1% rules), double the others, and the most clean-block
  false alarms (79).
- **Per family:** BDD is stronger on A1 and A2 (meter-level biases, which
  move the weighted residual directly); **AE is stronger on A3 zone
  redistribution** (0.667 vs 0.333) and the 5% partial-access A3 window.
  V_k20 favors BDD; PQ favors BDD.
- AE's high AUC but lower window detection at the 1% point reflects that its
  score ranks attacked above clean well, yet the fixed 99th-pct clean-val
  threshold leaves many attacked windows below the line; its median delay is
  also long (37 vs 12).

Numbers only.

## Suspicious / unknown

- The three detectors are **not at an identical test FAR** (BDD 1.13%, AE
  0.88%, fusion-OR 2.00%); comparisons are at the specified 1%-val operating
  point, not a matched test FAR.
- AE per-step detection stays low (≤5%) even with a high AUC — the attacked
  and clean score distributions overlap near the 99th-pct threshold despite
  separating well by rank. Whether a lower operating point or a per-channel
  (not max) decision rule would raise window detection is **unknown** (not
  tested).
- fusion-LR is **unknown** — not computed (see above).
- The AE used all 40 epochs for every d (no early stop); a longer schedule
  might lower its reconstruction MSE further. **Unknown** whether that changes
  detection.
