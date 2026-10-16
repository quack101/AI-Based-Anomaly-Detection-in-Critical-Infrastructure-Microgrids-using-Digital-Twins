# FDIA v1 detector comparison v5 — drift diagnosis + adaptive threshold

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix` (not run here). Detection uses the **v3 AE (d=32)**;
the v4 d=48 AE is retired for detection (it reconstructs anomalies
better, lowering detection AUC — see v4). The v3 AE's per-channel errors
were recomputed from its saved weights (an AE forward pass, numpy only,
not an estimator run). The adaptive threshold is label-free at test time;
its one constant k is set on val.

## Part 1 — drift diagnosis

Clean-timestep score distribution (mean / std / p90 / p99):

| detector | split | mean | std | p90 | p99 |
|---|---|---|---|---|---|
| bdd | val clean | 0.6523 | 0.1399 | 0.8286 | 1.0389 |
| bdd | test clean | 0.6566 | 0.1384 | 0.8342 | 1.0493 |
| ae | train clean | 2.9471 | 0.9671 | 4.0017 | 6.6468 |
| ae | val clean | 2.9062 | 0.8748 | 3.9131 | 6.2818 |
| ae | test clean | 2.8806 | 0.8439 | 3.8003 | 6.0583 |

Clean test score by day-of-series (9 test days): bdd mean 0.632–0.675,
ae mean 2.80–3.02 — **flat, no monotonic trend**; by hour-of-day: mild
(`detect_v5_drift_by_{day,hour}.csv`).

**Diagnosis (one line): the aggregate bdd and ae scores do NOT drift
val→test — mean, std and tails are nearly identical (ae test is if
anything slightly lower).** The earlier "drift" was confined to the
**per-channel** detector: calibrating 168 per-channel thresholds at a
per-channel tail and OR-ing them inflates the realized FAR (1% val →
5.47% test), a per-channel tail-calibration failure under the 168-way
union, not a global location/scale shift.

## Part 2 — adaptive rolling threshold (label-free) vs static

Flag s_t > median + k·MAD over the trailing W_adapt scores (robust to
the ~12% attacked points); k set on val for 1% clean-val FAR, applied to
test with the test-local rolling baseline. W_adapt picked on val
(W=720 for all three single scores).

| detector | static FAR / win | adaptive FAR / win | adaptive by family (A1/A2/A3) |
|---|---|---|---|
| bdd | 1.13% / 10/18 | 1.21% / 9/18 | 8/12, 1/3, 0/3 |
| ae | 0.88% / 7/18 | 0.95% / 6/18 | 5/12, 0/3, 1/3 |
| fusion_LR | 0.77% / 9/18 | 0.76% / 8/18 | 7/12, 1/3, 0/3 |
| **per-channel AE** | **5.47% / 14/18** (v4 static) | **0.84% / 6/18** | 5/12, 0/3, 1/3 |

The single-score detectors were already at ~1% test FAR statically (no
drift), so the adaptive rule changes little and slightly lowers
detection. The adaptive rule **does fix the per-channel FAR** (5.47% →
0.84%, back to target) — but with the FAR correctly controlled, the
per-channel detector catches only **6/18**, not 14/18. Its v4 14/18 was
an uncontrolled-FAR artifact. Per-channel localization at the controlled
point: 3/6 = 50%.

## Part 3 — final comparison (target 1% FAR)

| detector / rule | realized test FAR | attacked-step det. | window det. | A1 | A2 | A3 | per-ts AUC |
|---|---|---|---|---|---|---|---|
| bdd static | 1.13% | — | **10/18** | 8/12 | 1/3 | 1/3 | 0.607 |
| bdd adaptive | 1.21% | 3.96% | 9/18 | 8/12 | 1/3 | 0/3 | 0.607 |
| ae static | 0.88% | — | 7/18 | 5/12 | 0/3 | 2/3 | 0.753 |
| ae adaptive | 0.95% | 2.81% | 6/18 | 5/12 | 0/3 | 1/3 | 0.753 |
| fusion_LR static | 0.77% | — | 9/18 | 7/12 | 1/3 | 1/3 | 0.732 |
| fusion_LR adaptive | 0.76% | 4.15% | 8/18 | 7/12 | 1/3 | 0/3 | 0.732 |
| per-channel AE static (v4) | 5.47% | — | 14/18 | 11/12 | 1/3 | 2/3 | — |
| per-channel AE adaptive | 0.84% | — | 6/18 | 5/12 | 0/3 | 1/3 | — |

**Single best detector + threshold for the paper:** **physics (BDD),
single score, any-step, static 99th-pct threshold** — **10/18 windows at
a realized 1.13% FAR**, the best window detection among all correctly-
calibrated (~1% FAR) options, and it needs no learned model. Honest
realized FAR: **1.13%**.

**Did the adaptive threshold fix the drift?** For the per-channel
detector, **yes** — realized test FAR dropped 5.47% → 0.84% (≈ target),
but detection fell to 6/18 once the FAR was honest, so the fix removed an
artifact rather than adding capability. For the aggregate bdd/ae/fusion
scores there was **no drift to fix** (static FAR already ~1%), and the
adaptive rule slightly reduced window detection.

## Deltas vs v3 and suspicious / unknown

- **No aggregate val→test drift** (bdd/ae clean distributions match);
  this corrects the standing "val→test drift" caveat from v1–v4 — it was
  real only for the per-channel detector's 168-way thresholding.
- **The v4 per-channel 14/18 was a FAR artifact.** At a correctly
  controlled 1% FAR it is 6/18, in line with the single scores.
- **A2 single-meter bias stays near-undetected** by every correctly-
  calibrated rule (0–1 of 3). Still the clear gap; a dedicated
  single-channel test was not built.
- Best-at-1%-FAR window detection across the whole study is **10/18
  (BDD any-step)**; no learned detector beats it at a controlled FAR.
  Whether a larger/temporally-richer learned model would is **unknown**
  (out of scope: no torch, frozen estimator).
- The adaptive k and W_adapt are single-seed, val-tuned; robustness to
  reseeding is **unknown**.
