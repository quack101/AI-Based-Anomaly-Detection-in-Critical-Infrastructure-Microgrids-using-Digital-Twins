# FDIA v1 — detector hardening: fair W=1 comparison + reseed robustness

Simulated IEEE-123 feeder, `dataset/fdia_v1`. numpy only; no estimator
runs (reuse saved telemetry z, `residuals_*.npz`, bdd scores). Fit/
threshold on train+val only; test scored once per variant.

## Step A — fair W=1 comparison (telemetry vs residual)

Both a telemetry and a residual reconstruction AE (same GRU-AE cell,
d=32, **W=1**) trained on the **identical strided-clean train indices**
the residual-AE used; Mahalanobis-scored; ~1% FAR.

| feature domain | window det. | A1 | A2 | A3 | per-ts AUC | window AUC | realized FAR | localization |
|---|---|---|---|---|---|---|---|---|
| telemetry (W=1) | 9/18 | 7/12 | 0/3 | 2/3 | **0.865** | 0.755 | 0.85% | **77.8%** |
| residual (W=1) | 8/18 | 7/12 | 0/3 | 1/3 | 0.651 | 0.575 | 1.11% | 12.5% |

**Telemetry beats residual even at matched W=1** — per-timestep AUC
0.865 vs 0.651, window AUC 0.755 vs 0.575, localization 77.8% vs 12.5%.
The residual detector's inferiority is **intrinsic** (the WLS residual
is partly what the estimator minimizes and is dominated by pseudo/REGZI
modeling error), not an artifact of the earlier W-mismatch. (Telemetry
W=1 at 9/18 / AUC 0.865 is already close to the W=10 telemetry
Mahalanobis at 11/18 / AUC 0.874 — the temporal window adds only a
little.)

## Step B — reseed robustness (best detector, telemetry W=10 d=32, 5 seeds)

Per seed: retrain the AE, recompute the Mahalanobis-point score and the
BDD+Mahalanobis fusion; threshold at 1% clean val; score test.

| detector | per-ts AUC | window AUC | window det. (of 18) | realized FAR | A2 count |
|---|---|---|---|---|---|
| AE Mahalanobis (point) | 0.8714 ± 0.0011 | 0.7677 ± 0.0074 | 10.0 ± 0.6 | 0.85 ± 0.06 % | 0.0 ± 0.0 |
| fusion BDD + Mahalanobis | 0.8700 ± 0.0013 | 0.8005 ± 0.0053 | 10.6 ± 1.2 | 0.77 ± 0.06 % | 0.4 ± 0.5 |

Per-seed window detection: Maha {10,10,11,10,9}; fusion {12,10,10,12,9}.

## Verdict and honesty check

- **Telemetry features beat residual features**, confirmed at matched
  W=1 — AUC gap ~0.21, localization gap ~65 points.
- **A2 does not improve**: the Mahalanobis never catches an A2 window
  (0/18 every seed); the fusion catches one only in 2 of 5 seeds
  (A2 count 0.4 ± 0.5) — **A2 detection is not robust**.
- The best detector is **robust in per-timestep AUC (±0.001) and FAR
  (±0.06%)**, and moderately robust in window count (fusion 10.6 ± 1.2,
  range 9–12).
- **Honesty note:** the headline figures reported earlier (point
  Mahalanobis 11/18, fusion 12/18, fusion A2 1/3, seed 1234) sit at the
  **favorable end** of the seed distribution. The seed-mean is ~10/18
  (point) and ~10.6/18 (fusion), and the A2 catch is seed-dependent. The
  fusion's window AUC advantage over the point score (0.800 vs 0.768) is
  robust across seeds.

## Suspicious / unknown

- Fusion A2 is a coin-flip across seeds; whether a different feature or a
  dedicated single-channel test gives robust A2 detection is **unknown**.
- Only the AE seed is varied; the val split, the corpus, and the single
  logistic fit are fixed — robustness to those is **unknown**.
- Residuals/telemetry comparison uses one strided-train draw; **unknown**
  if a different stride changes the W=1 gap (large gap makes it unlikely).
