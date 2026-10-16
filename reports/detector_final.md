# FDIA v1 — final detector comparison

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix` (not run). v3 AE (d=32) per-channel errors recomputed
from saved weights; Mahalanobis fit (mean + Ledoit-Wolf cov, λ=7e-4) on
clean train; thresholds/fusion weights on clean val; test scored once.

## Step A — window-smoothing the Mahalanobis score (negative result)

| variant | realized FAR | window det. | window AUC | median delay |
|---|---|---|---|---|
| Mahalanobis **point** | 0.88% | **11/18** | 0.770 | 8 |
| Mahalanobis trailing-mean (L=15) | 1.75% | 7/18 | 0.746 | 14 |
| Mahalanobis CUSUM (k=0.5) | 0.00% | 7/18 | — | 15 |

Window-smoothing **does not help** — both the trailing mean (L picked on
val = 15) and CUSUM detect fewer windows (7/18 vs 11/18) and add delay.
The point Mahalanobis is already a strong per-timestep score (its attack
signal is a sharp spike); averaging dilutes the spike, and the smoothed
clean-score tail transfers worse val→test (trailing-mean realized 1.75%
FAR, above the 1% target). So the point score is kept.

## Step B — refit fusion (BDD + Mahalanobis)

Logistic regression on val, features [bdd_score, mahalanobis_score]
standardized by clean-val stats; val AUC 0.769; weights **bdd 0.366,
maha 0.395** (both contribute). Threshold at 1% clean val.
Test: **12/18 windows at 0.73% FAR**, per-timestep AUC 0.873, window AUC
0.811, family **A1 9/12, A2 1/3, A3 2/3**.

## Step C — final table (all at ~1% FAR)

| detector | realized FAR | attacked-step det. | window det. | A1 | A2 | A3 | per-ts AUC | window AUC | localization |
|---|---|---|---|---|---|---|---|---|---|
| BDD any-step | 1.13% | — | 10/18 | 8/12 | 1/3 | 1/3 | 0.607 | 0.784 | — |
| AE Mahalanobis (point) | 0.88% | 44.9% | 11/18 | 9/12 | 0/3 | 2/3 | 0.874 | 0.770 | 72.7% |
| AE Mahalanobis (trailing-mean L=15) | 1.75% | 41.4% | 7/18 | 6/12 | 0/3 | 1/3 | 0.867 | 0.746 | 71.4% |
| **fusion BDD+Mahalanobis** | **0.73%** | 43.4% | **12/18** | 9/12 | 1/3 | 2/3 | 0.873 | **0.811** | — |

**Single best detector + rule for the paper: fusion of the physics BDD
score and the AE Mahalanobis (point) score, logistic, at ~1% FAR.**
It gives **12/18 windows at 0.73% realized FAR**, per-timestep AUC 0.873,
the best window AUC (0.811), and it is the only learned-plus-physics rule
that catches an A2 window (1/3) while keeping A3 (2/3) and A1 (9/12). The
point AE Mahalanobis alone is a close, simpler second (11/18, 0.88% FAR).

## Deltas and suspicious / unknown

- Fusion vs point Mahalanobis: **+1 window at a lower FAR** (12/18 @ 0.73%
  vs 11/18 @ 0.88%), window AUC +0.04, and it **recovers A2** (1/3 vs 0/3)
  — the physics score contributes the single-meter-bias signal the AE
  misses.
- Fusion vs BDD any-step: **+2 windows at a lower FAR**, per-timestep AUC
  +0.27.
- **Window-smoothing the Mahalanobis is a net negative** — a reminder that
  smoothing helps spiky physics residuals (v3/v5) but hurts an already
  per-timestep-strong learned score.
- **A2 is still only 1/3** even for the best fusion; single-meter bias
  remains the hardest family. A dedicated single-channel test was not
  built — **unknown** whether it would close the gap.
- The trailing-mean realized 1.75% FAR (above target): the smoothed
  clean-score tail does not transfer val→test as well as the point score;
  cause **unknown** (point scores transfer fine, per v5).
- All numbers are single-seed / single val split; reseed robustness
  **unknown**.
