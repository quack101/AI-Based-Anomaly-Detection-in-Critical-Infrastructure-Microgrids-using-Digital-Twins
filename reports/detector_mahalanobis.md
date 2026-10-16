# FDIA v1 — Mahalanobis aggregation of the AE per-channel errors

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix` (not run). Detection uses the **v3 AE (d=32)**; its
168 per-channel normalized reconstruction errors are recomputed from the
saved weights (numpy AE forward, not an estimator run). mean + covariance
fit on **clean train**; thresholds on **clean val**; test scored once.

## Step 1 — covariance fit (clean train, n=60,471)

Ledoit-Wolf shrinkage toward μ·I: **λ (ρ) = 0.0007** (tiny — n ≫ p=168,
the sample covariance is already well-posed). Condition number:
**sample 6.46e3 → shrunk 5.74e3**.

## Step 2–3 — scores and test results (1% val operating point)

- `s_full = (e−μ)ᵀ Σ⁻¹ (e−μ)` (whitened; uses inter-channel correlations)
- `s_diag = Σ_c (e−μ)²_c / var_c` (ignores correlations)

| detector | realized test FAR | attacked-step det. | window det. | per-ts AUC | window AUC | localization |
|---|---|---|---|---|---|---|
| BDD any-step (prior best) | 1.13% | — | 10/18 | 0.607 | 0.784 | — |
| AE max-agg (v3) | 0.88% | ~5% | 7/18 | 0.753 | 0.806 | — |
| **AE Mahalanobis-full** | **0.88%** | **44.9%** | **11/18** | **0.874** | 0.770 | **8/11 = 72.7%** |
| AE Mahalanobis-diag | 0.89% | 2.7% | 7/18 | 0.727 | 0.649 | 5/7 = 71.4% |

By family (window detection): Mahalanobis-full **A1 9/12, A2 0/3, A3 2/3**
(max-agg was A1 5/12, A3 2/3; BDD A1 8/12).

## Step 4 — verdict

- **Mahalanobis-full beats BDD at a controlled FAR:** 11/18 windows at
  **0.88%** FAR vs BDD's 10/18 at 1.13%, and per-timestep AUC **0.874 vs
  0.607**. Attacked-step detection jumps from ~5% (max-agg / BDD) to
  **44.9%** — the whitened score separates attacked from clean per
  timestep far better. It also localizes (72.7%).
- **The gain is from inter-channel correlations.** The diagonal variant
  (no correlations) is no better than max-agg (7/18, AUC 0.727) — so
  whitening with the full covariance, not just per-channel scaling, is
  what helps.
- **A2 does NOT improve** — single-meter bias stays 0/3 under both
  Mahalanobis variants (as under every other rule). A small bias on one
  metered channel remains below the aggregate's reach.
- Unlike the per-channel union detector (v4/v5, whose FAR drifted 1%→5%),
  the Mahalanobis score is a single aggregate and **holds its FAR on test**
  (0.88% ≈ target), consistent with the v5 finding that aggregate AE/BDD
  scores do not drift.

**This is the single best detector in the study:** AE Mahalanobis-full,
11/18 windows at 0.88% FAR, per-timestep AUC 0.874, 72.7% localization —
clear of BDD any-step (10/18 @ 1.13%) and the AE max-aggregation (7/18).

## Deltas and suspicious / unknown

- vs AE max-agg (v3): **+4 windows, AUC +0.12, attacked-step 5%→45%**, at
  the same 0.88% FAR.
- vs BDD any-step: **+1 window at a lower FAR, AUC +0.27**.
- window-level AUC (0.770) is slightly below max-agg's (0.806) even though
  per-timestep AUC is much higher — the whitened score separates
  per-timestep strongly but its max over clean 60-blocks also spikes;
  **unknown** whether a window-smoothed Mahalanobis would raise window AUC.
- A2 undetected everywhere; a dedicated single-channel test remains the
  obvious gap. Not built here.
- cond(Σ)≈5.7e3 is moderate; shrinkage barely active (λ=7e-4). Whether a
  heavier-shrinkage or robust covariance would change test numbers is
  **unknown** (not swept — n≫p made it unnecessary).
