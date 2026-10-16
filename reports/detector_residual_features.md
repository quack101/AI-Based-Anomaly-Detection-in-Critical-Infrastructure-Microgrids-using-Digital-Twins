# FDIA v1 — learned detector on physics-residual features

Simulated IEEE-123 feeder, `dataset/fdia_v1`. The frozen estimator
(`estimator-v5-hfix`) was **run** (not changed) to log per-channel WLS
residuals; the question is whether a learned detector on those residuals
beats the telemetry-domain detectors. numpy only.

## Step A — residual logging (frozen estimator)

Per step, residual vector r_c = (z_c − h_c(x̂)) / σ_c over all **361
weighted rows** (165 meter + 112 REGZI + 84 PSEUDO). No ground truth
used. Strides: train every 4th clean, val every 4th clean + every
attacked, test every step. 50-solve timing 0.809 s → projected 7.3 h;
train 3.4 h < 8 h so stride 4 kept.

| split | rows | converged | time |
|---|---|---|---|
| train | 15,120 | 100% | 12,925 s |
| val | 4,433 (1,580 attacked) | 100% | 3,743 s |
| test | 12,960 (1,580 attacked) | 100% | 11,048 s |

Total 27,728 s ≈ 7.7 h. Saved to `residuals_{split}.npz` (float32)
instead of a ~130 MB CSV, purely for size.

**Deviation (noted):** train is logged at stride 4, so no 10
*consecutive* steps exist for the telemetry AE's W=10 temporal window
while test is every-step. A W=10 temporal AE cannot be trained
consistently across splits, so Step B uses a **W=1 reconstruction AE**
(same GRU-AE cell, d=32) on the standardized residual vector —
stride-invariant.

## Step B — residual reconstruction AE

W=1 GRU-AE, d=32, 146,921 params; trained on clean train (stride 4),
clean-val MSE 0.1379 at the 120-epoch cap (no early stop); gradient
check 1.2e-7.

## Step C/D — Mahalanobis + fusion, vs the telemetry detectors (≈1% FAR)

| detector | realized FAR | window det. | A1 | A2 | A3 | per-ts AUC | window AUC | localization |
|---|---|---|---|---|---|---|---|---|
| telemetry Mahalanobis (point) | 0.88% | 11/18 | 9/12 | 0/3 | 2/3 | 0.874 | 0.770 | 72.7% |
| telemetry fusion BDD+Maha | 0.73% | **12/18** | 9/12 | 1/3 | 2/3 | 0.873 | 0.811 | — |
| residual-AE Mahalanobis | 1.11% | 8/18 | 7/12 | 0/3 | 1/3 | 0.651 | 0.575 | 12.5% |
| fusion BDD+residual-Maha | 1.22% | 8/18 | 7/12 | 0/3 | 1/3 | 0.687 | 0.604 | — |

Residual-AE Ledoit-Wolf shrinkage λ=0.0176, cond(Σ)=1.5e3.

## Verdict

- **Residual features do NOT beat the telemetry detector.** The
  residual-AE Mahalanobis is markedly worse: 8/18 vs 11/18, per-timestep
  AUC **0.651 vs 0.874**, window AUC 0.575 vs 0.770, localization **12.5%
  vs 72.7%**. Fusing it with BDD gives 8/18 (vs the telemetry fusion's
  12/18). The telemetry-domain **fusion BDD + AE-Mahalanobis (12/18 @
  0.73%)** remains the best detector in the study.
- **A2 does not improve** (0/3 for both residual variants) — single-meter
  bias is still the hardest family.

## Why (interpretation) and unknowns

- The WLS residual is partly what the estimator *minimizes*: it fits x̂
  to the delivered z, absorbing a share of any consistent bias into the
  state, so the residual carries a weaker attack signal than the raw
  telemetry-reconstruction error. (A4/A5 are explicitly null-space /
  residual-preserving; A1/A2/A3 are not, but the fit still absorbs part.)
- The residual is dominated by the pseudo/REGZI rows' large modeling
  errors (the standing J/dof≈3 excess), which swamp the attacked meter
  channel — consistent with the **12.5% localization** (the top
  Mahalanobis-contributing residual channel is rarely the attacked
  target; it is usually a pseudo/REGZI row).
- The residual-AE hit the 120-epoch cap (clean-val MSE still falling);
  whether more epochs would narrow the gap is **unknown**, but the gap to
  telemetry (ΔAUC 0.22) is large.
- Single-seed; reseed robustness **unknown**.

Not merged; branch `exp-residual-features`.
