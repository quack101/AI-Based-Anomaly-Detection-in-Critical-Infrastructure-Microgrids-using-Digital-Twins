# FDIA v1 detector comparison v3 — physics, AE, and fusion (full)

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix`. Detectors read telemetry `z` only; labels calibrate
thresholds/fusion on val and score on test.

## Part A — attacked-val physics scores (one-time estimator pass)

Frozen estimator over the **1,580 attacked val** timesteps: **100%
converged, 1,454 s**. With the saved clean-val residuals this gives
`bdd_score` on all 12,960 val steps. `val_scores.csv` (timestamp,
attacked, bdd_score, ae_score): bdd non-null 12,960, ae non-null 12,951
(ae is NaN for the first W−1 rows with no history).

Mean scores, attacked vs clean val: **bdd 0.793 vs 0.652** (weak
separation); **ae 6.288 vs 2.906** (strong separation).

## Part B — logistic-regression fusion

Features [bdd_score, ae_score] standardized by clean-val mean/std, label =
attacked. Fitted weights (standardized space): **bdd 0.428, ae 0.389,
bias −2.435**; **val AUC 0.7395**. Both detectors contribute positively and
comparably. `fusion_LR` test score = logistic output.

## Part C — window-mean vs any-step detection (18 test windows)

Window score = trailing mean of the per-timestep score over L steps;
L chosen per detector by val window-AUC (**L=15 for all four**). Window
threshold = 99th pct of non-overlapping clean-val L-block scores (1%
window-FAR). An attack window is detected if the trailing mean crosses
the threshold at any step inside it.

| detector | L | window-mean detected | window-mean rate | realized window-FAR | any-step detected (1% pt) | any-step rate |
|---|---|---|---|---|---|---|
| bdd | 15 | 6/18 | 0.333 | 3.56% | 10/18 | 0.556 |
| ae | 15 | 7/18 | 0.389 | 1.64% | 7/18 | 0.389 |
| fusion_OR | 15 | 7/18 | 0.389 | 1.92% | 8/18 | 0.444 |
| fusion_LR | 15 | 6/18 | 0.333 | 1.23% | 9/18 | 0.500 |

The window-mean rule **lowers physics (bdd) detection** (10→6) because its
residual spikes are brief and average out over 15 steps, and it leaves
bdd's realized window-FAR above target (3.56%). It barely changes the AE
(sustained error). So the smoothing rule favors sustained-error detectors.

## Threshold-free AUC (test)

| detector | per-timestep AUC | window-level AUC |
|---|---|---|
| bdd | 0.6065 | 0.7842 |
| ae | 0.7529 | 0.8060 |
| fusion_OR | 0.7388 | 0.8085 |
| fusion_LR | 0.7321 | **0.8258** |

## Part D — multi-FAR (test), window-mean detection rule

| FAR target | detector | realized clean FAR | attacked-step detection | windows detected (of 18) |
|---|---|---|---|---|
| 0.5% | bdd / ae / OR / LR | 0.50 / 0.46 / 0.47 / 0.27 % | 3.1 / 2.4 / 2.6 / 3.4 % | 1 / 0 / 1 / 1 |
| 1% | bdd / ae / OR / LR | 1.13 / 0.88 / 0.98 / 0.77 % | 4.7 / 4.9 / 4.9 / 5.8 % | 1 / 3 / 2 / 3 |
| 2% | bdd / ae / OR / LR | 2.02 / 1.80 / 1.81 / 1.76 % | 6.8 / 10.7 / 9.9 / 10.5 % | 1 / 5 / 5 / 6 |
| 5% | bdd / ae / OR / LR | 5.24 / 4.56 / 4.71 / 4.63 % | 13.5 / 24.4 / 20.4 / 20.6 % | 3 / 7 / 7 / 8 |

(Per-timestep AUC and window-level AUC per detector are in the table
above; full rows in `validation/detect_v3_multiFAR.csv`.)

## Per-family window detection (window-mean rule, 1% window-FAR calibration)

| detector | A1 (12) | A2 (3) | A3 (3) |
|---|---|---|---|
| bdd | 6/12 | 0/3 | 0/3 |
| ae | 5/12 | 0/3 | 2/3 |
| fusion_OR | 5/12 | 0/3 | 2/3 |
| fusion_LR | 5/12 | 0/3 | 1/3 |

## Verdict (numbers only)

- **Overall:** by the threshold-free **window-level AUC, fusion_LR wins
  (0.8258)**, ahead of fusion_OR (0.8085), AE (0.8060), BDD (0.7842). By
  **per-timestep AUC, AE wins (0.7529)**.
- **Does fusion beat the better single detector?** At the window level,
  **yes, modestly** — fusion_LR's window-level AUC 0.8258 > AE's 0.8060, and
  its window-mean detection is ≥ AE at every FAR target (equal at 0.5%/1%,
  6 vs 5 at 2%, 8 vs 7 at 5%). At the per-timestep level, **no** — fusion_LR
  (0.732) < AE (0.753).
- **Per attack family:**
  - **A1 (random bias): BDD** (6/12 vs 5/12 for the others).
  - **A2 (single-meter bias): none** detect it under the window-mean rule
    (0/3 for all); only the any-step BDD/LR rules catch some A2 (see Part C).
  - **A3 (zone redistribution): AE and fusion_OR** (2/3 vs BDD 0/3,
    fusion_LR 1/3).
- The detector choice interacts with the decision rule: under the
  **any-step** rule BDD leads overall (10/18); under the **window-mean**
  rule AE/fusion lead. The physics detector is spiky (brief residual
  excursions), the AE/fusion detectors are sustained.

## Suspicious / unknown

- Physics (bdd) val/test separation is weak (attacked mean 0.793 vs clean
  0.652); the single physics score barely ranks attacked above clean
  (per-timestep AUC 0.607). Most of its window detection comes from brief
  spikes, which the window-mean rule removes.
- A2 single-meter bias is detected by none of the detectors under the
  window-mean rule, and poorly overall — a small bias on one meter moves
  neither the global residual nor the reconstruction much. Whether a
  per-channel (localized) rule would catch A2 is **unknown** (not tested).
- Realized test FARs sit slightly below the val targets for AE/fusion and
  slightly above for BDD at the window level — the clean val→test score
  drift noted in earlier tasks. Cause **unknown**.
- Fusion's gain over AE is small (window-level AUC +0.020); whether it is
  robust to reseeding/retraining is **unknown** (single seed).
