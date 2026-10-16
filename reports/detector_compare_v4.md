# FDIA v1 detector comparison v4 — converged AE, CUSUM, per-channel, diurnal

Simulated IEEE-123 feeder, `dataset/fdia_v1`. Estimator frozen at
`estimator-v5-hfix` (run once over attacked val in the prior task; not
run here). Detectors read telemetry `z` only; thresholds/tuning on val,
test scored once per variant. numpy only.

## Part A — AE retrained to convergence

No epoch cap, early-stop patience 10 (max 200). Sweep d ∈ {16,32,48} by
clean-val reconstruction MSE:

| d | clean-val MSE | best epoch | params |
|---|---|---|---|
| 16 | 0.26799 | 104 | 80,728 |
| 32 | 0.17467 | 123 | 84,968 |
| 48 | **0.11623** | 129 | 89,208 |

Chosen **d=48** (lowest MSE; v3 used d=32 @ 40 epochs, MSE 0.177).
Total 3,422 s. Per-channel errors saved for Part C.

**Negative result:** the better-reconstructing AE is a slightly *worse*
detector. Per-timestep AUC **0.743 (v4) vs 0.753 (v3)**; AE window-level
AUC (mean rule) **0.687 (v4) vs 0.806 (v3)**. A larger, fully-converged
bottleneck reconstructs anomalies better too, shrinking the anomaly
residual. Lower reconstruction loss ≠ better detection here.

## Part B — CUSUM (segment-reset) vs mean-window (both on v4 scores)

CUSUM g_t = max(0, g_{t-1} + s_std − k), accumulated within each segment
(attack window, or 60-step clean block); k picked by val window-AUC,
h at 1% clean-val window-FAR.

| detector | CUSUM k | CUSUM windows (FAR, delay) | mean-window windows (FAR) | CUSUM winAUC | mean winAUC |
|---|---|---|---|---|---|
| bdd | 0.5 | 4/18 (1.12%, 45) | 3/18 (0.43%) | 0.668 | 0.684 |
| ae | 1.0 | 7/18 (0.00%, 45) | 7/18 (0.61%) | 0.696 | 0.687 |
| fusion_LR | 0.25 | 7/18 (1.12%, 41) | 5/18 (0.47%) | 0.741 | 0.731 |

CUSUM did **not** beat the simple rule at window level (winAUC 0.67–0.74,
below v3's mean-rule 0.78–0.83), and its detection delay is long (41–45
steps) because it must accumulate. Reset after an alarm does not change
window-level detection or first-alarm delay (first crossing is identical),
so it is omitted from these metrics.

## Part C — per-channel localized AE detection

Per-channel CUSUM (k=0.5) on the normalized reconstruction errors;
per-channel thresholds by Bonferroni (α = 1%/168 = 5.95e-5) on clean val;
flag a timestep if any channel crosses. A tuned-global-level variant
(calibrated to 1% union per-timestep FAR on clean val) is reported too.

| variant | windows | per-timestep FAR (test) | block-FAR | localization | by family |
|---|---|---|---|---|---|
| Bonferroni | **14/18** | 5.16% | 25.8% | 8/14 = 57.1% | A1 11/12, A2 1/3, A3 2/3 |
| tuned→1% (val) | 14/18 | 5.47% | 28.7% | 8/14 = 57.1% | A1 11/12, A2 1/3, A3 2/3 |

Per-channel detection is the **highest window coverage of any rule**
(14/18, A1 11/12 vs ≤6/12 for the single-score rules) and gives
localization (top-CUSUM channel is an attacked target 57% of the time).
**But its FAR is uncontrolled on test:** both calibrations target ~1%
per-timestep FAR on clean val yet realize **~5.2–5.5%** on clean test
(and block-FAR ~26–29%). The clean val→test drift seen throughout defeats
the 1% calibration, so the 14/18 is not a true 1% operating point.

## Part D — diurnal (per-hour) thresholds

| detector | diurnal FAR / window | single FAR / window |
|---|---|---|
| bdd | 1.50% / 72.2% | 1.13% / 55.6% |
| ae | 2.06% / 61.1% | 0.96% / 44.4% |
| fusion_LR | 1.84% / 61.1% | 0.71% / 50.0% |

Per-hour thresholds raise window detection (bdd 55.6→72.2%, ae
44.4→61.1%, fusion 50→61.1%) **but push realized FAR above the 1%
target** (1.5–2.1%), because the per-hour clean-val quantiles (fewer
samples per bucket) generalize worse to test. Net: more detections at a
looser, less-controlled FAR.

## Part E — comparison and verdict

Threshold-free AUC, v3 → v4:

| detector | per-timestep AUC v3 → v4 | window AUC v3 (mean) → v4 (mean / CUSUM) |
|---|---|---|
| bdd | 0.607 → 0.607 | 0.784 → 0.684 / 0.668 |
| ae | 0.753 → 0.743 | 0.806 → 0.687 / 0.696 |
| fusion_LR | 0.732 → 0.728 | 0.826 → 0.731 / 0.741 |

Per-family window detection (A1/A2/A3), old (v3 mean rule) → improved:

| detector / rule | A1 | A2 | A3 |
|---|---|---|---|
| bdd v3 mean | 6/12 | 0/3 | 0/3 |
| bdd v4 CUSUM | 4/12 | 0/3 | 0/3 |
| ae v3 mean | 5/12 | 0/3 | 2/3 |
| ae v4 CUSUM | 5/12 | 0/3 | 2/3 |
| **AE per-channel (v4)** | **11/12** | 1/3 | 2/3 |

**Where it improved:** only the **per-channel AE detector** genuinely
raises detection (14/18 overall, A1 11/12) and adds localization (57%) —
but at ~5% test FAR, not 1%. Diurnal thresholds raise window detection
too, but at FAR above target.

**Where it did not:** retraining the AE to convergence *lowered* its
detection AUC (0.753→0.743 per-timestep; 0.806→0.687 window). CUSUM did
not beat the simple any-step/mean rule and added 41–45-step delays. At a
**controlled 1% test FAR, nothing in v4 beats v3** — the single-score
AUCs are flat or down.

**Single best detector+rule for the paper:**
- If a **controlled ~1% FAR** is required: **physics (BDD) single score,
  any-step rule** — 55.6% window detection at 1.13% FAR, the most
  reliable FAR control, and A1 is its strength.
- If **maximum window coverage + localization** is the goal and a higher
  FAR is acceptable: the **per-channel AE CUSUM** — 14/18 windows, A1
  11/12, 57% localization, but ~5% test FAR (needs FAR recalibration on a
  held-out clean set closer to test).
- A2 single-meter bias stays essentially undetected by every rule
  (≤1/3).

## Deltas vs v3 and suspicious / unknown

- **AE-to-convergence is a net negative** for detection (AUC down); the
  under-trained v3 AE separated anomalies better. Unknown whether an
  intermediate d / early epoch is optimal (not swept finely).
- **Per-channel FAR does not transfer val→test** (1% target → ~5.5%
  realized). The clean test residuals run hotter than clean val across
  all detectors; root cause **unknown** (seen since Part C of the v1
  task; possibly a val/test seasonal shift in the corpus).
- CUSUM window-AUC < mean-rule window-AUC here — CUSUM's accumulation
  helps sustained signals but the attacks' onset transients dominate;
  **unknown** whether a two-sided or adaptive-k CUSUM would differ.
- A2 undetected everywhere; a dedicated single-channel test is the
  obvious gap. Not attempted here.
