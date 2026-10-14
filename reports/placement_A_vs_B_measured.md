# Placement A vs B, measured with the frozen estimator

> **Result files are not committed.** `reports/measurements/`, `reports/literature_metrics/` and the generated registries (`config/sensor_registry*_{Arebuilt,Aall91,Ball91}*.json`) are regenerable output; rerun the scripts named next to each number to recreate them.

**Verdict: do not merge.** Set B (HybridV2, `simulation/representative_meters_B.json`) gives no
measurable improvement over set A on any main metric. On the same physics and the same 60 test
points, the two sets' accuracy differs by 0.02% (B better on 34 points, A on 26; sign test p = 0.37),
the null-space dimension is identical (113), convergence is identical (60/60), and the weakest node
is the same node with B very slightly worse. HybridV2's advantage in hop-distance coverage is its own
design objective and is **not** evidence of better estimation; on the estimator it does not translate
into anything measurable. v2 stays in the report as a coverage comparison only.

All work is on branch `placement-v2`; `fix-jacobian-bugs` is unchanged and nothing was merged.

## Main table

Every number is read from `reports/measurements/<label>.json` by `validation/ab_table.py`; the
measurements were made by `validation/measure_meter_set.py`, which runs the protocols of
`validation/phase7_scorecard_tsvd_default.py` (60-point stratified scorecard, seed 777) and
`validation/phase8_bdd_baseline.py` (n=90 clean points, null space at 3 points) with the frozen
estimator (truncated-SVD solver), unchanged.

| metric | A_exact | A_rebuilt | A_all91 | B_all91 | source |
|---|---|---|---|---|---|
| \|V\| error median (pu) | 0.02391 | 0.02386 | 0.02149 | 0.02188 | `validation/measure_meter_set.py` |
| \|V\| error p95 (pu) | 0.04080 | 0.04082 | 0.04172 | 0.04168 | `validation/measure_meter_set.py` |
| \|V\| error max (pu) | 0.06262 | 0.06258 | 0.07514 | 0.07497 | `validation/measure_meter_set.py` |
| converged (of 60) | 60 | 60 | 60 | 60 | `validation/measure_meter_set.py` |
| mean iterations | 4.60 | 4.50 | 4.50 | 4.32 | `validation/measure_meter_set.py` |
| weakest node | 83.1 | 83.1 | 83.1 | 83.1 | `validation/measure_meter_set.py` |
| weakest node median error (pu) | 0.04418 | 0.04409 | 0.04301 | 0.04319 | `validation/measure_meter_set.py` |
| median over nodes of median error (pu) | 0.02481 | 0.02448 | 0.02082 | 0.02169 | `validation/measure_meter_set.py` |
| null-space dim (unattackable rows) | 113 (442) | 113 (442) | 113 (442) | 113 (442) | `estimation.bad_data_detection.unattackable_nullspace_dimension` |
| J mean [not for verdict] | 2.817e+04 | 2.818e+04 | 2.771e+04 | 2.793e+04 | `validation/measure_meter_set.py` |
| J mean / 57 [not for verdict] | 494.2 | 494.3 | 486.1 | 489.9 | `validation/measure_meter_set.py` |
| measured dof (median) | 55.0 | 55.0 | 55.0 | 55.0 | `validation/measure_meter_set.py` |
| J99 clean, n=90 [not for verdict] | 60230.29 | 60239.35 | 85208.03 | 85286.42 | `validation/measure_meter_set.py` |
| LNR99 clean, n=90 | 145.98 | 145.98 | 177.22 | 174.06 | `validation/measure_meter_set.py` |
| HEAD_P_A mean signed NR | n/a | +96.6 | +95.4 | +95.6 | `validation/measure_meter_set.py head_p_stats` |
| HEAD_P_B mean signed NR | n/a | +97.9 | +95.8 | +95.4 | `validation/measure_meter_set.py head_p_stats` |
| HEAD_P_C mean signed NR | n/a | +96.1 | +93.8 | +93.7 | `validation/measure_meter_set.py head_p_stats` |
| share of J from HEAD_P rows (mean) | n/a | 95.1% | 94.5% | 94.0% | `validation/measure_meter_set.py head_p_stats` |

Columns:
- **A_exact**: the saved set-A registry (`config/sensor_registry_multiphase.json`) on the rebuilt set-A
  corpus (`simulation/phase6_corpus_Arebuilt`, original physics: only the 49 metered loads vary).
- **A_rebuilt**: the pipeline's registry for set A (`config/sensor_registry_multiphase_Arebuilt.json`,
  rule (ii')) on the same corpus.
- **A_all91 / B_all91**: each set's pipeline registry (rule (ii')) on its own all-91 corpus
  (`simulation/phase6_corpus_{A,B}all91`, `--drive all`). Same true physics in both
  (`test_x_true.npz` arrays identical, checked by `validation/paired_ab.py`).
- J, J/dof and J99 are reported but **do not count** for the verdict: the sigma recalibration uses each
  set's own residuals, so J is circular between sets. HEAD_P residuals were not recorded for A_exact
  (measured before that output existed; not rerun, as instructed).

## Paired check on identical points (decides item 5)

The reference scorecard protocol stratifies rows by the summed telemetry of the set's own `*_P`
columns, so A and B were scored on different rows (19 of 60 shared) although their physics is
identical. `validation/paired_ab.py` scores both on the **same** 60 points, stratified by the true
feeder-head P (meter-independent), same seed and A_flat/B_warm pairing
(`reports/measurements/paired_A_all91_vs_B_all91.json`):

| | A_all91 | B_all91 |
|---|---|---|
| converged | 60/60 | 60/60 |
| \|V\| error median / p95 / max (pu) | 0.021234 / 0.040837 / 0.06889 | 0.021271 / 0.040843 / 0.06873 |
| per-point median error, B − A | — | mean +3.6e-6 pu (+0.02%), median −1.8e-6 pu |
| points where this set is better | 26 | 34 (two-sided sign test p = 0.37) |
| weakest-node median error (pu) | 0.04353 | 0.04379 |

No difference beyond chance on accuracy, and the weakest node is very slightly worse for B.

## Acceptance checklist

| # | item | result | evidence |
|---|---|---|---|
| 1 | no change to any set-A file vs `fix-jacobian-bugs` | **PASS** | `git diff origin/fix-jacobian-bugs` on `simulation/representative_meters.json`, `config/sensor_registry.json`, `config/sensor_registry_multiphase.json`, `validation/phase7_scorecard_tsvd_default_results.csv`, `validation/phase8_bdd_baseline_results.csv` is empty; `simulation/phase6_corpus/` was never written (it does not exist on this clone). |
| 2 | all tests pass with the default (v1, set A) | **PASS** | full suite: 125 passed, 1 skipped, 8 errors, all 8 `FileNotFoundError` for data not on this clone (`simulation/phase6_corpus/*`, `artifacts/20260822T092205Z/.../cells.csv`). Those 8 tests pass (30/30 in their two files) when pointed at byte-identical substitutes without editing any test: the rebuilt corpus (hash-identical `x_true`/telemetry) and a fresh `validation.sweep_4a_tier1` run. |
| 3 | set-A scorecard reproduces the reference numbers | **PASS** | A_exact on the rebuilt corpus: median 0.02391, p95 0.04080, max 0.06262 pu, 60/60, 4.6 iterations, J mean 2.817e4 (494× dof 57), J99 60,230.29, LNR99 145.98, null space 113 / 442 rows: every reference number reproduced exactly **on this independent clone**. |
| 4 | consistency test passes for A and B | **PASS** | `tests/test_meter_set_consistency.py`: 9 passed, 1 skipped (set A's original corpus, absent); `check_meter_set` passes for the rebuilt, A_all91 and B_all91 meter JSON + registry + corpus triples. |
| 5 | B improves accuracy OR null space OR weakest-node precision, no other main metric clearly worse | **FAIL** | accuracy: no difference (paired p = 0.37); null space: 113 both; weakest node: same node, B 0.04379 vs A 0.04353 pu; convergence: 60/60 both. |

**Verdict: do not merge.**

## Other findings

**A_exact vs A_rebuilt (what the hand-made registry choices were worth).** The pipeline registry
(rule (ii')) differs structurally from A_exact in exactly one entry, as predicted: `M040_Q` keeps its
original sigma (0.1) because bus 60 is now excluded, where A_exact recalibrated it (5,422). Sigma
values also differ elsewhere for history reasons: `M042_Q` 5,128 vs 7,184 (A_exact applied the global
recalibration on top of the earlier `0366af8` widening, so it was amplified twice), `M042_P` +1.2%,
and ~100 channels by a median 0.1% (A_exact's row-46 reference estimate was made with the
already-widened registry). Effect on the scorecard: none that matters (median 0.02391 → 0.02386 pu,
J/57 494.2 → 494.3, J99 60,230 → 60,239).

**Rule (ii) demonstration: what the M040_P judgement call was worth.** With the regulator-only
exclusion, the formula gives `M040_P` sigma 230,907,154 (A_exact: 0.2). On 10 scorecard points
(`reports/measurements/A_rule_ii_demo10.json`, `validation/measure_meter_set.py --max-points 10`)
the estimator converged on **1 of 10** points, averaged **38.6 iterations** (cap 40; A_exact 4.6) and
**177 s per estimate** (A_exact 6.4 s under the same load). The hand-picked exclusion was protecting
the estimator from a near-singular channel.

**The exclusion rule (ii') was chosen after observing that blow-up.** Rule (ii') = regulator buses
OR switch buses (`observability.zero_injection._regulator_buses() | _switch_buses()`). It was adopted
**after** seeing the formula explode at bus 60, which sits on switch `Sw4` (60–160, r1=1e-3, x1=0,
Length=0.001), the same near-zero-impedance near-singularity as a regulator. Justification: the same
grouping `zero_injection.py` already uses for its zero-injection exclusions. It was not chosen in
advance. Set B has a meter on bus 60 too (`M028`), which rule (ii') excludes for the same reason.
The "unexplained extreme outlier" `M040_P` of commit `81ff01a` is therefore a switch-bus effect.

**All-91 physics vs original physics (A_all91 vs A_exact).** When the 42 unmetered loads vary as
well, accuracy shifts: the median error improves (0.02391 → 0.02149 pu) but the tail gets worse
(p95 0.04080 → 0.04172, max 0.06262 → 0.07514), and the clean-data bad-data thresholds rise (J99
60,230 → 85,208, +41%; LNR99 146 → 177). J mean barely moves (494 → 486 × 57). The reference numbers
were measured on a feeder where 42 loads never change, which understates the error tail and the
detection thresholds an attack test should use.

**HEAD_P dominates J.** In every set measured, the three `HEAD_P` rows have a mean signed normalised
residual of about +94 to +98 sigma and carry about 94–95% of J. This confirms what was seen on the
main branch: J (and J/dof) mostly measures a feeder-head P bias, not the meters' fit.

**The weakest node is the same for every set.** Bus 83 (83.1, then 82.1/83.2/81.1/80.1) is the
weakest-observed area for A and B alike, so it is a property of the feeder (the bus-83 capacitor
region), not of set A's placement.

## How to reproduce
1. `python -m validation.rebuild_corpus_candidates` (or `generate_phase6_corpus --meters ... --registry
   config/sensor_registry_multiphase_Arebuilt_noise.json --out ... --drive metered`): set-A corpus.
2. `python -m sensors.build_registry_for_set noise --set <S> --meters <json>`, then
   `python -m validation.generate_phase6_corpus --meters <json> --registry
   config/sensor_registry_multiphase_<S>_noise.json --out simulation/phase6_corpus_<S> --drive all`,
   then `python -m sensors.build_registry_for_set recal --set <S> --corpus simulation/phase6_corpus_<S>`.
3. `python -m validation.measure_meter_set --label <L> --registry <recal registry> --corpus <corpus>`.
4. `python -m validation.paired_ab ...` and `python -m validation.ab_table`.

Corpora and `dataset/node_loads_91/` are not committed (size); the registries and measurement
summaries are.
