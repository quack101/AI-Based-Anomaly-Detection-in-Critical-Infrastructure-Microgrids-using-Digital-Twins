# Meter count: 30 vs 45 vs 49, measured with the frozen estimator

> **Result files are not committed.** `reports/measurements/`, the `*all91*` registries and
> `simulation/phase6_corpus_*/` are regenerable output (git-ignored); every number below names the
> script that makes it.

**Verdict: 45 meters matches the 49-meter set A on every measured metric; 30 does not, because the
estimator stops converging.** The estimator is unchanged (truncated-SVD default). This does not show 45
is the minimum: 35 and 40 were not tested, and detection performance (RES.md Phase 12) was not
measured at any count.

## Setup
- Sets: A = frozen 49 (`simulation/representative_meters.json`); H45 and H30 = `HybridPlacement` (v1)
  at `--target 45` / `--target 30` (`simulation/representative_meters_H45.json`, `_H30.json`).
- Same physics for all three: all 91 loads driven from the IDEAL profiles (`--drive all`), 60 days,
  86,400 rows. Registries built by `sensors/build_registry_for_set.py` (rule (ii'), same rules for every set).
- Protocols: `validation/measure_meter_set.py` (60-point stratified scorecard, 90 clean points for BDD,
  null space at 3 points). Paired A-vs-H45 check: `validation/paired_ab.py` on the same 60 points.

## Results
| metric | A (49) | H45 | H30 | source |
|---|---|---|---|---|
| converged, scorecard (of 60) | 60 | 60 | **55** | `measure_meter_set` |
| converged, clean BDD points (of 90) | 90 | 90 | **82** | `measure_meter_set` |
| points at the 40-iteration cap | 0 | 0 | **5** | `*_scorecard_points.csv` |
| mean iterations | 4.5 | 4.5 | **24.3** | `measure_meter_set` |
| mean time per estimate (s) | 5.2 | 1.6 | **30.8** | `measure_meter_set` |
| \|V\| error median (pu) | 0.02149 | 0.02160 | 0.01819 \* | `measure_meter_set` |
| \|V\| error p95 / max (pu) | 0.0417 / 0.0751 | 0.0412 / 0.0750 | 0.0404 / 0.0775 \* | `measure_meter_set` |
| measured dof (median) | 55 | 51 | 36 | `measure_meter_set` |
| null-space dim (unattackable rows) | 113 (442) | 105 (450) | 75 (480) | `unattackable_nullspace_dimension` |
| J99 clean (n=90) \[circular, not for verdict\] | 85,208 | 86,672 | 83,658 | `measure_meter_set` |
| LNR99 clean (n=90) | 177.2 | 172.0 | 169.4 | `measure_meter_set` |

\* H30's error figures are **not comparable**: they are on a different 60 rows (the scorecard stratifies
by each set's own P columns) and include the 5 non-converged points. Treat as unreliable.

Paired A vs H45, identical 60 points (`validation/paired_ab.py`): median 0.021234 vs 0.021261 pu,
p95 0.04084 vs 0.04092, max 0.0689 vs 0.0687; 60/60 converged both; A better on 35 points, H45 on 25;
weakest node median error 0.04353 vs 0.04311. No difference beyond chance.

The H30 paired comparison was not run (about 35 min at 30 s per estimate); the convergence failures
already decide the verdict.

## Caveats
- Every count still fails the accuracy target (about 0.01-0.015 pu). The ~2% floor is the unmodelled
  regulator taps (`reports/step0_bias_control.md`), independent of count. "Within thresholds" here
  means no worse than 49.
- The control is the frozen set A, not a fresh v1 run at 49 (a fresh v1 differs by a few buses,
  `reports/set_A_reproducibility.md`).
- Null space shrinks with count (113 -> 105 -> 75): attack results at a different count are not
  directly comparable to those at 49.
- Timing differs between runs (machine load); use iteration counts, not seconds, for comparison.

## Reproduce
```
python -m placement.select_meters --strategy v1 --target 45 --out simulation/representative_meters_H45.json
python -m sensors.build_registry_for_set noise --set H45all91 --meters simulation/representative_meters_H45.json
python -m validation.generate_phase6_corpus --meters simulation/representative_meters_H45.json \
    --registry config/sensor_registry_multiphase_H45all91_noise.json \
    --out simulation/phase6_corpus_H45all91 --drive all
python -m sensors.build_registry_for_set recal --set H45all91 --corpus simulation/phase6_corpus_H45all91
python -m validation.measure_meter_set --label H45all91 \
    --registry config/sensor_registry_multiphase_H45all91.json --corpus simulation/phase6_corpus_H45all91
```
Same steps for `H30all91` (target 30) and `Aall91` (set A meters). Full run of three sets: about 2.5 h.
