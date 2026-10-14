# Step 0: what limits estimation accuracy (bias-control experiment)

> **Result files are not committed.** `reports/measurements/`, `reports/literature_metrics/` and the generated registries (`config/sensor_registry*_{Arebuilt,Aall91,Ball91}*.json`) are regenerable output; rerun the scripts named next to each number to recreate them.

**Answer: regulator tap ratios that the estimator does not model.** The estimator's network model
treats every voltage regulator as ratio 1.000, while the simulated regulators run at 1.019–1.050 (tap
steps of 0.625%). Every node downstream of a regulator is estimated too low by about that boost. This
sets the ~2% |V| error floor for every placement. Neither the HEAD_P channels nor the pseudo-
measurements of unmetered loads limit accuracy, and with the floor in place set A and set B stay
indistinguishable. **Improving the placement cannot improve accuracy until the estimator models the
regulator taps.** The estimator is frozen on this branch, so nothing here changes it: these are
findings for the estimator owners.

All runs use the all-91 corpora and rule-(ii') registries for A and B, the same 60 points (stratified
by true feeder-head P, `validation/paired_ab.py` `neutral_picks`), source
`validation/bias_control.py`, results `reports/measurements/bias_control.json`.

## 1. HEAD_P: a broken measurement function that inflates J but does not move the estimate
At the TRUE state, the estimator's model gives HEAD_P = 0.00 on all three phases (recorded: 823.5 /
562.5 / 688.9 kW at test row 46), while HEAD_Q matches (211.1 recorded vs 201.1 modelled).
Cause, `observability/jacobian.py:259` `build_branch_or_head_row`: head P is computed through the
`Vsource` element as Vt·Vk·(G·cos θ + B·sin θ) with G ≈ 0 (the source is almost a pure reactance) and
θ = angle(bus 150) − angle(source EMF). The source-bus angle is pinned to the source-EMF angle, so θ = 0
and P ≡ 0, with an all-zero Jacobian row. The recorded ~2 MW can never be matched: this is the
"+95 sigma HEAD_P, ~95% of J" seen on every set.

| variant (same 60 points) | A median | B median | A J mean | B J mean |
|---|---|---|---|---|
| V0 as deployed | 0.02123 | 0.02127 | 25,600 | 25,700 |
| V1 HEAD_P_A/B/C disabled | 0.02123 | 0.02127 | **1,140** | **1,250** |

Disabling HEAD_P cuts J 22× and changes accuracy not at all (zero Jacobian row → no influence on x̂).
J/dof and the clean-data J thresholds measured so far are therefore dominated by this one defect.

## 2. Pseudo-measurements are not the limit either
The pseudo values over-estimate the 42 unmetered loads by 126 kW on average (true 1,211 kW vs pseudo
1,337 kW; error std 151 kW; combined pseudo sigma 79 kW). Replacing every pseudo value with the load's
TRUE P,Q at that timestamp (oracle, same sigma; V2) leaves accuracy unchanged:

| variant | A median / p95 / max | B median / p95 / max | B−A per point | B better / A better | sign test |
|---|---|---|---|---|---|
| V0 as deployed | 0.02123 / 0.04084 / 0.0689 | 0.02127 / 0.04084 / 0.0687 | +0.017% | 34 / 26 | p = 0.37 |
| V1 HEAD_P off | 0.02123 / 0.04084 / 0.0689 | 0.02127 / 0.04084 / 0.0687 | +0.017% | 34 / 26 | p = 0.37 |
| V2 HEAD_P off + oracle pseudo | 0.02123 / 0.04085 / 0.0690 | 0.02126 / 0.04082 / 0.0687 | −0.042% | 38 / 22 | p = 0.05 |

All 60/60 converged in every cell. Even with perfect pseudo-measurements, the two placements differ by
0.04%.

## 3. The actual floor: regulator taps
The error is systematic, not noise: averaged over test rows 46, 300 and 600 (set A, all-91), 78% of
phase-nodes are estimated below the truth (mean signed error −0.016 pu, mean |error| 0.019 pu), and the
offset jumps across regulators: bus 160 −0.003 pu → bus 160r −0.034 pu; buses downstream of 160r
(67, 83, 97, 300) ≈ −0.033 to −0.041 pu; 150r −0.023 pu.

Voltage ratio across each regulator, truth (`test_x_true.npz`) vs estimate:

| regulator, phase | true ratio (row 46 / row 600) | estimated ratio | estimate − truth |
|---|---|---|---|
| 150 → 150r, all phases | 1.0187 / 1.0250 | 1.0000 | −0.019 / −0.025 |
| 160 → 160r, phase 1 | 1.0375 / 1.0500 | 1.0022 / 1.0001 | −0.035 / −0.050 |
| 160 → 160r, phase 2 | 1.0188 / 1.0125 | ~1.000 | −0.018 / −0.013 |
| 160 → 160r, phase 3 | 1.0250 / 1.0250 | ~1.000 | −0.025 / −0.026 |
| 25 → 25r, phase 1 | 1.0062 / 1.0125 | 1.0000 | −0.006 / −0.013 |
| 9 → 9r, phase 1 | 0.9937 / 0.9937 | 1.0000 | +0.006 / +0.006 |

The estimator infers ratio ≈ 1.000 every time; the true ratios move in 0.625% tap steps with load. The
regulators' effective admittance in the estimator's Y-bus is a frozen snapshot (`observability/
zero_injection.py` `_regulator_buses` docstring already notes the tap drift), so their voltage boost is
invisible to it. The size of the error equals the boost, which matches the ~2% median and the bus-83
"weakest node" (downstream of the largest boost, 160r).

## 4. Two related estimator-side observations (not investigated further)
- Meter P does not match the model at the true state either (row 46: M001_P recorded 21.43 vs modelled
  19.48 kW; M005_P 21.84 vs 17.70), while meter Q is close. A tap-ratio error changes the modelled power
  flows too, so this may be the same cause.
- The global recalibration (rule 81ff01a) gives the meter P channels very large sigmas (M001_P 6,404 kW
  on a ~20 kW load) because of large dP/dδ, so the estimator effectively ignores meter active power. If
  the angle/tap model is fixed, this recalibration should be revisited.

## 5. Consequence for the placement work
- Placement-driven accuracy improvements (greedy A/D-optimal, Singh index) cannot show any effect while
  this floor exists; the literature methods all assume an unbiased model.
- **Recommended next step for the estimator owners:** model regulator taps (tap ratio as an estimated
  state, or tap position as a measured input from the regulator controller) and compute HEAD_P from the
  network side of bus 150 (power into the feeder) rather than through the source element.
- Once that is fixed, rerun `validation/bias_control.py` and `validation/paired_ab.py`; only then do
  placement comparisons on accuracy become meaningful. Until then, placement can be judged on coverage,
  redundancy (BOI/SORI) and, later, attack detection.
