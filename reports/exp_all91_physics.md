# All-91 vs 49-load physics experiment (frozen estimator)

Script: `validation/exp_all91_physics.py`. Every number below comes from this one run unless noted.

## Points

- 30 stratified (seed=777): [46, 94, 154, 166, 181, 216, 285, 288, 359, 368, 412, 505, 535, 562, 621, 753, 801, 824, 844, 868, 903, 913, 941, 985, 1016, 1043, 1104, 1215, 1367, 1416]  [validation/exp_all91_physics.py::determine_points]
- scorecard B_warm (i+1), 30: [47, 95, 155, 167, 182, 217, 286, 289, 360, 369, 413, 506, 536, 563, 622, 754, 802, 825, 845, 869, 904, 914, 942, 986, 1017, 1044, 1105, 1216, 1368, 1417]
- baseline extra_clean, 60: [111, 356, 1587, 1672, 2447, 2762, 2883, 3239, 3320, 3395, 3419, 4331, 4405, 4537, 4558, 4852, 5127, 5341, 5685, 6307, 6518, 6668, 6991, 7087, 7386, 7507, 7533, 7925, 8250, 8314, 8319, 8419, 8535, 8987, 9038, 9188, 9381, 9459, 9492, 9578, 9686, 10266, 10354, 10556, 10775, 10852, 10941, 11332, 11344, 11437, 11504, 11519, 11638, 11803, 11877, 11894, 11981, 12228, 12520, 12541]
- scorecard60 distinct: 60, baseline90 distinct: 90, overlap: 30 (exactly the 30 stratified)
- UNION used for this experiment: 120 points

## Step 4 -- P0 vs P1 under the frozen estimator

### P0
- voltage error (pu), pooled n=33360: median=0.0234 p95=0.0415 max=0.0626
- J: mean=3.0103e+04  J/dof(57)=528.12x  in chi2 95% CI [38.0,79.8]: 0/120
- converged: 120/120  mean iterations: 5.04
- LNR: min=64.8706 median=99.9463 max=156.7683
- dominant channel frequency: {'HEAD_P_B': 59, 'HEAD_P_C': 32, 'HEAD_P_A': 29}
- 5 worst (point, max_v_err_pu): [(1105, 0.06261608873742064), (1104, 0.06261418119152423), (3419, 0.06255190863638378), (8314, 0.056356066054158584), (8319, 0.056352121756539364)]

### P1
- voltage error (pu), pooled n=33360: median=0.0234 p95=0.0419 max=0.0750
- J: mean=3.0849e+04  J/dof(57)=541.20x  in chi2 95% CI [38.0,79.8]: 0/120
- converged: 120/120  mean iterations: 5.02
- LNR: min=37.8001 median=98.3762 max=192.2728
- dominant channel frequency: {'HEAD_P_B': 60, 'HEAD_P_A': 31, 'HEAD_P_C': 29}
- 5 worst (point, max_v_err_pu): [(3419, 0.07504403928810398), (1104, 0.06885671793083141), (8319, 0.06884464223556165), (1105, 0.06260620491242005), (1044, 0.06258064224732582)]

**P0 vs scorecard reference** (median=0.0239, p95=0.0408, max=0.0626): this run gives median=0.0234, p95=0.0415, max=0.0626. Close to reference: True.

## Step 5 -- mechanism checks

### 5a -- z = (pseudo_mean - true_P) / sigma_pseudo, 42 unmetered loads x points
- P0: RMS(z)=1.7646  p95(|z|)=4.1150
- P1: RMS(z)=0.9735  p95(|z|)=1.9044

### 5b -- S = sum_42 |pseudo_mean - nominal_P| (kW), under P0, vs J and voltage error
- Spearman(S, J_P0) = -0.1652 (p=0.07129, n=120)
- Spearman(S, max_v_err_pu_P0) = -0.0526 (p=0.5684, n=120)

### 5c -- mean SIGNED normalized residual of HEAD_P_A/B/C
- P0 HEAD_P_A: mean=95.2797  n=120
- P0 HEAD_P_B: mean=96.9934  n=120
- P0 HEAD_P_C: mean=95.1528  n=120
- P1 HEAD_P_A: mean=93.3143  n=120
- P1 HEAD_P_B: mean=94.4104  n=120
- P1 HEAD_P_C: mean=92.7222  n=120

### 5d -- accuracy and J by hour-of-day bucket
- P0 hour[0-6): n=36  J_mean=1.8540e+04  pooled_median_v_err=0.0177  pooled_p95_v_err=0.0364  pooled_max_v_err=0.0525
- P0 hour[6-12): n=25  J_mean=3.6135e+04  pooled_median_v_err=0.0254  pooled_p95_v_err=0.0435  pooled_max_v_err=0.0626
- P0 hour[12-18): n=38  J_mean=3.2967e+04  pooled_median_v_err=0.0242  pooled_p95_v_err=0.0419  pooled_max_v_err=0.0563
- P0 hour[18-24): n=21  J_mean=3.7562e+04  pooled_median_v_err=0.0269  pooled_p95_v_err=0.0432  pooled_max_v_err=0.0626
- P1 hour[0-6): n=36  J_mean=1.2900e+04  pooled_median_v_err=0.0167  pooled_p95_v_err=0.0396  pooled_max_v_err=0.0501
- P1 hour[6-12): n=25  J_mean=3.9209e+04  pooled_median_v_err=0.0250  pooled_p95_v_err=0.0436  pooled_max_v_err=0.0750
- P1 hour[12-18): n=38  J_mean=3.5717e+04  pooled_median_v_err=0.0249  pooled_p95_v_err=0.0421  pooled_max_v_err=0.0626
- P1 hour[18-24): n=21  J_mean=4.2854e+04  pooled_median_v_err=0.0259  pooled_p95_v_err=0.0443  pooled_max_v_err=0.0689

## Runtime

- truth-build (single process, 120 points x 2 physics): 20.7s
- estimator solves (240 calls, <= 4 workers): 1176.7s
- total: 1197.4s

