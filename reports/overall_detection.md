# Overall catch rate of the detection system

`validation/overall_eval.py`. Every attack type, one window at a time on the clean test
telemetry; **numbers below are from the held-out ("report") half** of each benchmark (odd window
positions). Rules were chosen on the other half. Alarm = 3 consecutive steps above threshold;
thresholds at 99% of clean val.

## Final system

**V | P | AE**: alarm if any of the three fires:
- V-peer (`detect_a2_peer.py`): each meter's voltage predicted from the other meters, trailing means
- P-peer (`detect_peer_power.py`): each meter's active power predicted from the other meters
- AE-Mahalanobis: the existing learned detector (saved v3 weights)

False alarms on the clean test series: **1.13% of minutes**.

| Attack | Existing detector (AE) | V \| P | **V \| P \| AE** |
|---|---|---|---|
| A1 random multi-channel bias (27) | 16 (59%) | 19 (70%) | **22 (81%)** |
| A2 single-meter bias (27) | 4 (15%) | 20 (74%) | **20 (74%)** |
| A3 load redistribution (58) | 26 (45%) | 39 (67%) | **40 (69%)** |
| A4 stealthy, library sizes (6) | 6 (100%) | 6 | **6 (100%)** |
| A5 stealthy, library sizes (9) | 9 (100%) | 9 | **9 (100%)** |
| **All 127 windows** | **61 (48%)** | 93 (73%) | **97 (76%)** |
| A1-A3 only | 41% | 70% | **73%** |
| A1-A3, windows <= 60 min | 33% | 61% | **67%** |

By strength (V|P|AE): A1 L0/L1/L2 67/78/100%; A2 L0/L1/L2 44/78/100%; A3 2%/5%/10% 65/55/89%.
By length: A1 15/60/240 min 67/78/100%; A2 56/78/89%; A3 43/77/93%.

Chance that a clean window alarms anyway (V|P|AE): 7% (15 min), 23% (60 min), 64% (240 min).
The 240-minute rows are therefore weak evidence; the <= 60-minute rows are the fair comparison.

## Can it reach 80-90% on every type?

**At ~1% false alarms: not with this data.** What was tried (tuning half only):

| idea | result |
|---|---|
| Lower the threshold (98% / 95% of val) | 80-89% on A1-A3, but false alarms 2.1% / 5.3% per minute, and a clean 15-minute window then alarms by chance 12% / 24% of the time: the gain is largely false alarms |
| One combined score (max of normalised scores) instead of OR | no better than OR at equal false-alarm rate |
| Trailing means, top-k group statistics, AR(1) whitening on P (A3) | at most +2 of 58 held-out windows |
| Subtract each meter's recent baseline | noise unchanged (sd 1.00 -> 1.02): the P prediction error is short-range, not a slow offset |
| Add Q channels / per-meter joint P+Q statistic | A3 41 -> 42 of 59, A2 unchanged |

**Why:** the misses are the weakest levels and the shortest windows. A 2% A3 shift moves each
targeted meter by ~17 kW against ~5 kW of prediction error per minute; an L0 A2 voltage bias is
0.005 pu against 0.005 pu meter noise. These are at the noise floor of 1-minute data from 49
meters. Pinceti et al. (2019) report the same for load redistribution below 10%.

**What would raise it (needs new data or hardware, not more tuning):**
1. **More meters / better placement:** peer predictions improve with more and better-spread
   peers (Hybrid v3 coverage 79% vs set A 63% at 49 meters).
2. **Faster or more precise meters:** the A2 L0 bias equals the meter noise.
3. **Combine with the physics check (BDD)** on the benchmark (needs the estimator per row on
   Windows). BDD catches A1/A2 differently from the data-driven checks.
4. **Moving-target defence** for A3/A4/A5 (Rajasekaran et al. 2022): changes the grid so the
   attacker's model is wrong; needs D-FACTS hardware.

## Reproduce
```
python -m validation.overall_eval        # ~3 min; writes overall_eval_summary.json, overall_eval_windows.csv
python -m validation.meter_joint_test     # Q / per-meter joint test
```
Needs `dataset/attack_bench` (from `RUN_ATTACK_BENCH.bat`).
