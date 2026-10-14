# Placement metrics from the literature

> **Result files are not committed.** `reports/measurements/`, `reports/literature_metrics/` and the generated registries (`config/sensor_registry*_{Arebuilt,Aall91,Ball91}*.json`) are regenerable output; rerun the scripts named next to each number to recreate them.

The A-vs-B comparison (`reports/placement_A_vs_B_measured.md`) used estimation accuracy, convergence,
null-space dimension and weakest-node error. This report adds the metrics that published
meter/PMU-placement papers use and that had not been computed yet. Every number comes from
`reports/literature_metrics/*.json` (written by `validation/literature_metrics.py`) or the files named
next to it. The papers were read at abstract/summary level; check exact definitions in the PDFs before
citing them.

**Bottom line.** On every literature metric that can be computed for this feeder, HybridV2 (set B)
is **not better** than set A where accuracy is concerned (identical to four significant figures), and
on redundancy it **trades depth for reach**: it covers more nodes but with less redundancy. Degree
placement beats every method, including both Hybrid versions, on both BOI/SORI measures.

## 1. Estimation accuracy: Singh, Pal & Vinter (IEEE TPWRS 2009)
*Metric in the paper:* a per-bus probability index for the relative errors in voltage magnitude **and
angle** being within limits (bounded via a two-sided Chebyshev inequality), meters added to improve it.
*Here:* the **empirical** per-bus share of points where both errors are within the limit, on the same
60 points and the same physics for both sets (`validation/paired_ab.py` arrays). Source:
`reports/literature_metrics/accuracy_extras.json`.

| | Set A | Set B (HybridV2) |
|---|---|---|
| angle error median / p95 / max (deg) | 1.782 / 4.136 / 6.224 | 1.783 / 4.136 / 6.231 |
| share of node-points with \|V\| error ≤ 1% / 2% / 5% | 29.4% / 48.4% / 99.4% | 29.5% / 48.5% / 99.5% |
| per-bus index, \|V\| ≤ 2% and angle ≤ 2°: mean over buses | 0.352 | 0.353 |
| per-bus index, \|V\| ≤ 5% and angle ≤ 3°: mean / worst bus | 0.848 / 0.300 | 0.848 / 0.317 |
| per-bus index, \|V\| ≤ 5% and angle ≤ 5°: buses ≥ 95% / worst bus | 93.8% / 0.833 | 93.8% / 0.900 |

A 1° angle limit was also computed: almost no bus meets it for either set (median angle error ~1.8°),
so it does not separate them. B is marginally better on the worst bus at the loose limits; everything
else is identical.

## 2. Optimal-design criteria: Picallo, Anta & De Schutter (IEEE TPWRS 2019, tested on IEEE-123)
*Metric in the paper:* A-, D-, E-optimal criteria of the estimation-error covariance (trace, log-det,
largest eigenvalue), with bounds on the optimality gap.
*Here:* **empirical** versions from the 60 measured |V| errors, because the analytic covariance is not
usable on this feeder: its pseudo-inverse truncates ill-conditioned directions to zero variance (see the
`validation/measure_meter_set.py` docstring). Source: `accuracy_extras.json`.

| | Set A | Set B |
|---|---|---|
| A-criterion: trace of error second moment (sum of per-node MSE, pu²) | 0.17597 | 0.17599 |
| E-criterion: largest eigenvalue (pu²) | 0.16692 | 0.16698 |
| D-criterion (log det) | not computable: 60 samples of 278 nodes give rank ≤ 60 | same |

Identical (B +0.01%). The largest eigenvalue is almost the whole trace, so the error is dominated by
one shared direction (a common bias across nodes), not by placement-specific weak spots.

## 3. Redundancy: BOI / SORI (Ahmed et al. Energies 2024; Hyacinth & Gomathi Energies 2021)
*Metric in the papers:* BOI = number of PMUs observing a bus; SORI = sum of BOI. Used to rank
placements that tie on PMU count.
*Here:* our sensors are not PMUs, so the measurement analogue: BOI(node) = number of enabled measurement
rows whose Jacobian depends on that node's |V| (an injection meter depends on its bus and neighbours,
like a PMU). All methods at 49 meters, each with registries built by the same rules (meter P,Q,|V| +
head), flat start. Source: `reports/literature_metrics/boi_sori.json`.

| method | measurements | SORI | nodes seen (of 278) | unseen |
|---|---|---|---|---|
| Degree | 162 | **766** | **74.1%** | 72 |
| Community4 | 168 | 667 | 64.0% | 100 |
| Set A (frozen) | 168 | 664 | 63.3% | 102 |
| Community12 | 168 | 655 | 62.6% | 104 |
| Hybrid v1 (current code) | 168 | 624 | 62.9% | 103 |
| Random (10 seeds) | 164.4 | 597 ± 52 | 62.1% ± 3.3% | 105 |
| HybridV2 (set B) | 168 | 576 | 68.3% | 88 |

HybridV2 reaches 14 more nodes than set A but has the second-lowest redundancy: spreading meters
farthest-first leaves fewer nodes seen twice. Degree wins both, because high-degree buses touch more
neighbours.

## 4. Minimum meters for observability (standard PMU-placement headline)
Not reachable on this feeder with house meters alone, for any method. In the tier-2 sweep
(`validation/sweep_4a_tier2.py`, `artifacts/20261009T192004Z/sweep_4a_tier2/cells.csv`), every one of
152 placements has rank(H) = m: at most 170 of 553 states even with all 85 buses metered. Observability
comes from the zero-injection constraints and pseudo-measurements, not from meter placement.

## 5. Security index α_k: Sou, Sandberg & Johansson (arXiv 1204.6174)
*Metric:* minimum number of meters an attacker must compromise to fake one measurement.
*Here:* computed in the tier-2 sweep (L1 relaxation). With house meters only, every meter is a critical
measurement (critical count = m in all 152 placements) and α_k = 1 for every meter of every method.
No method differs. Not recomputed.

## 6. Vulnerability index: Deng, Xiao & Lu
*Metric:* per-meter vulnerability = likelihood of being targeted × damage if compromised, under a
protection budget. **Not computed:** it needs an attack-damage model (the attack injector and impact
quantification of RES.md phases 8 and 11), which this branch does not build.

## 7. Placement cost vs random: micro-PMU anomaly-detection paper (arXiv 1708.00118)
*Metric:* a coverage cost function, random vs greedy vs exhaustive, plus computation time.
*Here:* the closest analogue is the earlier coverage comparison (HybridV2 vs 100 random placements on
our own coverage tests, where it beats 79–100% of random placements). Those tests are our own, and
HybridV2 was designed to win them, so this is not independent evidence.

## Summary

| literature metric | computed? | set A vs set B |
|---|---|---|
| \|V\| accuracy vs truth (Singh; our main scorecard) | yes | same (paired p = 0.37) |
| angle accuracy (Singh) | yes | same |
| per-bus probability index (Singh, empirical) | yes | same; B marginally better worst bus at loose limits |
| A / E optimal-design criteria (Picallo, empirical) | yes | same |
| D criterion (Picallo) | not computable from 60 samples | — |
| BOI / SORI (Ahmed; Hyacinth & Gomathi) | yes, all methods | B covers more nodes, lower SORI; Degree best |
| min meters for observability | not reachable (rank = m ≤ 170 < 553) | — |
| security index α_k (Sou et al.) | yes | tie (α_k = 1 everywhere) |
| vulnerability index (Deng et al.) | no: needs attack-damage model | — |

Papers: [Singh, Pal & Vinter 2009](https://spiral.imperial.ac.uk/entities/publication/882cb238-66db-40a1-8a0d-0b4e79dce650);
[Picallo et al. 2019](https://arxiv.org/pdf/1908.03081); [Picallo et al., convex OPP](https://arxiv.org/pdf/1810.12195);
[BOI/SORI](https://dspace.bits-pilani.ac.in/items/8792b1a0-9eba-4fa0-b5d0-e3313911d281);
[Ahmed et al. 2024](https://doaj.org/article/70398bea6ea84ff0a45dc34d19ca6c91);
[Hyacinth & Gomathi 2021](https://doaj.org/article/03d87c392fbc49e6b3ddda5a0cb718b7);
[Sou et al., security index](https://arxiv.org/pdf/1204.6174);
[Deng, Xiao & Lu](https://personal.ntu.edu.sg/egxxiao/preprints/v14.pdf);
[micro-PMU anomaly detection](https://arxiv.org/pdf/1708.00118).
