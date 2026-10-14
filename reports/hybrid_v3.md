# HybridV3: preferring well-connected buses

> **Result files are not committed.** Numbers below come from `reports/literature_metrics/structure_metrics.json`, written by `python -m validation.placement_structure_metrics`.

**Result.** HybridV3 (`placement/hybrid_v3.py`, default `connection_priority="first_balanced"`) gets close to
Degree placement's redundancy while keeping Hybrid's phase balance. At 49 meters: SORI 702 (v1 624, set A 664,
v2 576, Degree 766), coverage 73.0% (Degree 74.1%), 6 dead-end meters (v2 32), phase deviation 1.0 pp (Degree
7.1 pp, over the 5 pp tolerance). At 20, 30 and 60 meters it covers more phase-nodes than Degree. Estimation
accuracy was **not** measured: with the regulator-tap error floor (`reports/step0_bias_control.md`) no
placement can change it, so accuracy for v3 only becomes meaningful once the estimator is fixed.

## Why
Of the 85 load buses, 36 are dead ends (topology degree 1) and exactly 49 are not. Degree placement at 49 meters
takes all 49 connected buses, which is why it has the highest redundancy: a meter on a dead end appears in few
measurement rows and touches one neighbour. v2's farthest-first branch order picks line ends (set B: 32 dead
ends), which is why its redundancy is the lowest. But Degree ignores phase balance and size classes.

## What v3 changes (on top of v2)
1. A load is *connected* if its bus has degree ≥ 2. Within a branch, connected loads are offered first.
2. The branch order puts branches with a connected load first, then v2's fewest-meters / farthest-first order.
3. Phase balance wins over connectivity: if no top-ranked branch can serve the most under-represented phase,
   v3 takes the best connected load on that phase, and only then a dead end on that phase.

Size-class quotas, scores, the phase-aware pick and the top-up are inherited unchanged. The two other modes are
kept for comparison: `within_tier` (branch diversity first, connections second) and `first` (connections first,
no phase override).

## Results (flat start, same registry rules for every method)

| meters | method | SORI | coverage | dead ends | branches | phase dev (pp) |
|---|---|---|---|---|---|---|
| 20 | Hybrid v1 | 256 | 21.6% | 14 | 19 | **8.2 ✗** |
| 20 | HybridV2 | 218 | 35.6% | 13 | 20 | 1.6 |
| 20 | **V3 (default)** | 279 | **46.8%** | 0 | 20 | 1.6 |
| 20 | Degree | **387** | 45.3% | 0 | 17 | **9.2 ✗** |
| 30 | Hybrid v1 | 368 | 35.3% | 19 | 29 | **5.5 ✗** |
| 30 | HybridV2 | 303 | 48.2% | 21 | 30 | 1.9 |
| 30 | **V3 (default)** | 433 | **60.4%** | 0 | 28 | 1.9 |
| 30 | Degree | **541** | 56.1% | 0 | 22 | 4.4 |
| 49 | Hybrid v1 | 624 | 62.9% | 27 | 42 | 3.2 |
| 49 | HybridV2 | 576 | 68.3% | 32 | 49 | 0.8 |
| 49 | **V3 (default)** | 702 | 73.0% | 6 | 36 | 1.0 |
| 49 | Degree | **766** | **74.1%** | 0 | 33 | **7.1 ✗** |
| 60 | Hybrid v1 | 729 | 72.7% | 34 | 51 | 1.6 |
| 60 | HybridV2 | 705 | 75.9% | 34 | 53 | 0.9 |
| 60 | **V3 (default)** | 829 | **78.8%** | 11 | 42 | 0.7 |
| 60 | Degree | **863** | 77.7% | 11 | 40 | 3.1 |
| 70 | Hybrid v1 | 833 | 77.3% | 35 | 53 | 1.9 |
| 70 | HybridV2 | 821 | 78.8% | 35 | 53 | 0.5 |
| 70 | **V3 (default)** | 913 | **81.3%** | 21 | 49 | 0.7 |
| 70 | Degree | **921** | 80.9% | 21 | 46 | 1.1 |

✗ = over the 5 pp phase tolerance of `tests/test_placement_regression.py` (which only tests v1 at 45 and 49
meters; v1 fails it at 20 and 30). The script also reports the other two v3 modes: at 49 meters `first` reaches
SORI 752 but 6.8 pp phase deviation, `within_tier` keeps 49 branches but only SORI 644.

**Trade-off.** v3 meters fewer distinct branches than v2 (36 vs 49 at 49 meters), because connected buses
cluster on fewer branches. Coverage still rises because a connected meter sees its neighbours too.

## Limits
- SORI and coverage are structural (which states a measurement touches), not accuracy.
- Not yet run: the security index and the attack tests. With house meters only, every meter is critical for
  every method (`reports/placement_literature_metrics.md` §5), so v3 is not expected to change that.
- To measure accuracy later: `python -m placement.select_meters --strategy v3` (writes
  `simulation/representative_meters_C.json`), then steps 2–4 of "How to reproduce" in
  `reports/placement_A_vs_B_measured.md` with set name `Call91`.

Tests: `tests/test_hybrid_v3.py` (valid selection, phase deviation ≤ 5 pp and fewer dead ends than v2 at
20/30/45/49/60/70 meters, determinism).
