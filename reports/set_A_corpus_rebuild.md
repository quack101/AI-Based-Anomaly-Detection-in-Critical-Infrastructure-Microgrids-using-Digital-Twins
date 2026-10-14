# Set A corpus rebuild (Step 1)

**Result.** The original set-A Phase-6 corpus is reproduced **exactly** for every file whose
content is deterministic: 9 of 12 files match the provided SHA256 hashes. The 3 that differ are the
`*_ground_truth.csv` files, and the only reason is the `solve_time_s` column, a wall-clock timing
that differs on every run. Every other field in those files is bit-identical between runs.

The original corpus was generated with a registry state that **was never committed**: the 168
channels enabled by `0366af8`, but with the **sigmas from before 2026-10-10** (the `ae0fcfb` values,
before `M042_Q` was widened and before the global recalibration).

Rebuilt corpus: `simulation/phase6_corpus_Arebuilt/` (not committed; regenerate with the script
below). `simulation/phase6_corpus/` was never written.

## Seed
`validation/generate_phase6_corpus.py:60`: `RNG_SEED = 42`, used at line 147 as
`rng = random.Random(seed)`, one stream across all 86,400 timesteps. Noise is added in
`sensors/sensor_layer.py` `build_measurement_vector()` as `value += rng.gauss(0.0, entry["sigma"])`,
once per enabled registry entry in file order.

## Hash comparison (reports/corpus_hashes_A.csv, not committed)

| file | result |
|---|---|
| train/val/test `_telemetry.csv` | **match** (candidate T168_orig) |
| train/val/test `_labels.csv` | **match** |
| train/val/test `_x_true.npz` | **match** |
| train/val/test `_ground_truth.csv` | differ: `solve_time_s` only (see below) |

## Which registry produced the original corpus

The corpus noise depends on each enabled entry's sigma and on the enabled-entry order. The
multiphase registry has four possible states, all generated in one pass (see method):

| candidate | enabled | sigmas | telemetry |
|---|---|---|---|
| T112 | 112 (`ae0fcfb` flags) | `ae0fcfb` | differ |
| **T168_orig** | **168 (`0366af8` flags)** | **`ae0fcfb`** | **match, all 3 splits** |
| T168_m042 | 168 | `0366af8` (= `2dd08f7`): M042_Q widened | differ |
| T168_recal | 168 | `81ff01a`: global recalibration (current branch) | differ |

`0366af8` and `2dd08f7` have identical multiphase sigmas on the 168 enabled entries. `ae0fcfb`→`0366af8`
changes only `M042_Q`'s sigma; `2dd08f7`→`81ff01a` changes 103 sigmas.

Consequence: the frozen set-A corpus was generated **before** both sigma changes on 2026-10-10, but
the estimator now runs it against the **recalibrated** registry. The corpus noise therefore uses the
old sigmas while the estimator weights use the new ones. That is how set A was evaluated, so set B
must be built the same way to compare like with like: corpus noise from the pre-recalibration
sigma rule, estimator weights from the recalibrated rule.

## Why ground truth differs
`solve_time_s` is `time.perf_counter()` around `dss.Solution.Solve()`. Two back-to-back runs of
the same 2,000 timesteps differ in `solve_time_s` only; `converged`, `iterations`, `v_pu_min/max`,
`losses_kw/kvar`, `feeder_head_p_kw/q_kvar` and all per-phase voltages are identical. The
`x_true.npz` files, built from the same solves, match the original hashes, so the physics is
reproduced exactly.

## Method
`validation/rebuild_corpus_candidates.py` mirrors `generate_phase6_corpus.main()` (same load updates
for the 49 set-A meters, same splits, same CSV writers) and writes all four candidate telemetries
in one pass. This is exact because CPython's `random.gauss(mu, sigma)` returns `mu + z*sigma` with
`z` independent of `sigma`: one `Random(42)` stream drawn as `gauss(0.0, 1.0)` gives `z`, and
`value + (0.0 + z*sigma_c)` equals what the generator writes with `sigma_c`. The 112-entry candidate
has its own `Random(42)` stream. Verified before the full run: for 300 timesteps each candidate's
telemetry is byte-identical to the project's own `build_measurement_dict()` driven by that commit's
registry. Full run: 86,400 timesteps in 265 s.

Run: `python -m validation.rebuild_corpus_candidates`

## Finding outside Step 1: the physics depends on the meter set
`generate_phase6_corpus.py` (lines 189–194) updates P and Q **only for the 49 metered loads**; the
other 42 loads stay at their DSS nominal values for the whole 60 days. So the ground-truth feeder
state differs depending on which loads are metered. Set B generated the same way would be evaluated
on a different physical feeder from set A, with some of B's meters on loads that never change. A
placement comparison needs the same physics for both sets. This must be decided before Step 3.
