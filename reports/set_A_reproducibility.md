# Set A reproducibility

**Question.** `simulation/representative_meters.json` ("set A", 49 loads) does not match a fresh v1 run:
4 of 49 load names differ.

| only in saved set A | only in a fresh v1 run |
|---|---|
| s30c, s64b, s66c, s76a | s12b, s29a, s2b, s84c |

**Answer.** Set A is the exact output of the **previous version** of `HybridPlacement`
(commit `6bc6d9c`, 2026-08-21). `placement/placement_strategy.py` was rewritten afterwards in
`ae0fcfb` (2026-09-20) and set A was never regenerated. The difference is a code-version
difference, not randomness and not a post-processing step.

**Confidence: high.** Running v1 from the old commit reproduces set A exactly (same 49 names,
same order); running the current code reproduces the 4-name difference under every hash seed tried.

## Evidence

**(a) Nothing happens between `select()` and saving.** `placement/select_meters.py` calls
`strategy.select(loads, context)` and passes the result straight to
`save_representative_meters()` (`placement/representative_meters.py`), which writes one record
per selected load in order, with no filtering, sorting or deduplication. A fresh in-memory run
reproducing that path gives the 4-name difference above.

**(b) Not randomness.** Fresh v1 runs (target 49) under `PYTHONHASHSEED` = 0, 1, 2, 3, 4 all give
the same 49 loads and the same 4-name difference from set A.

**(c) The `weights=` change on this branch has no effect.** `HybridPlacement(49)` and
`HybridPlacement(49, weights=<the default weights>)` select identical loads. The current
`placement_strategy.py` on `fix-jacobian-bugs` (git HEAD of that branch) is the code that
produces the 4-name difference; this branch only adds the optional argument.

**(d) Set A was saved before the last placement change.** `git log --follow` on
`origin/fix-jacobian-bugs`:

| file | commits |
|---|---|
| `simulation/representative_meters.json` | `6bc6d9c` 2026-08-21, `33204ac` 2026-08-15 |
| `placement/placement_strategy.py` | **`ae0fcfb` 2026-09-20**, `6bc6d9c` 2026-08-21 |
| `placement/feeder_context.py` | `6bc6d9c` 2026-08-21 |
| `placement/feeder_adapter.py` | `6bc6d9c` 2026-08-21 |
| `core/ieee123_model.py` | `6bc6d9c` 2026-08-21 |

Set A is byte-identical between `6bc6d9c` and `fix-jacobian-bugs`. Only `placement_strategy.py`
changed after it was saved.

**(e) The old code reproduces set A exactly.** In a temporary detached worktree at `6bc6d9c`
(outside the repo, removed afterwards), `HybridPlacement(49).select(...)` returned the same 49
load names as set A, **in the same order**.

What `ae0fcfb` changed in `HybridPlacement` (+416 / −119 lines), any of which can move picks:
- candidates collapsed to one load per bus first (`_one_load_per_bus`, highest kW kept) on the
  three multi-load buses 49, 65, 76 (set A contains `s76a`; fresh runs contain no bus-76 load
  from that split);
- phase balancing moved from a post-hoc swap pass (`_balance_phases`) to an incremental
  phase-aware pick at every selection;
- size classes processed in global round-robin turns instead of one class exhausted at a time;
- default `target_meters` read from `config/reference_configuration.json`.

**(f) No test checks this.** `tests/test_placement_regression.py` checks meter counts, no
duplicate buses, phase tolerance, and determinism of two **fresh** runs against each other
(`test_hybrid_is_deterministic`). `test_step2_gate_fixtures_are_consistent` only checks fixture
files against each other. Nothing compares the saved set A with a fresh run, so the drift was
invisible.

## Consequence for the A vs B comparison

Set A is **not** "what v1 produces today". It is v1 as of 2026-08-21. A v1-vs-v2 comparison must
say which v1 it means. This report does not change set A: it is frozen and the estimator,
scorecard and attack work on `fix-jacobian-bugs` are built on it. Set A is the incumbent to beat;
the comparison is "frozen set A vs set B (HybridV2)", and a fresh current-code v1 set is a
separate third set if one is wanted.

Reproduce: run `HybridPlacement(49).select(loads, context)` on the current code and at commit
`6bc6d9c`, and diff the `name` fields against `simulation/representative_meters.json`.
