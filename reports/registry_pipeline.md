# Registry pipeline for any meter set (Step 2)

`sensors/build_registry_for_set.py` builds a meter set's registries with rules only. Each step below is
the hand-made step it replaces, with where it lives and the commit that made it.

| # | what was done to set A | where / commit | rule in the pipeline |
|---|---|---|---|
| a | base registry: meter P,Q enabled, `|V|` and head disabled, sigma 1% of reading (P/Q/`|I|`), 0.5% of nominal (`|V|`) | `sensors/generate_registry.py:68-69` (`SIGMA_FRAC_*`), `:76` `build_meter_entries`, `fill_sigmas`; registry at `ae0fcfb` | `generate_registry.main(meters_path=...)` unchanged |
| a2 | substation head P,Q switched on (multiphase 106 → 112) | Phase-6 "deployment decision", `validation/generate_phase6_corpus.py` docstring; flags in `ae0fcfb` | enable `HEAD_P_*`, `HEAD_Q_*` (multiphase only) |
| b | every `V_magnitude` channel switched on (56: 49 meter `|V|`, 4 extra phases at M043/M044, 3 `HEAD_V_*`; 112 → 168) | commit `0366af8` (2026-10-10 01:08) | enable every `V_magnitude` entry (multiphase only) |
| c0 | `M042_Q` sigma widened to 5080.46 kvar (dQ/dV at bus 83) | `0366af8`, mirrored to the default variant in `2dd08f7` | subsumed by c: the global formula applies the same widening to every channel |
| c | global sigma recalibration | commit `81ff01a` (02:09), `validation/phase7_global_sigma_recalibration.py`; constants at `:54-56`, formula `:121` | same formula and constants on the set's own corpus, see below |
| c1 | 6 channels reverted to the original sigma by hand: `M004_P/Q`, `HEAD_Q_A/B/C` (regulator bus), `M040_P` ("unexplained outlier") | `81ff01a` commit message | **regulator-bus rule only** (decision Q3 = (ii)): every P/Q channel whose bus is in `observability.zero_injection._regulator_buses()` keeps its sigma. `M040_P` (bus 60, not a regulator bus) is recalibrated. |
| d | `REGZI_83_A_Q`, `REGZI_83_B_Q` sigma amplified | `observability/estimator_model.py:176` (`AMPLIFIED_SIGMA_REGZI_IDS`), `_amplify_bus83_regzi_sigma`; `81ff01a` | no per-set step: these entries come from the network layout (regulator/capacitor-adjacent nodes) at a flat start, not from the meters, so they are identical for every meter set |

Recalibration (c), constants from `phase7_global_sigma_recalibration.py`:
- reference state: the estimate at test row `TARGET_I = 46` (first 1,440 test rows), estimator built with the
  set's noise-stage registry and the set's own corpus;
- Q-type (`Q_injection`, `Q_head`): `amp = |dQ/dV| * 0.04 * v_base`;
- P-type (`P_injection`, `P_head`): `amp = hypot(dP/dV * 0.04 * v_base, dP/ddelta * 0.03)`;
- `sigma_new = hypot(sigma_old, amp)`; `V_magnitude` untouched; one sigma per channel id, written to both variants.

Outputs: `config/sensor_registry{,_multiphase}_<SET>_noise.json` (stages a–b: the corpus noise is drawn from
this) and `config/sensor_registry{,_multiphase}_<SET>.json` (stage c: what the estimator weights with). The
pipeline refuses to write set A's two registry files.

## Verified on set A
The noise stage for set A (`--set Arebuilt`) reproduces, field by field, the registry the frozen corpus was
generated with: same 336 ids in the same order, the same 168 enabled (= `0366af8`), every sigma identical to
`ae0fcfb`, every other field identical. That registry is exactly the one whose corpus matches the original
hashes (`reports/set_A_corpus_rebuild.md`).

The recal stage differs from the saved set-A registry ("A_exact") by design, because the hand-picked
`M040_P` exclusion and the separate `M042_Q` pre-widening are replaced by rules. The differences and their
effect on the scorecard are in `reports/placement_A_vs_B_measured.md`.
