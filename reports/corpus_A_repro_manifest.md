# Corpus reproduction manifest (Set A, phase6_corpus)

Date: 2026-10-15. Run window 11:58-12:05 IST. Checked against corpus_hashes_A.csv (12 rows).

## Result
9 of 12 files match the original SHA256 hashes exactly (all telemetry, labels and x_true files).
The 3 *_ground_truth.csv files do not match. They contain the solve_time_s column
(time.perf_counter() around dss.Solution.Solve(), validation/generate_phase6_corpus.py:198-200),
which is wall-clock and changes on every run. Whether solve_time_s is the ONLY differing column
cannot be verified here: the original ground_truth files are lost. reports/set_A_corpus_rebuild.md
(origin/placement-v2, c38eaa7) states it is the only difference in its own rebuild.

## Candidate used
- Worktree: D:\wt_corpus_repro, `git worktree add --detach` at commit 0366af8
  (2026-10-10 01:08:13, "Fix LM lambda ratchet and amplified M042_Q sigma at bus 83").
- Overlay: config/sensor_registry_multiphase.json, ONE field changed in the scratch worktree only:
  M042_Q sigma 5080.461307 -> 0.0999960592236547 (the value at commit ae0fcfb).
  This is a field-level edit, not a whole-file `git checkout <commit> -- <path>` overlay; no single
  commit holds this state. Resulting registry: 168 enabled entries (0366af8 flags), all 336 sigmas
  equal to ae0fcfb's, entry order identical to both commits.
- Evidence: reports/set_A_corpus_rebuild.md (c38eaa7): candidate T168_orig = "168 (0366af8 flags),
  ae0fcfb sigmas -> match, all 3 splits". Diff ae0fcfb->0366af8 measured here: 56 enabled-flag
  changes, 1 sigma change (M042_Q), 0 other-field changes.
- Generator inputs identical (git blob) between 0366af8 and main HEAD 4124982:
  validation/generate_phase6_corpus.py, sensors/sensor_layer.py, sensors/measurement_functions.py,
  sensors/registry.py, simulation/opendss_utils.py, observability/state_vector.py.
- Load data: the 91 CSVs from D:\capstone_dss\dataset\node_loads_91 (COPIED, not linked);
  hashes in D:\node_loads_91_hashes.csv (identical to the audit-time hashes, 91/91).
- Only one candidate was tested; it met the >=9 threshold, so no other candidate was run.

## Seed / environment
- RNG_SEED = 42 (validation/generate_phase6_corpus.py:60), one random.Random stream.
- Python 3.11.9; numpy==2.4.6; pandas==3.0.5; scipy==1.17.1;
  opendssdirect.py==0.9.4; dss-python==0.15.7; dss-python-backend==0.14.5.
- Single process, no multiprocessing.

## Run time
Generator internal: 230.4 s. Wall clock: 237 s (11:59:56 -> 12:03:53). 86,400 timesteps,
0 convergence failures. Split rows: train 60,480 / val 12,960 / test 12,960.

## Hashes (SHA256)
| File | Original (manifest) | Rebuilt | Match |
|---|---|---|---|
| test_ground_truth.csv | B955FDBAEDFFCFBA8C32A2494629191EF43975F5360F94231811A9FB96E0A255 | 4B1FBEFE71F99F39BED5E5E940319D22A1D814AB43D664B5F0507D2BDF4EABA9 | NO (contains solve_time_s) |
| test_labels.csv | 023C135DDECCA03A52469ABB4188D452D8BE8346D72133344C5FB7FAFAE22623 | same | YES |
| test_telemetry.csv | 9AD44AF799F495B4965F07C5D19AB440DCAB71E1AE8EC5FAD0F6C0104B4204A9 | same | YES |
| test_x_true.npz | C1F638EF820684C9FB904D1627B887B7B33818BAC0D9036F1EAB89DB1D56A84B | same | YES |
| train_ground_truth.csv | 64912643FF95E654E07A2132AB93F68841C8D3300C9BA40ABD60730D224193CA | BE13215EA28094360BEBBF2B97B80D89CB157043DD99E04919C307F1728A0C5A | NO (contains solve_time_s) |
| train_labels.csv | 3263CF24253A1AC23FB32CCDB1890632E773C11E3A14A2A8F86636EE60AFEC99 | same | YES |
| train_telemetry.csv | 83115BCA2BBD65C3912621031163A10EDD714A9C89D768FF724702B6428A1A63 | same | YES |
| train_x_true.npz | 12A87E45D038B5E440F98D71E4F556C1B3E4C32C01F6860AC435A3A1B23793A5 | same | YES |
| val_ground_truth.csv | 39F9AE16769E04B14AFBC339E56FB85A3FDE988E845423BF2EA1D8E03DDFE72D | A883A26D8698B4100C22847FE63CDC051795A9E5107ED4C64C21BE0864DCA7F0 | NO (contains solve_time_s) |
| val_labels.csv | 1BA2CFF06739E0A7A9BC4A43F08AA4E02817CDA70F30E186E439DDFC3DAB0A73 | same | YES |
| val_telemetry.csv | 5BE3B47EDF570432DA14FA1B3CCA6C4D2E99868EB4345EA5945B133210D4A173 | same | YES |
| val_x_true.npz | 65746C101311FC51BE71920C939086C735810C3E45177E2EF582800DE047960F | same | YES |

## Where the files are
- D:\capstone_dss\simulation\phase6_corpus\ (12 files, read-only, hashes equal the worktree output)
- C:\Users\Sriharshita\Documents\capstone_backup\phase6_corpus_A\ (12 files, hashes equal)
- D:\wt_corpus_repro\simulation\phase6_corpus\ (generator output, worktree left in place)

## Exact recipe (to regenerate corpus A)
1. `git worktree add --detach <path> 0366af8`
2. In that worktree only, edit `config/sensor_registry_multiphase.json`: set the `sigma` of the entry
   with `"id": "M042_Q"` from `5080.461307` to `0.0999960592236547` (its value at `ae0fcfb`).
   No other field changes. Result: 168 enabled entries, all 336 sigmas equal to `ae0fcfb`.
3. COPY (do not link) the 91 `node_*.csv` files into `<path>/dataset/node_loads_91/`;
   verify them against `reports/node_loads_91_hashes.csv`.
4. From `<path>`, run `python -m validation.generate_phase6_corpus` (seed 42 is hardcoded,
   `validation/generate_phase6_corpus.py:60`). Single process. ~230-265 s.
5. SHA256 the 12 outputs and compare with `reports/corpus_A_hashes.csv`. Expected: 9/12 exact;
   the 3 `*_ground_truth.csv` files differ because of the wall-clock `solve_time_s` column.

Corpus regenerated 2026-10-15 with this recipe; the copies in `simulation/phase6_corpus/` are the
files listed above (ground_truth files are the 2026-10-15 regeneration, not the lost originals).
