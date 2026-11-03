# A3, A4 and A5 detection: peer-consistency checks on a 147-window OpenDSS benchmark

Branch `a2-detection` (local, not pushed). New files only. Follows `reports/a2_detection.md`
(A2) and `reports/literature_survey_A3_A4_A5.md` (survey). numpy only for detection;
OpenDSS only to generate attacks.

## 1. What was built

| File | What |
|---|---|
| `validation/detect_peer_power.py` | **P-peer** detector (A3): ridge prediction of each meter's P from all other meters, max \|error\|/sd, threshold 99% of clean val, alarm after 3 steps |
| `validation/detect_a2_peer.py` | **V-peer** detector (A2), unchanged |
| `validation/gen_attack_bench.py` + `RUN_ATTACK_BENCH.bat` | generates A3/A4/A5 windows with the library's own `attack()` and OpenDSS (Windows, `.venv-1`) |
| `validation/eval_attack_bench.py` | scores V-peer, P-peer, V-or-P and the existing AE-Mahalanobis on the benchmark |
| `validation/eval_a45_scaled.py` | A4/A5 at realistic (small) sizes |
| `validation/a3_variants.py` | A3 variants, tuned on even ids, reported on odd ids |
| `dataset/attack_bench/` | the 147 generated windows (`a_<id>.npy`, `windows.jsonl`); not committed (dataset/ is git-ignored) |

**Benchmark generation (Windows, OpenDSS 0.9.4, Python 3.13, 6 workers, 21.8 min):**
117 A3 (levels 2/5/10%; full access x step/ramp/pulse x 15/60/240 steps x 3; partial access x
step/ramp/pulse x 15/60 x 2) + 15 A4 + 15 A5 (0.01/0.02/0.05 pu, 15 steps, 5 each). 147 ok, 0
failed. **Self-check:** fdia_v1 test window 0 (A3) was rebuilt by the same script and matches
the shipped dataset to 1.9e-13, so the benchmark is made exactly like fdia_v1.

Scoring is the same as for A2: each window's attack vectors are added to the clean test
telemetry rows they were computed on, and the detector sees 30 clean rows of history first.
Detected = an alarm (3 consecutive steps above threshold) inside the window. **Chance** = the
probability that a CLEAN window of the same length alarms with the same rule.

## 2. A3 (load redistribution), 117 windows

False alarms per step on the clean test series (K=3): V-peer 0.62%, **P-peer 0.37%**,
**V or P 1.01%**, AE-Mahalanobis 0.25%.

| A3 windows | V-peer | P-peer | **V or P** | AE-Maha (existing) |
|---|---|---|---|---|
| **all 117** | 62 (53%) | 75 (64%) | **83 (71%)** | 60 (51%) |
| level 2% (39) | 12 | 14 | **20** | 7 |
| level 5% (39) | 20 | 24 | **26** | 19 |
| level 10% (39) | 30 | 37 | **37** | 34 |
| 15 steps (45) | 14 | 24 | **25** | 16 |
| 60 steps (45) | 27 | 29 | **33** | 25 |
| 240 steps (27) | 21 | 22 | **25** | 19 |
| full access (81) | 41 | 51 | **56** | 38 |
| partial access (36) | 21 | 24 | **27** | 22 |

Chance at 15 / 60 / 240 steps: P-peer 2 / 8 / 28%, V or P 6 / 19 / 53%, AE-Maha 3 / 8 / 25%.
The 240-step rows are therefore weak evidence for every detector; 15- and 60-step windows are
the fair comparison.

- **The gain is at small and short attacks:** 2%: 20/39 vs 7/39; 15 steps: 25/45 vs 16/45. At
  10% the existing detector already catches 34/39 (P-peer 37/39).
- **2% is the limit:** a 2% shift moves each targeted meter by roughly 17 kW against a P
  peer-error sd of about 5 kW (3-7 kW), below the 5.3 sd threshold needed for <1% FAR.
- **Tried, not kept** (even/odd split, `a3_variants.py`): trailing means of the signed error
  (L = 5, 15), the sum of the top-4 squared errors. Best variant +2 of 58 held-out windows, not
  more than noise.
- fdia_v1 (18 test windows), V or P, K=3: **14/18 (A1 9/12, A2 3/3, A3 2/3)** at 0.98% FAR on
  untouched clean rows; val 15/18 (A3 3/3). The current fusion: 10.6 +/- 1.2 /18.

## 3. A4 / A5 (stealthy), 15 + 15 windows

**At the library's sizes (0.01-0.05 pu) every detector catches every window (30/30),
including the existing AE-Mahalanobis.** Reason, measured: a 0.01 pu kept-direction state shift
makes meters report **3,000-10,000 kW** of P (the whole feeder carries about 3,490 kW). The
attack satisfies the estimator's model, so the residual test cannot see it, but the readings are
physically absurd as loads.

**Realistic sizes, by scaling:** A4 is linear (a = H c with c = size x unit direction), so
scaling an OpenDSS-generated A4 vector gives the A4 attack of the smaller size on the same rows.
A5 adds a Newton correction that vanishes as the size goes to 0; measured \|\|a\|\| / size is the
same at 0.01, 0.02 and 0.05 pu for A4 and A5, so scaling is a close approximation for A5.

| size (pu) | typical max \|dP\| | V-peer | P-peer | V or P | AE-Maha |
|---|---|---|---|---|---|
| A4 0.000001 | ~0.05 kW | - | 1/15 | - | - |
| A4 0.000003 | ~0.13 kW | - | 3/15 | - | - |
| A4 0.00001 | ~4.5 kW | 2/15 | **15/15** | 15/15 | 14/15 |
| A4 0.00003 | ~13 kW | 10/15 | 15/15 | 15/15 | 15/15 |
| A4 0.0001 | ~45 kW | 13/15 | 15/15 | 15/15 | 15/15 |
| A4 0.0003-0.001 | 134-447 kW | 15/15 | 15/15 | 15/15 | 15/15 |
| A5 0.00001 | ~4.0 kW | 1/15 | **15/15** | 15/15 | 15/15 |
| A5 0.00003 | ~12 kW | 8/15 | 15/15 | 15/15 | 15/15 |
| A5 0.0001-0.001 | 40-400 kW | 13-15/15 | 15/15 | 15/15 | 15/15 |

(15-step windows; chance 2-6% per window. "-" = not run at that size.)

- Detection holds down to about **1e-5 pu** and breaks between 3e-6 and 1e-5 pu. The flagged
  channel is usually not an attacked meter's own P (0.4 kW change against 5.7 kW noise): the
  stealthy vector moves all voltages in a pattern the other meters' P cannot explain, and the
  peer prediction picks that up. This is the spatial-consistency effect the GNN papers rely on
  (Boyaci et al.; Abdulin & Narimani).
- **Honest conclusion: the existing AE-Mahalanobis already detects the library's A4/A5**; the
  new detectors do not add much here. Only the residual/BDD test is blind to them, by
  construction.

## 4. Limits

- **A3 at 2%:** about half caught by V or P. A 2% shift is close to the meter-level noise.
- **P/Q-only single-meter attacks** (from the A2 work) remain near chance for V-peer.
- **A5 at small sizes is approximated by scaling**, not re-generated: the Windows session locked
  before the small-size OpenDSS run (`RUN_A45_SMALL.bat`) could start. A4 scaling is exact.
- **V or P runs at 1.01% FAR** (two detectors OR-ed, each at ~0.4-0.6%). Report it as ~1%.
- The window rule (any alarm in the window) is inflated for long windows; chance is reported
  next to every number for that reason.
- BDD is not re-run on the benchmark (it needs the estimator per row); the stealth claim for
  A3-full/A4/A5 rests on their construction and the phase-8 reports.

## 5. Reproduce

```
RUN_ATTACK_BENCH.bat                        # Windows: 147 windows, ~22 min (needs ..\.venv-1)
python -m validation.eval_attack_bench      # A3/A4/A5 table, ~3 min
python -m validation.eval_a45_scaled        # small-size A4/A5, ~3 min
python -m validation.a3_variants            # even/odd A3 variants, ~3 min
```
