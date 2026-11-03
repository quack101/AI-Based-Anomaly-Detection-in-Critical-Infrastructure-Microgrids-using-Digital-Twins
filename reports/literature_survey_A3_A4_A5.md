# Literature survey: detecting A3 (load redistribution) and A4/A5 (stealthy) attacks

Date: 2026-11-04. Follows `reports/literature_survey_A2_A3_A4.md` (A2) and
`reports/a2_detection.md`. Only papers whose text was read are summarised with numbers;
numbers taken from a model-extracted part of a PDF (not verbatim) are marked [S] and must be
checked against the PDF before they go into the paper.

## 0. What our attacks are (attacks/library.py)

| | construction | why the physics check (BDD) misses it |
|---|---|---|
| **A3** zone load redistribution | dP_total = 2/5/10% of total metered P moved from 4 metered single-phase loads to 4 others on the same phase; attacker re-solves the power flow and sets a = h(shifted) - h(clean) on every channel (`full`) or only on the 8 loads' meters (`partial`) | `full`: the delivered z is a physically valid operating point, so the residual stays small |
| **A4** kept-direction | a = H c with c in the estimator's TSVD-kept subspace, size 0.01-0.1 pu, re-synthesised every row from that row's clean estimate | linear stealth: r(z + Hc) = r(z) to first order |
| **A5** Newton-corrected A4 | A4 corrected with Newton steps so the full nonlinear model is consistent | nonlinear stealth: a = h(x + c) - h(x) |

Measured on fdia_v1 (val + test, 6 A3 windows): an A3 window changes the 8 targeted meters' P by
**30-580%** of their normal value (2% level: 30-160%). A3 is large per meter but consistent with
the physics. fdia_v1 has no A4/A5 windows.

## 1. Papers

| Paper (read) | System | Attack | Detector and key idea | Reported result |
|---|---|---|---|---|
| Pinceti, Sankar & Kosut, *Detection and Localization of Load Redistribution Attacks on Large Scale Systems*, arXiv 1912.09453 (2019/20) | Synthetic Texas 2000-bus | Intelligent LR attacks on a congested line, load shift 1-15%; random LR | Nearest-neighbour distance of the current **load vector** to historical load vectors, 35 load groups with per-group thresholds | Shift 10-15%: **80-100% detection at 0-3% FAR**; below 10% "does not perform well" |
| Kaviani & Hedman, *A Detection Mechanism against Load-Redistribution Attacks in Smart Grids*, arXiv 1907.13294 | 3-bus, IEEE 6-bus, Polish 2383-bus [S] | Optimal LR attack (net load unchanged, +/-alpha L_i, alpha 5-10%) aimed at overloading a target line | Counts "proper deviations" of estimated loads at buses sensitive (PTDF) to the target line; flag if > half of them deviate in the attack direction | 6-bus example correct; 2383-bus: 4000/4000 attacks flagged [S]. No FAR/AUC reported. Needs the attacker's target line to be known |
| Xu, Guo, Wen, Deng & Zhang, *Detecting FDIA in Smart Grids with Modeling Errors: A Deep Transfer Learning Approach*, arXiv 2104.06307 (2021) | IEEE 14 / 118 | **AC-stealthy** a = h(x+c) - h(x) (our A5 type), one bus, 10-30% intensity | DNN on raw P/Q/load measurements, pre-trained on simulated attacks, transferred to the real (mismatched) model with MMD | ACC 99.99-100% for line-parameter error 0-50%; BDD 50% (= blind). Baselines (LR/SVM/RF/DNN) 54-85% at 50% mismatch. **Needs labelled attacks for training** |
| Boyaci et al., *GNN Based Detection of Stealth FDIA in Smart Grids*, IEEE Systems J. 2021 (arXiv 2104.02012) | IEEE 14 / 118 / 300 | AC-stealthy optimisation attack on a BFS neighbourhood (up to ~30% of meters, <=2% \|V\| change) [S] | Chebyshev graph convolution on raw per-bus P, Q; single snapshot | DR 84 / 91 / 97%, FA 2.4 / 1.2 / 0.7%, F1 90 / 94 / 98% [S]; BDD cannot detect |
| Boyaci et al., *Joint Detection and Localization of Stealth FDIA using GNNs*, arXiv 2104.11846 (2021) | IEEE 57 / 118 / 300 | Optimisation (stealthy), replay, distribution and scaling attacks | ARMA graph filters on raw per-bus P, Q; per-bus outputs for localization | DR 99.1-99.97%, FA 0.14-0.28%, F1 99.4-99.9% [S]; localization 79-83% sample-wise [S] |
| Abdulin & Narimani, *Attention-Enhanced Graph Filtering for FDIA Detection and Localization*, arXiv 2601.18981 (2026) | IEEE 14 / 300, NYISO loads | AC-consistent attacks "bypass conventional BDD"; localized | ARMA graph filters + Transformer; idea: normal measurements vary **smoothly across the network**, attacks give abrupt local deviations | Results section not retrieved |
| Rajasekaran et al., *Detection of Stealthy FDIA in Unobservable Distribution Networks*, ISGT 2022 (from the A2 survey) | IEEE 37 (distribution) | Stealthy injection keeping power balance over a bus neighbourhood | Matrix completion + WLS + LNR, with **D-FACTS moving-target defence** (changes line reactances the attacker does not know) | 100% detection for perturbation 0.4-1.0; FAR <15% above 0.6 |

## 2. What the literature says, in one line each

- **A3/LR and A4/A5 share one weakness:** they are consistent with the *physics*, not with how
  the *loads normally behave*. Every detector that works uses the second: historical load
  vectors (Pinceti), sensitive-bus deviation patterns (Kaviani), learned measurement
  distributions (Xu), or spatial smoothness across neighbouring buses (Boyaci, Abdulin).
- **Residual/BDD alone cannot catch them**, by construction (Liu, Ning & Reiter 2009); this is
  a known limit, not a bug in our estimator.
- **The alternative that changes the grid** (moving-target defence, Rajasekaran) works but needs
  D-FACTS hardware and a modified estimator; out of scope.
- **Supervised detectors** (Xu; Boyaci) report 84-100% but are trained on labelled attacks of
  the same kind they are tested on, on transmission cases with 1% noise. They do not report how
  they do on attack types they were not trained on.

## 3. What is implementable here, and the plan

Our peer-prediction checks (reports/a2_detection.md) are an unsupervised version of the same
"consistency with normal load behaviour" idea: each meter is predicted from all other meters by
a model fitted on clean data only, so no attack labels are needed. This is closest to Pinceti
(learned normal load patterns) and Abdulin/Boyaci (spatial consistency), without training on
attacks.

1. **A3:** the P-peer check already catches 5/6 fdia_v1 A3 windows at 0.37% FAR (the miss is
   the weakest window: 2%, pulsed, 10 attacked steps). A larger A3 benchmark needs OpenDSS (the
   attacker re-solves the power flow), so it is generated on the Windows machine with
   `validation/gen_attack_bench.py`, using the library's own `attack()`.
2. **A4/A5:** generated the same way (the library re-synthesises the stealthy vector every row
   from the estimator's clean estimate). Test the V-peer and P-peer checks on them. The
   literature predicts that spatial-consistency checks can see stealthy attacks that BDD cannot.
3. **Compare like for like:** report detection against chance (the probability that a clean
   window of the same length alarms), as in reports/a2_detection.md, and against the existing
   AE-Mahalanobis detector.

## Sources
- https://arxiv.org/abs/1912.09453
- https://arxiv.org/pdf/1907.13294
- https://arxiv.org/html/2104.06307v3
- https://arxiv.org/abs/2104.02012
- https://arxiv.org/pdf/2104.11846
- https://arxiv.org/html/2601.18981
- https://www.osti.gov/biblio/1908546
