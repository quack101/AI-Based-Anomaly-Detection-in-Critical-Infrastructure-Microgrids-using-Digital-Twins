# Literature survey: detecting A2, A3 and A4 attacks

Date: 2026-11-03. Purpose: find papers that report detection numbers for attacks like our
A2 (single-meter bias), A3 (load redistribution) and A4/A5 (stealthy), so we can check
(a) whether our attack setups are comparable and (b) what detection rate is realistic.

Our current results (reports/detector_robustness.md, fusion BDD + AE-Mahalanobis, ~1% FAR,
5 seeds, fdia_v1 test split): A1 7-9 of 12 windows, **A2 0-1 of 3 (not robust)**, A3 1-2 of 3,
A4/A5 not in the test split.

Only papers whose text was actually read are summarised with numbers. Several publishers
(OSTI full text, NSF-PAR, Springer, ResearchGate) block automated reading; those are listed
at the end as "to read manually".

## 1. Summary table

| Attack | Paper (read) | Feeder | Attack setup | Detector | Reported result |
|---|---|---|---|---|---|
| A2-like | Zideh & Solanki, *Physics-Informed Convolutional Autoencoder for Cyber Anomaly Detection in Power Distribution Grids*, arXiv 2312.04758 (2023) | IEEE-13 and **IEEE-123**, OpenDSS, with PV/wind | **One channel**: voltage magnitude at bus 35 (123-bus); multiplicative bias, \|alpha\| <= **5%** of the true value | Conv. autoencoder + Kirchhoff P/Q residual in the loss; **per-channel** anomaly score | 123-bus: accuracy 99.80%, precision 100%, **recall 95.0%**, F1 97.44%. Plain ConvAE: recall 58.33%. Weak below ~3% magnitude (figure only). FAR not reported. |
| A3 | Pinceti, Sankar & Kosut, *Detection and Localization of Load Redistribution Attacks on Large Scale Systems*, arXiv 1912.09453 (2019/2020) | Synthetic Texas 2000-bus (transmission) | Intelligent LR attacks on a congested line, load shift 1-15%; random LR attacks | Nearest-neighbour distance to historical load vectors, 35 load groups with per-group thresholds | **Shift 10-15%: detection 80-100% at 0-3% FAR. Below 10%: "does not perform well."** |
| A4-like | Rajasekaran, Natarajan, Pahwa & Wu, *Detection of Stealthy FDIA in Unobservable Distribution Networks*, IEEE ISGT 2022 | IEEE-37 unbalanced | Stealthy injection keeping power balance over a bus and its neighbours (buses 19, 21, 22 attacked) | Matrix-completion imputation + WLS + **LNR test**, with D-FACTS devices that change line reactance unknown to the attacker (moving-target defence) | **100% detection** for D-FACTS perturbation 0.4-1.0 (100 Monte Carlo runs); FAR < 15% for factor > 0.6 |
| Context | Husnoo et al., *False Data Injection Threats in Active Distribution Systems: A Comprehensive Survey*, arXiv 2111.14251 | Survey | - | - | Distribution-system FDIA detection is "underexplored"; small single/few-meter injections (e.g. Lo & Ansari's CONSUMER attack) defeat several ML detectors; stealthy attacks bypass residual BDD by design. No performance table. |
| A2 method | Chen & Wang, *Quickest Attack Detection in Smart Grid Based on Sequential Monte Carlo Filtering*, arXiv 1712.08659 | Single 3-phase bus | Persistent bias with threshold 1.5 sigma; intermittent FDI | SMC estimator + **CUSUM** | Shortest detection delay vs chi-square and windowed LLR detectors; numbers only in figures |

## 2. What this means for us

### A2 (single-meter bias)
- **Closest benchmark: Zideh & Solanki.** Same feeder (IEEE-123, OpenDSS), one attacked
  channel, magnitude up to 5%. They get 95% recall with a **per-channel** score. Their plain
  autoencoder, with no physics term, gets only 58%.
- **Our setups differ in two ways that matter:**
  1. Our A2 voltage bias is 0.005-0.02 pu (about 0.5-2% of nominal), smaller than their 5%.
     Their text says detection is weak below about 3%, so **our A2 is harder than their
     benchmark**. Our P/Q bias (5-20%) is comparable or larger.
  2. They report per-timestep recall with no false-alarm rate. We report windows caught at
     a fixed ~1% FAR. The numbers are not directly comparable until we score the same way.
- **The likely reason we miss A2:** our Mahalanobis score aggregates all 165 channels, so one
  biased meter (1-3 channels) is diluted. Their per-channel score, the CUSUM literature, and
  the classical LNR test are all *single-channel* tests.
- **Plan:**
  1. Add a per-channel score: max over channels of the AE's whitened per-channel error. The
     AE already computes these, and localization is 77.8%, so the signal is there.
  2. Add a per-channel CUSUM on that error for slow, persistent bias (our A2 lasts 15-240
     minutes and is often a ramp).
  3. Fuse with BDD + Mahalanobis via the existing logistic fusion, threshold at 1% clean val.
  4. **Reproduction check:** add an A2 variant that matches Zideh & Solanki (one voltage
     channel, +/-5%) and report per-timestep recall, so we can say whether we reproduce their
     ~95%.

### A3 (load redistribution)
- Pinceti et al. reach 80-100% only for shifts of 10-15%; below 10% they "do not perform well."
- **Our A3 levels are 2%, 5% and 10%**, mostly below their reliable range, so 1-2 of 3 is in
  line with the literature, not a bug.
- Options: add a 15% level to match their range; or try their nearest-neighbour load-pattern
  check on our metered P/Q (cheap, no training).

### A4/A5 (stealthy)
- By construction a stealthy attack keeps residuals small, so any residual test (our BDD)
  misses it. The paper that reaches 100% does it by changing the grid (D-FACTS, a
  moving-target defence) so the attacker's model is wrong. That is hardware we don't model.
- Realistic options for us:
  (a) test whether the AE catches A4/A5 through temporal inconsistency (the attack is
      recomputed per timestep from the clean estimate);
  (b) report honestly that stealthy attacks are undetectable by the physics check, and that
      MTD is future work.
- A4/A5 are not in the fdia_v1 test split, so (a) needs a small new attacked set.

## 3. To read manually (blocked for automated fetch)
- *A False Data Injection Attack Detection Strategy for Unbalanced Distribution Networks State
  Estimation* (ResearchGate 367031620)
- *Detecting FDIA in Smart Grids: A Semi-Supervised Deep Learning Approach* (OSTI 1812900)
- *An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution
  Grids* (arXiv 2404.02923, rate-limited; retry)
- *Detection and Mitigation of Load Redistribution Attacks in Power Distribution Systems Using
  ML* (Springer, 2026)
- *Distributed Quickest Detection of Cyber-Attacks in Smart Grid* (NSF-PAR 10119500)
- Liu, Ning & Reiter (2009), *FDIA against State Estimation in Electric Power Grids*: the
  original stealthy-attack paper, cite for A4

## Sources
- https://arxiv.org/html/2312.04758v1
- https://arxiv.org/abs/1912.09453
- https://www.osti.gov/biblio/1908546
- https://arxiv.org/html/2111.14251v2
- https://arxiv.org/pdf/1712.08659
