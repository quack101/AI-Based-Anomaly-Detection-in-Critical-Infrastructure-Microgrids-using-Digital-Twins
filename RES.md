# FDI Detection on an IEEE-123 Digital Twin — Literature Synthesis and Architecture Review

**Scope of evidence.** 17 PDFs were originally supplied and read in full (§0.1); a further
14 papers from 2025–2026 were located by direct search and added in a second tranche (§0.1b),
giving **31 papers total**. All "no paper does X" claims in this document are scoped to those
31 and are **not** claims about the literature at large. Throughout, claims are tagged:
**[P]** = paper-supported, **[I]** = current implementation as described, **[R]** = engineering/research recommendation.

31 reviewed papers (17 + 14), plus Liu-Ning-Reiter (primary source, added later; read
only for Thm 3.1/3.2 and Sec. 2). Novelty claims are screened against the 31, not
against LNR.

**Companion documents.**

| File | Role |
|---|---|
| `RES.md` (this file) | Research and architecture authority. Defines *what must be true*. |
| `IDEAL_preprocessing.md` | Operational spec for the Phase-1 data rebuild, handed to the data teammate. Defines *how to produce it*. Subordinate to this file. |
| `CLAUDE.md` | Agent operating rules for implementation. |

If a companion document conflicts with RES.md, RES.md governs and the companion is corrected.

## 0. Evidence base and one important gap

### 0.1 Papers actually in the uploaded set

|Key|Paper|
|-|-|
|**Liang**|Liang, Zhao, Luo, Weller, Dong — *A Review of FDIAs Against Modern Power Systems*, IEEE TSG (2015/2017)|
|**Musleh**|Musleh, Chen, Dong — *A Survey on the Detection Algorithms for FDIAs in Smart Grids*, IEEE TSG 11(3), 2020|
|**Caleb**|Toro Dama Caleb, Shao, Kaabouch — *FDIAs on Smart Grids: Attack Models, Challenges and Future Directions*, Int. J. Inf. Secur. 25:94, 2026|
|**Irfan**|Irfan, Sadighian, Tanveer, Al-Naimi, Oligeri — *A survey on detection and localisation of FDIAs in smart grids*, IET CPS T&A 9(4), 2024|
|**Sayghe**|Sayghe, Hu, Zografopoulos, Liu, Dutta, Jin, Konstantinou — *Survey of ML methods for detecting FDIAs in power systems*, IET Smart Grid 3(5), 2020|
|**Youssef**|Youssef, Labeau — *FDIAs against State Estimation in Smart Grids: Challenges and Opportunities*, IEEE CCECE 2018|
|**Anwar**|Anwar, Mahmood — *Vulnerabilities of Smart Grid State Estimation against FDIA*, Springer GET ch. 17, 2014|
|**RahmanMR**|Rahman, Mohsenian-Rad — *FDIAs with Incomplete Information Against Smart Power Grids*, GLOBECOM 2012|
|**BiZhang**|Bi, Zhang — *Graphical Methods for Defense Against FDIAs on Power System State Estimation*, IEEE TSG 5(3), 2014|
|**Lei**|Lei, Pang, Wen, Hou, Han — *FDI Attack Detection at the Edge of Smart Grids Based on Classification of Predicted Residuals*, IEEE TII 18(12), 2022|
|**ZhangYang**|Zhang, Yang — *FDIA detection in dynamic power grid: a recurrent neural network-based method*, Front. Energy Res. 10, 2022|
|**Xue**|Xue, Jing, Liu — *Detection of FDIAs in Smart Grid Utilizing ELM-Based OCON Framework*, IEEE Access 7, 2019|
|**Xie**|Xie, Mo, Sinopoli — *False Data Injection Attacks in Electricity Markets*, SmartGridComm 2010|
|**Esmalifalak**|Esmalifalak, Shi, Han, Song — *Bad Data Injection Attack and Defense in Electricity Market Using Game Theory Study*, IEEE TSG 4(1), 2013|
|**QZhang**|Q. Zhang, Li, Shi, Tomsovic, Sun, Ren — *Profit-Oriented FDI on Energy Market: Reviews, Analyses, and Insights*|
|**Maleki**|Maleki, Pan, Lakshminarayana, Konstantinou — *Survey of Load-Altering Attacks Against Power Grids*, arXiv:2410.22007, 2024|
|**WangCPA**|X. Wang, Xue, Lu, Jiang, Bompard, Masera, Wu — *Coordinated cyber-physical attack on power grids based on malicious power dispatch*, IJEPES 155, 2024|


### 0.1b Second evidence tranche (2025–2026)

The original 17 papers skew 2009–2024 and missed a dense 2025–26 wave of work on
unbalanced three-phase distribution FDIA. Fourteen further papers were located by
direct search and read in full. Short codes used throughout:

|Code|Paper|
|-|-|
|Cui26|Cui, Zhang, Kang, Zhang, Gómez-Expósito — "Privacy-Preserving FDIA Detection for Radial Distribution Systems Considering Attacker Preference," IEEE TPWRS 41(1):270–282, Jan 2026|
|XuIoT26|Xu, Cao, Liu, Wu, Hu, Zhang, Wu, Yu — "A Multiarea Data Reconstruction Framework to Mitigate FDIA in IoT-Enabled Power Distribution Systems," IEEE IoT J. 13(14):31737–31751, 2026|
|Deh25|Dehbozorgi, Rastegar, Arani — "FDIA Detection and Localization Framework in Power Distribution Systems Using a Novel Ensemble of CNNs and XAI," IEEE TIA 61(3):4801–4811, 2025|
|Rah26|M. M. Rahman, A. Rahman, W. Sun — "Cyber Resilience for DER-Rich Distribution Grids: Ensemble Detection and Explainable Localization of Unobservable FDIAs," IEEE Access 14:92116–92134, 2026 (CC BY 4.0)|
|Zid|Jabbari Zideh, Khushalani Solanki — "Multivariate Physics-Informed Convolutional Autoencoder for Anomaly Detection in Power Distribution Systems with Widespread DER" (preprint; venue/year unconfirmed, refs to 2024)|
|Sha25|Sharma, Chakrabarti, Sharma — "Angle Reference Based Three-Phase State Estimation of Power Distribution Systems," IEEE TII 21(12):9515–9525, Dec 2025|
|XuUD26|Xu, Wu, Hu, Xu, Han, Ma — "A Unified Framework for Numerically-Stable State Estimation and FDIA Detection in Distribution Networks," IEEE TII 22(2):1173–1181, Feb 2026|
|XuDS25|J. Xu, Zhang, Lin, Jiang, Ruan, Wei, Hu, Yu — "A Distributed Secure State Estimation Framework for Unbalanced Active Distribution Systems," IEEE TSG 16(5):3714–3727, Sep 2025|
|TAPINE|Feng et al. — "TAPINE: Topology-Adaptive Physics-Informed Neural Estimator for Fast DSSE," IEEE TSG 2026 (early access)|
|BMAML|Wang, Cheng, Liu, Wang, Shi, Yu — "Topology-Aware DSSE Using Bayesian Model-Agnostic Meta-Learning," IEEE TII, accepted May 2026|
|DKL25|Duan et al. — "Topology-Aware Deep Kernel Learning for DSSE," IEEE ETFG 2025|
|Ald25|Alduhaymi, Singh, Ul Nazir, Pal, Ahmadi — "Forecasting-Aided State Estimation With Deep Learning-Generated Pseudo Measurements," IEEE TIM 74, art. 2519612, 2025|
|TF25|IEEE Task Force (Zografopoulos et al.) — "Cyber-Physical Interdependence for Power System Operation and Control," IEEE TSG 16(3):2554–2573, May 2025|
|Rin25|Rinaldi, Della Giustina, Vasile — "A Digital Twin Framework Exploiting Smart Meters Data for LV Fault Validation: A Field Case Study," IEEE ISGT Europe 2025|

**Author-independence note:** XuUD26 and XuDS25 share authors (Zaijun Wu / Qinran Hu /
Junjun Xu, Southeast University / Nanjing Normal) and XuUD26 cites XuDS25 — one research
programme, not two independent data points. XuIoT26 builds directly on Cui26's stealth
proof. Rah26 and Deh25 are independent groups (UCF; and the Deh25 method is a benchmarked
comparator in Rah26 Table 5).

**Verification caveat:** citations in this tranche were transcribed from the uploaded
PDFs, not from a verified bibliographic database. Confirm each before final submission.


### 0.2 The three "foundational" papers are **not** in the zip

You listed Liu–Ning–Reiter (CCS'09 / TISSEC 2011), Mohammad (arXiv:1809.07039), and Ahmed & Kim as foundational. Mohammad and Ahmed & Kim are still not in hand. **Liu–Ning–Reiter's PDF is now in hand** and is added below as a directly cited primary source:

> Y. Liu, P. Ning, M. K. Reiter, "False Data Injection Attacks against State Estimation in Electric Power Grids," *ACM Trans. Inf. Syst. Secur.* 14(1), Art. 13, May 2011, DOI 10.1145/1952982.1952995. (Journal extension of the CCS'09 paper.)

* **Liu, Ning & Reiter** is heavily and consistently described *inside* the uploaded set (Liang ref [10]; Sayghe ref [12]; Youssef refs [6],[7]; RahmanMR ref [3]; BiZhang ref [3]) — those secondary attributions are kept below alongside the primary cite, not replaced by it. **Primary cite, now verifiable directly:** Theorem 3.1 (p. 13:9, eq. 4) for the `a = Hc` stealth result; §3.2 / Theorem 3.2 for the limited-meter (`k ≥ m − n + 1`) case. Wherever a claim below was previously marked "as reported by" the uploaded secondary sources, the primary citation is now added alongside it.
* **Mohammad (arXiv:1809.07039)** is not cited by name in any uploaded paper I can match. **I make no claims from it.** You must read it yourself; do not let me stand in for it.
* **Ahmed & Kim** — the correct citation appears to be Ahmed, M. & Pathan, A.-S. K., *FDIA: An Overview and New Metrics for Fair Evaluation of its Countermeasures*, cited as ref [46] in **Caleb**. Caleb characterises it as criticising the absence of standard datasets and proposing new evaluation metrics for countermeasures in consumer-facing domains. That is the only thing I can responsibly say about it. Its actual metrics are directly relevant to your Deliverable-9 evaluation design and you should obtain it.

### 0.3 Disagreements found in the set (reported, not smoothed over)

1. **Is the perfect-knowledge attacker realistic?** RahmanMR argues a realistic FDIA is *necessarily* an incomplete-information attack (tap-changer positions, breaker states, admittances drift with weather). Caleb §4.1 agrees and calls perfect knowledge a limitation of much of the literature. Liang §III-A and Sayghe §3.1.1 nonetheless present the full-knowledge case as the primary construction. **Resolution [R]:** treat full knowledge as an upper bound on attacker capability, not as the modelling default.
2. **Can imperfect-knowledge attacks still be perfectly stealthy?** RahmanMR Theorem 1 says yes — knowing admittances on *one cut* of the network suffices for δc = 0. Youssef §III-A, reporting Teixeira et al., says undetectable attack magnitude is proportional to the accuracy of the attacker's model, implying a graded rather than binary outcome. These are compatible but framed very differently; the cut-based result is the stronger claim.
3. **Which detector family wins?** Musleh finds model-based ≈ 52.3% and data-driven ≈ 47.7% of reviewed algorithms with no clear winner, and explicitly warns that reported detection rates are not comparable across papers because system, attack type, and thresholds differ. Sayghe §6 concludes supervised and deep learning achieve the highest rates, but also states that declaring an optimal detector is infeasible for the same reason. **Do not cite either as "the literature recommends X."**
4. **Does OpenDSS-style multi-phase modelling belong in SE?** Anwar §17.3 and Sayghe §2.3 both insist on three-phase modelling for distribution. Musleh's statistics show only 3.5% of the reviewed detection literature actually targets distribution systems — so the requirement is asserted far more often than it is met.
5. Is the AC-stealth construction settled or still open? Cui26 (eq. 9–14, algebraic
proof at eq. 13) and XuIoT26 (eq. 11–15, phase-resolved, citing Cui26 for the proof)
both establish residual-invariance of `a = h(x̂+b) − h(x̂)` for full nonlinear
three-phase unbalanced AC WLS — the construction is settled and proven for the
*continuous nonlinear* case. Rah26 (eq. 5–7) instead constructs `a = Hc` against a
per-operating-point *linearised* `H` and verifies stealth against the same
linearisation (Fig. 1) — an internal-consistency check, not a nonlinear-AC test.
**These are not the same claim**, and the gap between them is precisely where A4's
negative control lives: nobody in the 32-paper set re-evaluates a linearly-constructed
`a` against a fully re-linearised, iteratively-converged nonlinear `h_abc(·)`.

---
# DELIVERABLE 1 — Cross-paper literature synthesis

## 1.0 Evidence matrix

|Paper|Grid/system|FDI definition|Attack surface|Modified measurements|State estimator|AC/DC|Detection method|Physical impacts|Dataset/simulation|Key limitation|
|-|-|-|-|-|-|-|-|-|-|-|
|**Liang**|Transmission (generic modern power system)|`a` added to `z`; stealthy iff `a = Hc`; *generalized* FDIA iff `J(x̂_bad) < C` even when `a ≠ Hc`|Meter measurements + breaker/switch status|Bus V, bus P/Q injections, branch P/Q flows; discrete breaker status in topology attacks|WLS (DC closed form; AC iterative)|Both; §III-D explicitly AC|Reviews LNR, `J(x̂)` χ², protection of basic measurement set, PMU placement|Load shedding, line overload, physical damage (via LR attack); LMP manipulation|Review — no new experiments|Review only; states AC-model FDIA results were then few and mainly *generalized* FDIAs|
|**Musleh**|Smart grid, mostly transmission (77.2%); distribution only 3.5%|`z^α_t = z_t + α_t`; stealth iff `α_t = h(c_t)`|Physical (RTU firmware), communication, network, cyber/application layers|V magnitude, V angle, power flow, current|WLS, MF, Kriging, ML, MMSE, KF/DKF/EKF, UIO, VAR|Both; explicitly quasi-static vs dynamic|Taxonomy: model-based (L2, LNR, χ², CUSUM, KLD, GLRT) vs data-driven (SVM, ANN/RNN/CNN/AE/ELM, KNN, DT/RF/GB, KMC, IF, DBN, HMM, RL)|Generation rescheduling, load shedding, AGC frequency deviation, blackout propagation|Review; tabulates test systems|79.5% of reviewed work tests only *basic* FDIA; only 10.2% does real-time testing|
|**Caleb**|Whole CPPS: steady-state, transient/aux, substation, energy&load, AI-based|Same WLS/`a=Hc` core; AC case defined by `J(x̂_bad) < τ`|**29 attack targets** across 5 domains incl. AMI, DER, V2G, IEC 61850, DNP3, **digital twins**|Meter/PMU/RTU data, breaker status bits, load data `P_D`, price signals, EV SoC, DT synchronisation links, ML training data|SSE, DSE, and AI/physics-informed estimators|Both; §4.4 argues DC assumptions are a resilience blind spot|Surveys BDD, AI detectors, and adversarial attacks *against* those detectors|Line overload, false tripping, cascading failure, blackout, market loss, DT desynchronisation|Review (2026)|Review; asserts distribution/unbalanced FDIA formulations are largely missing|
|**Irfan**|Smart grid, joint detection **and localisation**|Standard `z_a = z + a` bypassing BDD|Physical, communication, network, process layers|Not enumerated per-variable; framed by data source (PMU, smart meter, FDR, relays)|Data-driven and mathematical-model approaches|Splits reviewed papers into AC-system and DC-system groups (Table 15)|Localisation as multi-label classification / multi-cluster clustering; GSP, sparse group lasso, χ², matrix-based, observer residuals|Blackouts, disruption, islanding|Mostly MATPOWER-generated; IEEE-14 and IEEE-118 most common; NYISO load profiles|Very limited ground-truth datasets; no black-box (zero-knowledge) papers found|
|**Sayghe**|TS **and** DS explicitly contrasted|`z_a = z + a`, `a = Hc` (DC); `a = h(x̂_a) − h(x̂)` (AC)|Sensor compromise (intrusive) or channel spoofing (non-intrusive)|V magnitude, V angle, P/Q injections, P/Q flows, current magnitudes; DS adds branch currents & line-to-line V|WLS (AC via Newton–Raphson), DC, DSE (UKF); DSSE for unbalanced 3-φ|Both, plus explicit 3-φ unbalanced DSSE equations (14)–(27)|ML survey: SVM, DNN, RNN, CNN, ELM, ensemble|Not primary focus; tabulates LR attacks, energy routing, market|Review; notes IEEE-14/30/118 dominate|Reports ML detection is dominated by TS work; DS detection called an open direction|
|**Youssef**|Transmission SSE|`z_a = z + a`; `x̂_FDI = x̂ + Ma`, `r_FDI = r + (I − HM)a`; perfect stealth iff `a = Hc`|SCADA meter readings + breaker/switch status|Real/reactive injection, power flow, breaker/switch status (topology attacks)|AC SSE and DC SSE (Schweppe/Rom lineage), WLS|Both; frames AC SSE as the true model, DC as approximation|Statistical anomaly (GLRT, AR forecasting), ML (SVM+PCA), graph theory (CCT / Markov random field)|LR attacks, replay attacks, topology attacks masking outages|Survey — no new experiments|5-page survey; GLRT assumes known number of compromised meters; graph detectors assume trusted topology|
|**Anwar**|**Distribution / smart grid SE** (best fit to your problem)|Standard DC FDIA case study on 3-bus system|AMI/smart meters and SCADA measurement data|V magnitude & angle as states; measures P/Q injections, P/Q flows, **line current magnitude** eq. (17.7)|WLS, Gauss-Newton/Newton-Raphson, PSO variants|AC formulation given (17.3–17.7), DC used for the worked example|LNR, `J(x̂)` performance index, hypothesis testing|Operational failure, loss of synchronisation, blackout|3-bus worked example; references OpenDSS as a comparison tool for DSSE|Case study is DC and 3-bus; distribution claims are qualitative|
|**RahmanMR**|Transmission|`a = H̄c = Hc + δc`; perfect attack iff `δc = 0`|Sensors and PMUs|Bus injections and both-direction line flows (assumed measurement set)|WLS, `x̂ = (HᵀWH)⁻¹HᵀWz`|DC|Residue test `J(x̂_a) ≤ C`; proposes a topology **vulnerability measure**|Not the focus — focus is detectability|Analytical + simulation|DC only; assumes bus-injection + both-end line-flow measurement set that distribution feeders do not have|
|**BiZhang**|Transmission (IEEE test cases)|Undetectable FDIA on SE|Meter measurements|Meter measurements protected as a set|WLS SE|DC|**Defense, not detection**: optimal meter protection as a variant Steiner-tree problem; exact + pruning approximation|Cites regional blackout and price manipulation as motivation|IEEE standard testcases|Protection-set size can equal the number of states — costly; DC graph model|
|**Lei**|Microgrid + smart-meter testbed (real hardware)|`z_a_t = f(x_t) + e_t + a_t`; stealth `a_t = Fc_t` gives `Γ_a_t = Γ_t`|Terminal instruments (meters, gateways, RTUs), uplink channel, control centre|Aggregated measurement flows: generation, load, energy consumption|WLS referenced; **detector does not require SE**|Nonlinear `f(·)` (AC-consistent)|**GRU predicts next measurement → predicted residual `Γp_t = ‖z_t − y_t‖` → wavelet-CNN classifies the residual sequence**|Not modelled; focus is detection at the edge|Self-collected microgrid testbed (60 meters, 5-min interval) **and** an open Austin, TX dataset|Attack magnitudes tested are very small (η ≈ 0.5–2.5‰); no feeder physics or SE consequence modelled|
|**ZhangYang**|Transmission, IEEE standard bus systems (e.g. IEEE-30)|`z'_t,i = z_t,i + a_t,i`; safe attack interval derived from the normalised-residual bound|Measurement data|Phase angles and voltage magnitudes as states|**Kalman filter (DSE)**|Linear `H` (DC-style measurement matrix inside a KF)|Normalised residual → observation `o_t = ‖z_t − Hx̂_t‖/‖ω‖` → **LSTM over m past observations → DNN binary classifier**|Not modelled|IEEE standard bus systems, synthetic attacks|Linear `H` and KF; balanced transmission; no AC nonlinearity|
|**Xue**|Transmission, IEEE 14-bus|Bus-based FDIA on AC SE|Meter measurements|`P_km`, `Q_km` line flows and `P_k`, `Q_k` bus injections|AC SE|**AC** (argues AC is the practical case)|ELM-based one-class-one-network (OCON): per-bus subnets + global layer; adds a **prediction-based recovery** step using spatial correlation|Not modelled|IEEE 14-bus + real NYISO load data|14-bus, balanced, transmission; one subnet per bus does not scale to a feeder|
|**Xie**|Deregulated market over transmission SE|FDIA on SE that propagates to ex-post LMP|SCADA measurements feeding SE|Measurements used by SE for ex-post LMP|SE feeding DC-OPF/LMP|DC|None (attack paper)|**Financial**: profitable virtual bidding between two nodes|IEEE 14-bus|Market-layer; not applicable to a distribution feeder twin|
|**Esmalifalak**|PJM 5-bus market|Bad-data injection to move prices|Individual measurements|Per-measurement price sensitivity|SE|DC|Zero-sum attacker/defender game over which measurements to attack/defend|Price change in a chosen direction|PJM 5-bus|Market-layer, 5-bus, DC|
|**QZhang**|Electricity market|Profit-oriented FDIA on SE→LMP|SCADA raw measurements from RTUs at substations and generators|Power flow, power injection, bus voltage|SE feeding RT market clearing|DC-oriented (`H` = linearised `h(x)`)|Review of market-attack detection|Financial arbitrage; UTC/virtual bidding|Review|Market scope|
|**Maleki**|Transmission **and** distribution; IoT/DER edge|**LAA, not FDIA**: direct manipulation of load rather than of measurements|High-wattage IoT loads, EV chargers, smart inverters, DER firmware|The *physical load itself* — explicitly distinguished from LRA which falsifies measurements|Model-based observers/SMO, PINN|Both; distribution impact treated as voltage rather than frequency|Model-based (residual/observer), data-driven (FFT/spectral, PINN, smart-meter analytics), hybrid|Voltage violations in distribution; frequency instability; market distortion|Review + CVE table|Different threat class — do not conflate with FDIA (see 1.B)|
|**WangCPA**|Transmission, IEEE-118 / IEEE-300|**False *command* injection**, not false measurement injection|Dispatch commands; coordinated with physical line disconnection|Dispatch setpoints (`P` of units), plus physical line outages|—|AC power-flow-based vulnerability index|None (attack paper)|Cascading failure via inverse-community boundary lines|IEEE-118, IEEE-300|Command-injection + physical attack; outside a measurement-integrity threat model|

## 1A. Formal FDI definition

**Measurement vector `z`.** `z = (z₁,…,z_m)ᵀ` is the vector of telemetry actually reported to the control centre. In transmission SCADA it comprises bus voltage magnitudes, bus real and reactive power injections, and branch real and reactive power flows [P: Liang §II-A1]. Musleh's more general listing adds voltage angle and current [P: Musleh §III-A]. Anwar adds line current *magnitude*, `|I_ij| = √(P_ij² + Q_ij²)/V_i`, as an explicitly measured quantity in distribution practice [P: Anwar eq. 17.7].

**State vector `x`.** The states are the complex nodal voltages: magnitudes and phase angles at every bus [P: Musleh §III-A; Sayghe eq. (1)]. For an `n`-bus single-phase-equivalent system, `x = [δ₂ … δ_n, V₁ … V_n]ᵀ` with dimension `(2n − 1)`, because the reference-bus angle is fixed at 0 rad [P: Anwar eq. 17.1; Sayghe eq. (1)].

**Measurement model.** AC: `z = h(x) + e`, where `h(·)` is the nonlinear map determined by grid structure and line parameters, and `e` is zero-mean Gaussian noise [P: Liang eq. (1); Musleh eq. (1); Sayghe §2.2.1]. The component equations are the standard injection and flow relations, e.g. `P_i = V_i Σ_j V_j(G_ij cos δ_ij + B_ij sin δ_ij)` [P: Sayghe eqs. (2)–(3); Anwar eqs. 17.3–17.6]. DC: `z = Hx + e` with `H` constant, obtained by assuming line resistance negligible against reactance, small angle differences, and `V ≈ 1 p.u.` [P: Liang eq. (3); Sayghe §2.2.2; Youssef §II].

**Estimation.** WLS: `x̂ = argmin (z − h(x))ᵀW(z − h(x))`, `W = diag(σ_i⁻²)`, solved iteratively by Gauss-Newton/Newton-Raphson in the AC case and in closed form `x̂ = (HᵀWH)⁻¹HᵀWz` in the DC case [P: Liang eqs. (2),(5); Sayghe eqs. (5)–(13); Anwar §17.2.3]. Redundancy `m ≥ n` is required [P: Youssef §II].

**Attack vector.** `z_bad = z + a`, `a ≠ 0`; the induced state deviation is `c` such that `x̂_bad = x̂ + c` [P: Liang §II-B; Caleb eqs. (1)–(4); RahmanMR eqs. (5)–(6)].

**Stealth conditions.**

* *DC, perfect:* if `a = Hc`, the residual is unchanged, `‖z_bad − Hx̂_bad‖ = ‖z − Hx̂‖`, so the attack passes residual BDD for any `c` [P: Liang eq. (9); Caleb eqs. (5)–(8); Sayghe eq. (36); RahmanMR eq. (5); Liu, Ning & Reiter (2011) Theorem 3.1, eq. (4), p. 13:9; reported by all of these as the Liu–Ning–Reiter result]. **LNR Theorem 3.1 is proven for the DC (linear) model only. AC extension: Rah26 §II.C, Cui26.**
* *DC, generalized:* if `a ≠ Hc` but `J(x̂_bad) < C`, the attack still evades the χ² threshold by hiding inside tolerated measurement error [P: Liang §II-B].
* *AC:* if `a = h(x̂_bad) − h(x̂)`, the residual is likewise unchanged [P: Sayghe eq. (37); Liang eq. (15), attributed to Rahman & Mohsenian-Rad 2013]. Liang eq. (16), attributed to Hug & Giampapa, gives the partitioned form `a₂ = h₂(x̂₁, x̂₂+c) − h₂(x̂₁, x̂₂)` in which only a sub-block of measurements is altered.
* *Incomplete information:* with `H̄ = H + δ`, `a = Hc + δc`; the attack is *perfect* iff `δc = 0` and *imperfect* otherwise [P: RahmanMR §III; Liang eqs. (11)–(13)]. RahmanMR Theorem 1: knowing the admittances of all lines forming **one cut** of the network suffices to guarantee `δc = 0`, provided `c` is constant within each of the two resulting islands.
* *Dynamic/DSE:* stealth is defined relative to the dynamic detector, requiring `‖a(k+1) + h(x̂(k+1)) − h(x̂(k+1))‖ ≤ ε` and bias accumulation across steps [P: Caleb §3.1.2]. ZhangYang derives the equivalent admissible interval `H_i x̂⁻ − V_t,i τ − z_t,i ≤ a_t,i ≤ V_t,i τ + H_i x̂⁻_t,i − z_t,i` [P: ZhangYang eq. (11)].

**Residual / bad-data detection.** `r = z − h(x̂)`. Three standard tests: (i) L2 norm `‖r‖₂ > τ`; (ii) largest normalised residual `Δ_k = r_k/√Σ_kk` compared against a threshold; (iii) χ² performance index `J(x̂) < C` [P: Youssef eqs. (4),(6); Musleh eqs. (18)–(20); Sayghe eqs. (28)–(31)]. Musleh adds CUSUM, KLD, and GLRT as further model-based tests [P: Musleh eqs. (21)–(22)]. Sayghe §2.4 gives the full iterative LNR removal loop. All sources agree these tests fail against stealth FDIA by construction.

**Attack classes.**

* *Random / basic:* attacker perturbs whatever measurements are reachable, aiming to corrupt SE without a specific target state; detectable by residual tests [P: Musleh §III-A "basic FDIA"; Sayghe §6; Irfan Table 16].
* *Targeted:* attacker injects a specified error into specified states [P: Sayghe §6].
* *Sparse:* minimise the number of compromised meters, `α_k = min ‖Hc‖₀` subject to a target-measurement constraint; NP-hard in general, solved by MILP, matching pursuit, LASSO, or a min-cut relaxation [P: Liang eq. (10) and §III-A; Sayghe eq. (38)]. Liang, reporting Liu et al., states the attacker needs `k ≥ m − n + 1` compromised meters for an attack vector to be guaranteed to exist [Liu, Ning & Reiter (2011) §3.2, Theorem 3.2].
* *Stealthy / model-aware:* satisfies the stealth condition above, undetectable by residual BDD [P: all].

**DC-FDIA vs AC-FDIA.** DC-FDIA has a closed-form stealth condition and a constant `H`, making construction and analysis tractable [P: Liang; Sayghe §2.2.2]. AC-FDIA has no closed-form mapping between `a` and `x̂` because the estimate is obtained iteratively; Caleb states this directly [P: Caleb §3.1.1]. Liang notes that convergence of Gauss-Newton under the AC attack construction is not guaranteed and the fixed point is not obtainable analytically [P: Liang §III-D]. Three sources make the same critical point in different words: **a DC-constructed attack fails against an AC estimator** — Liang states that implementing DC-model FDIAs in an AC system is impractical and that such attacks are easily detected when the operator runs a nonlinear estimator [P: Liang §III-D, reporting Rahman & Mohsenian-Rad]; Caleb §4.4 repeats this and calls DC-only threat modelling a resilience blind spot; Caleb §2 further reports (its ref [8]) that attack strategies and metrics effective in DC-modelled transmission often fail or behave differently in AC-modelled distribution. **This single point is the strongest literature-based justification for your AC/OpenDSS choice [R].**

## 1B. What attackers actually modify (per-variable verdict)

Verdicts below distinguish *"the paper treats this as a telemetry variable exposed to the attacker"* from *"the paper merely mentions the quantity."*

|Candidate target|Verdict|Evidence|
|-|-|-|
|**P/Q bus injections**|**YES — primary target, unanimous**|Liang §II-A1 (meter measurements include bus real/reactive injections); Xue §II (`P_k`, `Q_k` bus injections in the measurement vector); Sayghe §2.2.1; Anwar §17.2.1|
|**P/Q branch flows**|**YES — primary target, unanimous**|Liang §II-A1; Xue (`P_km`, `Q_km`); RahmanMR §II-A (measurement set = bus injections plus both-direction line flows); Sayghe|
|**Voltage magnitude**|**YES**|Musleh §III-A lists voltage magnitude in `z`; Liang §II-A1; Youssef §III (a slow scaling manipulation of voltage magnitudes across a grid section is given as the example of a stealth FDIA); Caleb §3.4.1 (V among smart-meter recorded quantities)|
|**Voltage angle**|**YES, but conditionally**|Musleh §III-A lists it in `z`. Youssef §II is precise: angle is a *measured* quantity at generator-type buses and an *estimated* quantity at load buses. Sayghe §2.2.1 lists angles among measurements. **[R]** In a distribution feeder without PMUs, angle is estimated, not measured — do not model it as an attack surface unless you place a PMU.|
|**Current / phasor measurements**|**YES, where the sensor exists**|Anwar eq. (17.7) gives line current magnitude as a measurement function; Anwar §17.4 notes PMUs enable current measurement at PMU nodes; Sayghe §2.3.1 eq. (23) makes branch currents `I_abc,ij` explicit for branch-current-based DSSE; Caleb §3.4.1 lists current among smart-meter quantities|
|**Load measurements (AMI/consumption)**|**YES — and this is *your* surface**|Caleb §3.4.1 models deductive theft as `P_i(t) = P_i(t)^act − d_t`, `d_t ∈ [d_min, d_max]`, and additive attacks as over-reporting; Caleb also reports a multi-parameter attack manipulating `P`, `I`, and `S` **simultaneously while preserving `S_i(t) = V_i(t)·I_i(t)`** to degrade detector performance (Caleb ref [204]); Liang §VI(3) explicitly calls for more FDIA work on the distribution and user side, noting user-side meters can be manipulated|
|**PMU measurements**|**YES**|RahmanMR abstract (sensors *and* PMUs); Liang §V-B (PMUs are harder to compromise but GPS time-stamp spoofing works around this, reporting Gong et al.); Caleb §3.3 lists PMUs as a substation-control target|
|**Topology / breaker / switch status**|**YES, as a separate class**|Youssef §III-B (topology attacks target the breaker/switch status feeding topology estimation, i.e. `H` itself); Liang §III-C; Caleb Table 2 ("topology poisoning: altering breaker status bits to modify `H`"); Musleh §II-B1 (attack on breaker statuses producing losses up to ~$100k in 24 h, reporting its ref [32])|
|**Equipment ratings**|**NO**|No uploaded paper models ratings as an attacker-modifiable telemetry variable. Ratings appear only as *constraints* in SCOPF/SCED formulations [Caleb Table 2]. **Do not model rating tampering as FDIA.**|
|**Transformer tap position**|**NO (as a target); YES (as attacker uncertainty)**|RahmanMR §II-C treats tap position as something the *attacker* does not know, contributing to `δ`. It is not modelled as an attacker-modified measurement.|
|**Dispatch/control commands**|**Separate threat class**|WangCPA is false *command* injection combined with physical line disconnection; Caleb §3.4.3 models falsified DER setpoints. **[R]** Out of scope for a measurement-integrity study — say so explicitly rather than silently importing it.|
|**The physical load itself**|**Separate threat class (LAA)**|Maleki draws the distinction explicitly: LAAs manipulate the load *directly* through IoT-enabled appliances, whereas load-redistribution attacks falsify *measurements*. **[R]** Your project is a measurement-integrity project. Do not blur into LAA.|

**Consequence for you [R].** OpenDSS computes bus voltages, branch currents, line flows, losses, and transformer loadings. **None of these is a measurement until you instantiate a sensor object that samples it, adds noise, and publishes it to Kafka.** Rule 8 of your own evidence discipline is exactly right and the literature backs it: `h(x)` in every paper above maps states to the outputs of *modelled instruments*, not to every quantity the solver can print.

## 1C. Downstream impact chain

```
attacked measurement  →  biased state estimate  →  biased derived quantities  →  operational consequence
```

|Stage|Supported?|Evidence|
|-|-|-|
|**Attacked measurement → state estimate**|Yes, exactly|`x̂_bad = x̂ + (HᵀWH)⁻¹HᵀWa = x̂ + c` [Liang eq. (8); Caleb eqs. (1)–(4); Youssef eq. (5) `x̂_FDI = x̂ + Ma`]|
|**Bus voltage / angle error**|Yes|The states *are* voltages and angles, so bias enters directly [Liang abstract; Musleh §III-A]. BiZhang reports an experiment producing a bias exceeding 50% of nominal without a BDD alarm (its ref [4])|
|**Line current, line P/Q/S flows**|Yes, but indirectly and rarely quantified|Derived from estimated states via `h(·)` [Anwar eqs. 17.5–17.7]. **No uploaded paper quantifies the induced error in derived line flows for a distribution feeder.** This is a genuine gap, and a legitimate contribution of your twin.|
|**Line congestion / overload**|Yes, experimentally in the LR-attack literature|Yuan et al.'s delayed LR attack causes line overloading unnoticed by the control centre, leading to physical damage [Liang §IV-B, reporting Yuan et al.]; Caleb Table 2 (SCED attacks may induce physical line overloads)|
|**Transformer loading**|**Not supported**|No uploaded paper treats transformer loading as an FDIA impact metric. **[R]** You may compute it from the twin, but you cannot cite the literature for it as an established impact.|
|**System losses**|**Not supported as an FDIA impact**|Losses appear as an *LAA/energy-routing* objective (Caleb §3.4.2: false link-state attacks increase transmission cost; Caleb §3.4.1: maximising power losses as an AMI attacker objective) but not as a validated FDIA-on-SE consequence metric. Treat as **[R]**, not **[P]**.|
|**Generation dispatch**|Yes|LR attacks target SCED and drive uneconomic redispatch [Liang §IV-B]; Caleb Table 2 SCOPF/SCED rows; Musleh §II-B2 (unnecessary generation rescheduling and load shedding)|
|**Voltage / security violations**|Yes|Caleb §3.1.4 (SCOPF compromise → N-1 non-compliant operation); Musleh §II-B2|
|**Physical instability**|Yes, mainly for AGC/frequency, i.e. transmission|Musleh §II-B2 (AGC frequency deviation; GPS spoofing → out-of-step load shedding); Caleb §3.2. **[R]** Not directly transferable to a radial 4.16 kV feeder twin.|
|**Cascading consequences**|Yes|Caleb §1 and §3.1.4; WangCPA demonstrates cascading failure on IEEE-118/300 via coordinated command injection + line disconnection|
|**Market consequences**|Yes, extensively|Xie (profitable virtual bidding between two nodes via ex-post LMP manipulation, IEEE 14-bus); Liang eq. (17) profit formulation; Esmalifalak (per-measurement price sensitivity, PJM 5-bus, zero-sum game); QZhang (review of profit-oriented FDIA). **[R] Out of scope** — a distribution feeder digital twin has no LMP.|

**Distribution-specific impact, which is what you actually need.** The uploaded set is thin here and honest about it. Maleki is the closest: it states that in distribution systems load changes manifest as **voltage** problems rather than frequency problems, and that leaf nodes of a distribution network are more vulnerable (Maleki, reporting its ref [68]). Caleb §4.2 explicitly calls for research on the **vertical propagation** of falsification from the residential/aggregator level upward. **[R] Your feeder twin is well positioned to produce exactly this missing quantification: attacked meter → biased state → voltage-magnitude error per phase-node → ANSI C84.1 band violation, plus line-loading error.** That is a real contribution, not a repackaging.

---
# DELIVERABLE 2 — What each component actually is

Your stated confusion is the right confusion to have: five of these eight things live in different ontological categories, and conflating them is the single most common way this class of project becomes unpublishable.

## 2.1 The three-layer ontology

```
LAYER 1 — PHYSICAL TRUTH (exists in reality; in simulation, it is what OpenDSS computes)
   Ground-truth phase-node voltages, branch currents, flows, losses, tap positions.
   In a real feeder: unknowable in full. In your twin: exactly known, by construction.

LAYER 2 — CYBER MEASUREMENT (a sampled, quantised, noisy, *transmitted* view of Layer 1)
   Only exists where you instantiate a sensor. Has a location, a sampled quantity,
   a noise model σ, a sampling period, and a transport (Kafka topic).
   THIS IS THE ONLY LAYER AN ATTACKER CAN TOUCH.

LAYER 3 — INFERENCE (what the operator believes)
   AC state estimate x̂, residual r, forecast ŷ, anomaly score, detection decision.
   Corrupted only *via* Layer 2.
```

Every uploaded paper's `z` lives in Layer 2 and every `x` lives in Layer 3, estimating Layer 1 [P: Liang eq. (1); Sayghe §2.1; Musleh §III-A]. Anwar §17.2 states the purpose of SE plainly: it estimates unmeasured variables and filters measurement error, i.e. it exists precisely because Layer 1 is not directly observable.

## 2.2 Component-by-component

**1. Smart-meter measurements.** Layer 2. A meter at a load node samples that node's `P`, `Q`, and typically `|V|` and `|I|`, and reports them over AMI [P: Caleb §3.4.1 — smart meters record time-based voltage, current, reactive power, and kWh, sometimes at minute or second resolution]. This is your **primary attack surface** and the *only* one you currently have. **[I]** Your 49 meters are the AMI layer.

**2. OpenDSS power flow.** Layer 1 *generator*. Given the full network model, all parameters, and all load inputs, it produces the complete physical solution by solving the network equations. It is not an estimator: it has no measurement noise, no redundancy, no residual, and no notion of what is or is not observed. **It answers "what happens?", not "what do we think is happening?"**

**3. AC state estimation.** Layer 3. Given a *finite, noisy, incomplete* measurement set plus the network model, it infers `x̂` by minimising `Σ w_i (z_i − h_i(x))²` [P: Anwar eq. 17.9; Sayghe eq. (5)]. Its defining property — the one that matters for you — is **redundancy**: it needs `m ≥ n` [P: Youssef §II], and it is the *excess* of measurements over states that produces a nonzero residual and hence any detection capability at all.

**4. Physical ground truth.** Layer 1. In your twin this is the OpenDSS solution driven by the **true** (unattacked) load inputs. **[R]** You must retain this: run the twin twice per timestamp — once on clean inputs (ground truth, for labelling and impact measurement) and once on the delivered/possibly-attacked telemetry (operator's view). Without the clean run you have no labels and no impact metric.

**5. Simulated sensors.** The bridge from Layer 1 to Layer 2. A sensor object is `z_k = h_k(x_true) + e_k`, `e_k ~ N(0, σ_k²)`, published at a stated rate from a stated location. **A quantity becomes a cyber measurement at exactly the moment you instantiate such an object and publish it.** Before that it is a Layer-1 number in the solver's memory.

**6. LSTM forecasting.** Layer 3. It models the *temporal* structure of a signal. Which signal it consumes is the entire architectural question — see Deliverable 5.

**7. FDI attack surface.** Strictly the set of Layer-2 objects, plus (if modelled) discrete topology telemetry [P: Youssef §III-B; Caleb Table 2]. Formally: the attacker applies `z_delivered = z_true + a` somewhere on the path between sensor and consumer. Lei §III-B enumerates the three realistic insertion points: compromise the terminal instrument before encryption, hijack the uplink channel, or compromise the control centre directly; and observes that attackers prefer the edge because it is computationally weaker.

**8. Anomaly detection.** Layer 3. Consumes Layer-2 data and/or Layer-3 residuals and emits a decision.

## 2.3 Direct answers to your questions

|Question|Answer|
|-|-|
|What is actually **measured**?|Only what a sensor object publishes. **[I]** Today: 49 × (P, Q) at 49 load nodes. Nothing else.|
|What is **estimated**?|Nothing, today. **[I]** You have no state estimator; you have a simulator driven by measurements.|
|What does **OpenDSS calculate**?|The complete physical solution: all phase-node voltages, all branch currents and flows, losses, transformer quantities, given *all* inputs and the full model.|
|What is **physical ground truth**?|The OpenDSS solution driven by the **true** load inputs (the clean run).|
|What can an attacker **directly modify**?|Only Layer 2: the published telemetry of instantiated sensors, in transit or at the instrument. **[I]** Today, exactly the 49 P/Q streams on `smartgrid.telemetry`.|
|Which OpenDSS outputs are **merely simulated physical states**?|All of them, by default — bus voltages, currents, flows, losses, transformer loadings — until you wrap them in a sensor.|
|When does an OpenDSS quantity **become a cyber measurement**?|When you instantiate `Sensor(location, quantity, σ, rate)` that samples it and publishes it. Not before.|
|Where does the **LSTM** belong?|Layer 3, on the telemetry stream. See D5 — recommendation is per-meter temporal prediction of Layer-2 signals.|
|Where does **AC-SE** belong?|Layer 3, between telemetry ingestion and detection — **but only once measurement redundancy exists** (D4).|
|What should **NOT** be treated as an attack surface?|Unsensored OpenDSS internals; equipment ratings (no paper support); the physical load itself (that is LAA, a different class [P: Maleki]); dispatch commands (that is command injection [P: WangCPA]); OpenDSS's internal solver state.|

## 2.4 Clean end-to-end architecture

```
                    ┌───────────────────────────────────────────────┐
                    │  L1  GROUND-TRUTH PHYSICS (clean run)          │
                    │  OpenDSS IEEE-123, 3-φ unbalanced              │
                    │  driven by TRUE loads (metered + pseudo)       │
                    └───────────────┬───────────────────────────────┘
                                    │ x_true  (all phase-node V∠δ, I, S)
                    ┌───────────────▼───────────────────────────────┐
                    │  L2  SENSOR LAYER   z = h(x_true) + e          │
                    │  • 49 AMI meters       (P,Q,|V|)               │
                    │  • 1 substation head   (P,Q,|V|,|I|) per φ     │
                    │  • k branch flow units (P,Q or |I|) per φ      │
                    │  • [optional] 1-2 PMUs (V∠δ)                   │
                    └───────────────┬───────────────────────────────┘
                                    │ z_true
                    ┌───────────────▼───────────────────────────────┐
                    │  ATTACK INJECTOR   z_delivered = z_true + a    │
                    │  in-transit, between producer and consumer     │
                    │  emits ground-truth label + attack metadata    │
                    └───────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │  KAFKA  smartgrid.telemetry                    │
                    │  (+ smartgrid.labels, smartgrid.state)         │
                    └───────────────┬───────────────────────────────┘
                                    │ z_delivered, buffered by timestamp
        ┌───────────────────────────┼───────────────────────────────┐
        │                           │                               │
┌───────▼────────┐   ┌──────────────▼──────────────┐   ┌────────────▼──────────┐
│ L3a TEMPORAL   │   │ L3b  AC-DSSE (3-φ, WLS)      │   │ L3c OPERATOR TWIN     │
│ per-meter GRU/ │   │ z_delivered + network model  │   │ OpenDSS run on the    │
│ LSTM predictor │   │        ↓                     │   │ DELIVERED telemetry   │
│      ↓         │   │   x̂, r = z − h(x̂)           │   │        ↓              │
│ Γp = ‖z − ŷ‖   │   │   J(x̂), LNR                  │   │ V/I/loading estimate  │
└───────┬────────┘   └──────────────┬──────────────┘   └────────────┬──────────┘
        │                           │                               │
        └───────────────┬───────────┴───────────────────────────────┘
                        │
              ┌─────────▼──────────────────────────────┐
              │ L4  FUSION DETECTOR                     │
              │ predicted-residual classifier           │
              │ + SE residual test                      │
              │ + feeder-head power-balance check       │
              │  →  {benign, attacked}, location, score │
              └─────────┬──────────────────────────────┘
                        │
              ┌─────────▼──────────────────────────────┐
              │ L5  EVALUATION                          │
              │ vs. ground-truth labels + clean run:    │
              │ TPR/FPR/precision/recall, detection     │
              │ latency, ΔV error, Δloading error       │
              └────────────────────────────────────────┘
```


### 2.5 Terminology — two words that mean different things in different literatures

**"Digital twin."** Three of the located papers use the term in three incompatible senses:
- **TF25 §III-B**: a model class — something that *"emulate[s] the behavior of an actual
  asset... relying on a model of the asset and its measurements,"* defined by real- or
  near-real-time synchronisation. Closest to our usage.
- **Rin25 §III-C**: an *enterprise integration platform* — an ADMS consolidating GIS,
  SCADA, AMI, call-centre and asset management into one system of record. Explicitly not
  a physics simulation.
- **Ald25**: does not use the term at all; the same functionality is called FASE/DSSE.

**[R] Define ours narrowly and early:** an OpenDSS AC three-phase power-flow model of the
IEEE-123 feeder, synchronised per timestamp to a streaming telemetry feed, run in two
instances (clean ground truth and operator-side delivered) to permit impact differencing.
Cite TF25 and Rin25 to show the term's spread. Without this, a reviewer may read "digital
twin" in Rin25's sense and ask why we do not integrate GIS and asset management.

**"Unobservable."** Two senses, both standard, and Rah26's title conflates them:
- **Classical power-systems sense** (ours in D4): rank-deficiency of `H` — too few
  measurements to determine the state at all.
- **FDIA-literature sense** (Rah26, and A4/A5): an attack leaving the *residual*
  invariant, regardless of whether the system is classically observable.

These are independent properties. Our feeder is unobservable in the first sense **and**
is the target of unobservable attacks in the second. State the distinction explicitly in
the attack-taxonomy section.

---
# DELIVERABLE 3 — Evaluation of your current implementation

**Overall diagnosis, stated once, up front.** Your twin is **driven by** the measurements. That makes it an exact input→output map: if an attacker changes the 49 P/Q streams, the twin produces a perfectly self-consistent physical solution for the *falsified* loads. There is no residual, no contradiction, nothing to detect — **not because your detector is weak, but because none is possible in the current topology of information.** Detection requires either *spatial redundancy* (`m > n`, which is what makes SE residuals nonzero [P: Youssef §II; Liang §V-A]) or *temporal redundancy* (prediction from history, which is what Lei's CPR scheme exploits). You have neither today. Everything below follows from this.

|Component|Current status **[I]**|Strength|Weakness|Research validity|Required change **[R]**|
|-|-|-|-|-|-|
|**Representative-meter selection**|49 of ~123 nodes, working|Principled and defensible; matches the real constraint that distribution has very low measurement redundancy [P: Anwar §17.3(e); Sayghe §2.3]|Selection criterion optimises *feeder coverage*, not *estimation observability* or *attack-surface criticality* — three different objectives|Valid as an engineering choice; **not yet** a research contribution|Reframe the same scoring function as (a) an observability-aware placement and (b) an attack-criticality ranking. This is your cheapest real novelty (D9-2)|
|**Hybrid placement / scoring**|Topology, downstream load, branch, depth, backbone coverage, size class, score|Uses genuine feeder structure; scales to >49 by design|No baseline comparison; no measure of what is lost vs. full metering; no link to `‖Hc‖₀` sparsity or to protected-set theory [P: BiZhang; Liang §V-A]|Currently unvalidated|Add: (i) random-placement and degree-based baselines; (ii) an observability/redundancy metric per configuration; (iii) a critical-measurement ranking|
|**49 synthetic datasets**|Correlated Gaussian noise + Cholesky over household power data|Reproducible; gives 49 synchronised 1-min streams; correct instinct that inter-meter correlation matters (Youssef §IV notes that even residual-preserving stealth attacks damage the correlation structure among measurements, and that this is exploitable)|**This is your weakest component, with four distinct defects.** See 3.1 below|**Currently a validity threat to every result you would publish**|Rebuild per 3.1|
|**Kafka telemetry**|Working producer→topic→consumer|Directly addresses a real gap: Musleh reports only 10.2% of reviewed detection work does real-time testing; Irfan §8.5 names the offline→online transition as an open direction|Single topic; no schema/versioning; no separation of measurement, label, and state streams; no place to insert an attack|Strong differentiator once exploited|Split topics: `telemetry.raw`, `telemetry.delivered`, `labels`, `state`. Insert the attack injector **between** producer and consumer, so the attack is a transport-layer event, matching Lei §III-B's channel-hijack model|
|**Timestamp synchronisation**|Buffers, waits for all 49, then solves|Correct and necessary; SE and any snapshot detector need a coherent time slice|Blocking on all 49 is brittle: one late/dropped meter stalls the pipeline, and **a DoS on one meter becomes a silent outage of the whole twin**. Real DSSE must tolerate missing data [P: Sayghe §2.3 — DS measurements are often not time-synchronised and are collected infrequently]|Valid for a controlled experiment; unrealistic as an industry claim|Add a window timeout + missing-data policy (hold-last, or pseudo-measurement substitution). Log dropouts as a distinct event class so you can distinguish loss from attack — Irfan §8.7 names cyber-vs-physical discrimination as an open problem|
|**OpenDSS IEEE-123**|Solves once per timestamp|**Your single strongest choice.** Unbalanced 3-φ AC is exactly what the literature says is required and largely missing: Anwar §17.3 (positive-sequence modelling does not work for LV distribution; full multi-phase needed); Sayghe §2.3.1; Caleb §4.2 (extend FDIA formulations to multi-phase unbalanced distribution) and §4.4 (abandon DC simplifications)|Currently used as a *simulator only*; contributes nothing to detection|Excellent foundation|Use it in **three distinct roles**, kept separate: ground-truth generator, `h(·)`/Jacobian provider for the estimator, and operator-side consistency checker|
|**Telemetry extraction**|Partial; "not all relevant quantities extracted"|—|Undefined extraction = undefined measurement model. This is the gate on everything downstream|Blocking issue|Define the measurement model **first** (D10 Phase 2), then extract exactly those quantities and nothing more. Extracting everything invites the Layer-1/Layer-2 confusion this review exists to prevent|
|**FDI model**|Not implemented|—|—|—|See D6. Build the injector as a transport-layer component with a formal `a` and a ground-truth label per timestamp|
|**AC state estimation**|Not implemented|—|**Cannot be built on 49 P/Q streams** — see D4. Attempting it will produce an underdetermined system that either fails to converge or converges to an arbitrary point in the null space|—|Add redundancy first (substation head + branch flows + pseudo-measurements), then build 3-φ WLS|
|**LSTM/GRU/TCN**|Not implemented|—|Risk of adding it for the wrong reason. Musleh's own criteria for future detectors are selectivity, robustness, self-configuration, deterministic computing, and layered schemes — none of which is "use deep learning"|—|See D5: use it where it supplies *temporal redundancy you do not otherwise have*|
|**Residual detection**|Not implemented|—|Classical BDD residual is **provably useless against stealth FDIA** by construction [P: Liang eq. (9); Sayghe eq. (36); Caleb eqs. (5)–(8); Lei Fig. 7 shows the classical residual of a constructed FDIA sitting near zero]|—|Implement it anyway — as the **baseline you beat**, and as the detector for non-stealthy attacks. Do not present it as your contribution|

## 3.1 The synthetic dataset problem — RESOLVED by dataset replacement

The four defects below were identified against the original UCI + Cholesky corpus.
**Defects 1–3 are resolved by replacing the data source entirely** rather than by
patching the generator; defect 4 is resolved by explicit modelling. The Cholesky
generator is deleted, not fixed.

**Replacement source: IDEAL Household Energy Dataset** (University of Edinburgh),
CC BY 4.0, 1-second native resolution, whole-home `electric-mains`/`electric-combined`
channel, ~232 homes after filtering (from 255; home 223 excluded — no released mains
data; further exclusions for install duration and propagation quality — record the
exact post-filter count in generator metadata).

1. **Dimensional and physical incompatibility — RESOLVED by aggregation.** Household
   kW substituted directly for feeder spot loads under-loaded the metered subset by
   ~14.7× (Phase 0 measured: 49 meters contributing ~135 kW against 1985 kW nominal),
   appearing as only ~2× at feeder aggregate because static unmetered load diluted it.
   Both figures are correct at their own denominator; **the 14.7× figure sizes the fix.**
   The fix is *aggregation*, not rescaling: each metered node is built as the sum of N
   drawn IDEAL household series plus a modest correction factor to nominal. Rescaling a
   single household ~15× preserves single-household spikiness at a node that physically
   serves tens of homes and should be far smoother; summing produces the smoothing for
   the same reason real feeder nodes are smooth. `Sub_metering_*`-class channels are
   never used as power quantities.

   **Reactive power is wrong in shape, not only magnitude, in any household source.**
   Households run at pf ≈ 0.995; IEEE-123 nominal is pf ≈ 0.876 (3490 kW / 1920 kvar).
   Discard any dataset-supplied reactive column and derive Q from the node's own nominal
   power factor: `Q_k(t) = P_k(t)·tan(φ_k^nom)`. Aggregate kvar is checked against
   nominal alongside kW.

2. **Household voltage/current are not feeder measurements — RESOLVED by construction.**
   The dataset supplies **only `P`**. Voltage and current at a metered node come from the
   OpenDSS solve, wrapped by a Phase-2 sensor object: `|V|_k = |V|_k(x_true) + e`.
   Enforced at the producer schema level by rejection-on-presence, not by omission at
   read time, so household-service-point quantities cannot re-enter during Phase 2.
   This remains the single most consequential correction in this review.

3. **Manufactured correlation — RESOLVED by source replacement.** Cholesky-correlated
   copies of one household differ by *noise*; real meters differ by *behaviour*. IDEAL
   supplies ~232 genuinely distinct households with per-home metadata (`occupancy`,
   `hometype`, `residents`, `occupied_days`/`nights`), so heterogeneity and peak
   desynchronisation come from the source rather than from imposed jitter, and the draw
   can be *stratified* across occupancy and dwelling type — demonstrable from metadata,
   not merely asserted. Youssef §IV's point stands: correlation structure is the
   exploitable signal, so it must not be manufactured. Residual correlation between nodes
   arising from shared weather, time-of-day and day-of-week is **correct and physical** —
   do not drive it to zero.

   Non-stationary per-meter events and full generator metadata (draw seeds,
   home-to-node mapping, window, correction factors) remain required: Phase 9's
   correlation-ablation and Phase 13's release both depend on the corpus being
   regenerable with imposed structure stripped.

4. **Unmetered nodes — RESOLVED by explicit pseudo-measurement modelling.** Use the
   measured count: Phase 0 confirms 91 `Load` objects, 49 metered, **42 unmetered** —
   not the ~74 a bus-count subtraction suggests. Loads and buses are different
   populations; load-free buses require no *load* pseudo-measurement (their
   zero-injection constraints are a separate and more valuable input — see §4.3 and
   the D10 Phase 7 amendment). Model the 42 as pseudo-measurements with an explicitly
   *estimated* σ_pseudo (see D10 Phase 5 amendment), recorded in config.

**Dataset differentiation [R]:** none of the 15 papers in §0.1b use real multi-household
metering data at this scale — all use OpenDSS/Matpower Monte Carlo synthetic loads, with
the partial exception of Zid (one hour of real µPMU data) and Ald25 (CER Ireland, 30-min).
IDEAL at 1-min aggregated from ~232 homes is a genuine methodological differentiator and
should be stated as a strength in the write-up, not treated as incidental plumbing.
---
# DELIVERABLE 4 — Observability and measurement design

## 4.1 The distinction you asked me to make explicit

```
OpenDSS (power flow):
    full network model + all parameters + ALL load inputs specified
        → complete physical solution, unique, no redundancy, no residual
    It is a well-posed forward problem: #equations == #unknowns, by construction.

AC State Estimator:
    finite, noisy, INCOMPLETE measurement set + network model
        → x̂ that best explains the measurements in a weighted-least-squares sense
    It is an over-determined inverse problem: it REQUIRES m ≥ n [P: Youssef §II],
    and its usefulness comes from the surplus (m − n), which is what makes r ≠ 0.
```

The trap you correctly anticipated: *OpenDSS being able to compute the whole feeder state says nothing about observability*, because OpenDSS is given the inputs. An estimator is not. Anwar §17.2 frames SE's purpose as estimating unmeasured variables and filtering error — a purpose that presupposes the inputs are *not* known.

## 4.2 Requirements for observability, and where you stand

**(a) State dimension.** For a single-phase-equivalent `n`-bus system, `dim(x) = 2n − 1`
[P: Anwar eq. 17.1; Sayghe eq. (1)]. For an **unbalanced three-phase** distribution feeder
this must be counted **per energised phase-node**, because Anwar §17.3 and Sayghe §2.3.1
both require full multi-phase modelling with a 3×3 phase impedance matrix `Z_abc,ij` per
branch [P: Sayghe eq. (21)].

**[R] Do not assume 123, and do not assume the angle reference.** The IEEE-123 feeder is a
*node* feeder with a mix of single-, two- and three-phase laterals, so `N_phase_nodes` is
substantially larger than the bus count. Take it from the **compiled** circuit
(`DSS.ActiveCircuit.NumNodes` / `AllNodeNames`, snapshotted in `circuit_topology.json`),
never from design-level metadata — the compiled circuit contains nodes the design
description never names, notably the regulator secondaries `150r.1/2/3`, which carry a
different voltage from `150` and are therefore genuine states.

**The angle reference changes the state dimension, and the naive choice is
ill-conditioned.** Sha25 §III shows that the single-angular-reference (SR) approach —
pinning phase a of the reference bus and leaving phases b and c free — makes most Jacobian
columns linearly dependent, producing a rank-deficient problem *"possible only if all phase
angles of the reference bus are already known."* Sha25 Table II corroborates this on
iteration count: methods with fixed known angular references converge in 3–4 iterations,
the SR/PR variants in 4–5, and on IEEE-37 with Δ–Δ/Y–Y transformers at the root the SR
method fails to converge acceptably at all. Sha25 §V.A further warns that SR is sensitive
to flat-start initialisation (all reference angles at 0° gives *"irregular results"*).

XuDS25 §II.D solves the same problem by placing a µPMU at each subarea reference bus and
using its **measured** angle — simpler, but requires hardware this architecture does not
budget.

**[R] Resolution — measured, not assumed.** Model verification confirms the OpenDSS
`Vsource` is balanced to within **0.002°** of 0 / −120 / +120 at both bus 150 and the
regulated 150r, so all three source angles are known exactly by construction and are fixed
as references rather than estimated:

```
N_phase_nodes = 278                    (across 132 buses)
dim(x) = 2 · 278 − 3 = 553
```

Source **magnitudes** remain states. The substation-head sensor measures `|V|` per phase
(§4.3 item 2), so treating them as measured rather than known is both standard and
consistent with the sensor registry. This settles the earlier 555 / 553 / 550 ambiguity:
**553 is the figure; 555 (single pinned angle) is the configuration Sha25 shows to be
ill-conditioned, and should no longer be quoted.**

Note that this ill-conditioning is **specific to the classical WLS + Gauss-Newton family**.
DKL25, TAPINE and BMAML sidestep it (feed-forward regressors, no Jacobian inversion at
inference) and XuUD26's cubature filter avoids explicit Jacobian formation — so the absence
of the problem in those papers is not evidence against Sha25.

**(b) Measurement count and redundancy.** WLS requires `m ≥ n`; practical BDD requires
`m > n` with the redundancy distributed such that no measurement is *critical* (a critical
measurement has zero residual by construction and its error is invisible). Liang §V-A,
reporting Bobba et al. and Liu et al., makes the dual statement precisely: the size of a
**basic measurement set** — a minimum set of measurements whose rows in `H` are linearly
independent and sufficient to solve for the states — equals `n`; and if an attacker can
compromise `k ≥ m − n + 1` meters, an undetectable attack vector always exists [Liu, Ning &
Reiter (2011) §3.2, Theorem 3.2]. **Read that inequality against the configuration below.**

**(c) Your configuration, evaluated.**

| Quantity | Value |
|---|---|
| Sensors **[I]** | 49 AMI meters |
| Measured quantities per sensor **[I]** | `P`, `Q` (the twin's load inputs) |
| `m` | **98** |
| `n = 2·N_phase_nodes − 3` | **553** (measured) |
| `m − n` | **−455** |
| `m / n` | **0.18** |

**Verdict: the 49-meter configuration alone cannot support an AC state-estimation
formulation for IEEE-123. It is short by a factor of 5.6.** Publishing `|V|` at every meter
(`m = 147`) leaves a 3.8× deficit. Even at the metering ceiling of **85 meters** (six of the
91 loads share buses — see §4.3 and D10 Phase 4 sub-study 4A), `m = 170` against `n = 553`
is still 3.3× short. **No meter density closes this gap; zero-injection constraints do.**

Note the second-order consequence: with `m < n`, `HᵀWH` is singular, the WLS normal
equations have no unique solution, and Gauss-Newton will either fail to converge or
converge to an arbitrary point in the null space — a failure mode that can masquerade as
"it ran."

**Why this is unsurprising and defensible.** The literature predicts it exactly. Anwar
§17.3(e) lists "very low redundancy of measurement units" as a defining property of
distribution networks, and §17.3 names limited measurement devices, reliance on
historical-data pseudo-measurements of limited accuracy, and heavy use of current
measurements as the three properties that make DSSE hard. Sayghe §2.3 says the same: fewer
measurement points than transmission, infrequent collection (every 15 minutes or longer),
possibly unsynchronised and inaccurate, so **relying on pseudo-measurements for DSSE is
common practice**. Caleb §3.5.1 reports physics-informed graphical learning (GNN +
model-based Kalman filter) developed specifically for distribution systems **with limited
observability** — the field's answer to this exact situation. **This is a real property of
distribution systems, not a design defect.**

**Corroboration from the second tranche.** No paper in the 15 validates classical WLS DSSE
at anything close to m/n ≈ 0.18. Rah26's own WLS comparator is formally *over*-determined —
IEEE-13: `x∈ℝ⁹⁰`, `z∈ℝ¹³⁶` (m/n ≈ 1.51); IEEE-123: `x∈ℝ⁶³⁸`, `z∈ℝ⁹⁰²` (m/n ≈ 1.41)
[Rah26 §IV.A]. BMAML §III.B assumes *"the active and reactive powers at all buses"* as
measurements. Sha25's baseline is comfortably instrumented. The learned estimators (DKL25,
TAPINE, BMAML) sidestep classical observability entirely rather than demonstrating it is
achievable at low redundancy.

This **corroborates** the m ≪ n framing — four independent data points, none near our ratio
— but it also means Phase 7's classical arm operates in a regime nobody has validated.
Budget for that.

One adjacent technique not currently in the plan: XuDS25 §II.D eq. (32)–(39) reformulates
the filter update as *"redundant linear regression"* to raise **effective** measurement
redundancy against outliers without adding physical meters. Not a fix for m ≪ n, but a real
mitigation worth evaluating.
## 4.3 Minimum additional measurement set

**[R], justified against the papers named.** Ordered by cost-effectiveness. Add in this order and stop when your redundancy target is met.

| # | Addition | What it measures | Count | Why — paper basis |
|---|---|---|---|---|
| 0 | **Zero-injection equality constraints at load-free nodes** | `P = 0, Q = 0`, known *exactly* | 2 × (load-free phase-nodes) | **The decisive addition.** Most of the 278 phase-nodes carry no load; each contributes two exact constraints at zero cost. Sha25 §II eq. (1)–(6) enforces these as **hard Lagrangian equality constraints**, explicitly motivated by numerical conditioning — *not* as high-weight pseudo-measurements. TAPINE eq. (15) uses a soft penalty instead, but that is a neural-network training convenience operating on a loss function, not a weighted normal-equations system with a condition number, and does not transfer to classical WLS. **Enumerate these explicitly in Phase 4.** |
| 1 | Pseudo-measurements at the 42 unmetered **Load** nodes | `P, Q` estimated from historical data, estimated σ_pseudo | 84 | Anwar §17.3(b); Sayghe §2.1, §2.3. **Distinct from row 0** — these nodes *have* consumption, just unmeasured, so they are pseudo-measurement candidates, not equality-constraint candidates. Conflating the two categories is a modelling error. 84 measurements do not close a several-hundred-state gap on their own; row 0 does most of the work. |

|#|Addition|What it measures|Count|Why — paper basis<br />|
|-|-|-|-|-|
|<br />0|Zero-injection equality constraints at load-free nodes|`P = 0, Q = 0`, known *exactly* \| 2 × (load-free phase-nodes) \||2 × (load-free phase-nodes)|Most of the 278 phase-nodes carry no load; each contributes two exact constraints at zero cost. Sha25 §II eq. (1)–(6) enforces these as **hard Lagrangian equality constraints**, explicitly motivated by numerical conditioning — *not* as high-weight pseudo-measurements. TAPINE eq. (15) uses a soft penalty instead, but that is a neural-network training convenience operating on a loss function, not a weighted normal-equations system with a condition number, and does not transfer to classical WLS. **Enumerate these explicitly in Phase 4.*|
|**1**|Pseudo-measurements at the 42 unmetered|`σ_pseudo`nodes \| `P, Q` estimated from historical data, estimated σ_pseudo|2 × (unmetered nodes)=84|Anwar §17.3(b); Sayghe §2.1, §2.3. **Distinct from row 0** — these nodes *have* consumption, just unmeasured, so they are pseudo-measurement candidates, not equality-constraint candidates. Conflating the two categories is a modelling error. 84 measurements do not close a several-hundred-state gap on their own; row 0 does most of the work.|
|**2**|**Substation head measurement** (per phase)|`P`, `Q`, `\|V\|`, `\|I\|` at the feeder head|4 × 3 = 12|Gives an independent constraint on total feeder throughput. Anwar §17.3.1 (as reported via Sayghe eq. 17) treats the substation three-phase power injection, phase-to-neutral voltage magnitude, and current injection magnitude as available measurements in DSSE formulations. **Operationally this is the single highest-value sensor on a real feeder and it is nearly free to add.** It also enables the power-balance cross-check in D7.|
|**3**|**Branch flow / current sensors at backbone segments**|`P`, `Q` (or `\|I\|`) per phase on the main trunk and each major lateral head|~6–12 locations|Youssef §II: meters placed on lines to monitor transmitted power provide **measurement redundancy**, which is used to validate the dataset and identify or eliminate bad data — this is the textbook justification for line measurements. Sayghe §2.3.1 eq. (23) makes branch currents `I_abc,ij` explicit for branch-current-based DSSE; Anwar eq. (17.7) gives the current-magnitude measurement function. **[R]** Use your existing backbone-coverage score to place these — it is already the right criterion, applied to a different sensor class.|
|**4**|**Voltage-magnitude reporting at existing meters**|`\|V\|` per metered phase-node|+49|Free in a real AMI (Caleb §3.4.1: smart meters record voltage) and free in your twin (OpenDSS already solves it). Directly constrains the voltage half of the state vector.|
|**5**|*(Optional)* **1–2 PMUs** at electrically significant nodes|`V∠δ`|2|Liang §V-B: PMUs provide synchronised phasors and are harder to compromise; Kim & Poor's result (via Liang) is that attacks may become infeasible when ~1/3 of buses are PMU-protected. **[R] Do not chase 1/3 coverage** — that is a transmission-economics result. Two PMUs give you an angle reference and a stealth-limiting anchor at negligible cost. Note the counter-evidence: Liang also reports GPS time-stamp spoofing as an attack that bypasses PMU protection entirely (its ref [38]).|
|**6**|**Transformer sensors**|—|0|**Not justified.** No uploaded paper models transformer telemetry as an FDIA surface, and RahmanMR treats tap position as *attacker uncertainty*, not as a measurement. Adding these would be adding components without support.|

**Acceptance test for observability [R].** Do not declare observability by counting. Build `H(x) = ∂h/∂x` at a flat start and check `rank(H) = dim(x)` and `cond(HᵀWH)` within tolerance; then compute per-measurement redundancy and flag any **critical measurements** (those whose removal makes the system unobservable — their residual is identically zero, so an attack on them is invisible regardless of your detector). That critical-measurement set is simultaneously your observability report *and* your highest-priority attack surface, which is a neat and honest way to connect D4 to D6.

---
# DELIVERABLE 5 — Where the LSTM belongs

## 5.1 Evaluation

**A — LSTM predicts load, feeds OpenDSS.** This is *forecasting*, not detection. It produces `P̂, Q̂` for the next interval and drives the twin forward. There is no detector in this loop at all: nothing compares prediction to observation. Caleb §3.5.4 in fact lists load/price forecasting models (naming LSTMs) as an **attack target** — poisoning their training set is an attack, not a defence. **Useful as a simulation utility; rejected as a detection architecture.**

**B — LSTM predicts raw meter telemetry; detect on the predicted residual.** This is Lei's CPR scheme, essentially exactly. Lei defines the predicted residual `Γp_t = ‖z_t − y_t‖` where `y_t` is the GRU's prediction from history, and states the key property plainly: because the prediction is derived from historical data it is **independent of false data injected at the current instant**, so `Γp_t` substitutes for the classical BDD residual. Lei then classifies the *sequence* of predicted residuals with a wavelet-transform CNN rather than thresholding it. Fig. 7 of that paper shows the classical residual of a constructed FDIA collapsing toward zero while the scheme still detects at injection magnitudes of ~0.5–2.5‰. It was validated on a real microgrid testbed **and** an open Austin, TX dataset, on edge hardware (ARM Cortex-A7, 4 GB), which is a real-time result — rare, per Musleh's 10.2% statistic. Sayghe §5 and Musleh §III-B1 independently record RNN-family detectors as a mainstream approach.

**C — AC-SE → residual sequence → LSTM.** This is ZhangYang, precisely: a Kalman filter produces `x̂_t`, the normalised observation `o_t = ‖z_t − Hx̂_t‖/‖ω‖` is formed, `m` past observations are fed through an LSTM, and a two-layer DNN emits the binary decision. Their stated motivation is the correct one: fixed thresholds trade detection accuracy against false alarms, and the LSTM learns the threshold behaviour implicitly. **But** it presupposes a working estimator, and per D4 you cannot have one on 49 meters. Their `H` is also a linear measurement matrix inside a KF on balanced IEEE bus systems — the transfer to a 3-φ unbalanced feeder is not free.

**D — Fusion.** No uploaded paper implements exactly your D. The nearest support is architectural rather than empirical: Musleh's closing criteria for future detectors include **layered detection schemes**, on the explicit reasoning that no single algorithm is 100% secure; Musleh Table II also frames model-based and data-driven families as having complementary failure modes (model-based fails under parameter uncertainty; data-driven fails on overfitting and unseen attacks — a point Youssef §IV repeats, noting ML detectors tend to outperform on known attack models but perform poorly on unknown ones). Caleb §3.5.1 documents physics-informed hybrids (GNN + Kalman filter; physics-guided autoencoders) for limited-observability distribution SE. So fusion is **well-motivated but not yet demonstrated** for this configuration — which is exactly what makes it a contribution rather than a replication.

## 5.2 Comparison

|Criterion|**A** load forecast|**B** telemetry prediction + residual classification|**C** SE residual → LSTM|**D** fusion|
|-|-|-|-|-|
|Detects stealthy FDI|No detector|**Yes** — prediction is independent of the current injection [P: Lei §III-D]|Yes, if the SE is sound [P: ZhangYang]|Yes, strongest|
|Detects non-stealthy FDI|No|Yes|Yes (classical BDD already suffices here)|Yes|
|Detects sparse/targeted FDI|No|**Yes, and localises natively** — per-meter models give per-meter scores, addressing Irfan's localisation gap|Only if the estimator localises; SE residual is global|Yes|
|Distribution-grid fit|n/a|**High** — needs no observability, no `H`, no redundancy|**Low today** — blocked by D4|Medium (inherits C's prerequisite)|
|Industry relevance|Low|**High** — edge deployment demonstrated on real hardware [P: Lei §IV-C]|Medium — DSSE is not universally deployed|High but heavy|
|Implementation complexity|Low|**Low–medium**|High|High|
|Explainability|n/a|Medium (per-meter residual is interpretable; the classifier is not)|Medium–high (residual is physically meaningful)|**Highest** — you can say *which* evidence fired|
|Computational cost|Low|**Low after training**, and parallel per meter [P: Musleh Table II — data-driven cost is front-loaded into training]|High — iterative AC-SE per timestamp, `O(n³t)` for nonlinear WLS [P: Musleh §IV-B1]|Highest|
|Publishability|Very low|Medium alone — **high** in this setting, because Lei is a microgrid/AMI result and no uploaded paper transfers it to an unbalanced feeder twin|Medium — ZhangYang already did the transmission version|**Highest**|
|Fit to your code|**Excellent** — the consumer already assembles per-meter time series keyed by timestamp|Excellent|Poor today; good after D4 sensors are added|Good after D4|

## 5.3 Recommendation

> **Primary architecture: B, hardened by a physics cross-check — i.e. a deliberately
> *reduced* D that does not require a state estimator on day one. C is added later, as a
> scientific completion, once the D4 sensor set makes a genuine estimator possible.**

Concretely, the primary detector is:

```
per-meter telemetry stream  →  GRU/LSTM one-step prediction  ŷ_k,t
                            →  predicted residual  Γp_k,t = ‖z_k,t − ŷ_k,t‖
                            →  classifier over the residual SEQUENCE (window w)
                            →  per-meter score  s_k,t
                                                    ⊕
feeder-head power balance:  P_head − (Σ_k P_k + P_loss^twin)  →  imbalance residual
                                                    ⊕
[phase 2] AC-DSSE residual r = z − h(x̂), J(x̂), LNR
                                                    ↓
                                        fusion → decision + localisation
```

**Why this and not the others [R]:**

* **B is the only option that works with the information you actually have.** It needs no
  `H`, no redundancy, no observability — it *manufactures temporal redundancy where spatial
  redundancy is absent*. Given D4, that is decisive.
* **The physics cross-check costs almost nothing and closes B's blind spot.** A
  prediction-based detector can be defeated by an attack slow enough to look like a
  plausible trend. The feeder-head balance catches any attack that changes net feeder load,
  because the head measurement is an independent constraint the attacker did not touch.
* **The two have complementary blind spots, and the literature names both.** A
  load-redistribution attack keeps the aggregate load profile unchanged while shifting it
  between nodes [P: Youssef §III-B; Caleb Table 5]. Such an attack is invisible to the
  power-balance check *by construction* — and highly visible to the per-meter temporal
  predictor, because individual meters must move in unnatural, anti-correlated ways.
  Conversely, a slow uniform drift on all meters may slip past per-meter prediction and
  shows up immediately in the head balance. That is a real, defensible, paper-grounded
  argument for exactly two detectors — which is also the smallest defensible number.
* **C is deferred, not discarded.** ZhangYang's construction is sound and should be
  reproduced once an estimator exists — as the model-based arm of the fusion, satisfying
  Musleh's layered-scheme criterion.
* **Do not use an LSTM merely because the project is AI-based.** The justification here is
  specific and falsifiable: the LSTM supplies the temporal redundancy that a 49-meter
  unbalanced feeder cannot supply spatially. If you later add enough sensors to make
  `m ≫ n`, that justification weakens — and you should say so in the paper.

**Architecture choice within B [R]:** start with a **shared-weights per-meter model** (one
model, meter identity as a feature) rather than 49 independent models — 49 models will not
scale to a higher meter count, and Xue's one-subnet-per-bus OCON design is a cautionary
example of exactly that scaling problem. Lei's GRU-vs-LSTM-vs-CNN RMSE comparison (their
Fig. 8 / Table I) is a ready-made ablation to replicate rather than invent.

**[R] AMENDMENT — the per-meter-only design has a structural blind spot.**

Deh25 §III.C reports that **spatiotemporal 2-D CNNs (voltage × time) outperform
temporal-only 1-D CNNs by 17.33 pp AUC on IEEE-13 and 21.81 pp on IEEE-123**, against
the same nonlinear stealth-attack family we target, benchmarked against the same
authors' earlier temporal-CNN work. That is a large, reproducible effect, not a
marginal one.

Our detector as originally specified — per-meter GRU → per-meter `Γp` → 1-D CNN over
time, with no cross-meter mixing — has **no channel that models spatial (inter-meter)
structure**. The feeder-head balance check is a single scalar and does not supply one.

This matters most for **A3 (load redistribution)**, which by construction (a) leaves
the feeder-head aggregate unchanged, defeating the second arm outright, and (b) can be
crafted so each individual meter's own temporal trajectory stays smooth and plausible,
potentially defeating the per-meter GRU as well. **A3 may defeat both arms of the v1
architecture simultaneously.**

Two acceptable responses, and the choice must be made explicitly:

- **(a) Add a cross-meter channel.** Minimum viable version: feed the vector of all 49
  per-meter `Γp` scores jointly into a 2-D convolution or a pairwise-correlation feature
  before the decision layer. This preserves the shared-weight per-meter predictor (still
  scalable) while adding the spatial view Deh25 shows to matter.
- **(b) Keep the v1 architecture and report A3 as a designed failure case.** Legitimate
  and citable — a demonstrated blind spot with a stated mechanism is a real finding — but
  it must be *predicted in advance*, not discovered and reframed after the fact.

**(a) is recommended**, with (b) retained as the ablation that proves the cross-meter
channel earns its place (Phase 12).

# DELIVERABLE 6 — Rigorous FDI attack model

## 6.0 The boundary, stated once

```
CYBER MEASUREMENT                          PHYSICAL STATE
z_k = h_k(x_true) + e_k                    x_true, and everything OpenDSS derives from it
published on Kafka                          lives in solver memory
                                            
ATTACKER MODIFIES:  z_delivered = z_true + a      ← ONLY THIS
ATTACKER NEVER MODIFIES:  x_true, OpenDSS internals, the network model
```

An attack changes what the operator *believes*. The physical grid keeps doing whatever the true loads make it do. Directly editing OpenDSS bus voltages would model a *physical* attack, not an FDIA, and would make your results uninterpretable. Note also Caleb §3.5.5, which describes FDIA against digital twins specifically as **desynchronising the twin from the physical asset by injecting false data into the synchronisation link** — i.e. attacking the ingest path. That is precisely the attack your Kafka pipeline should model, and it is the one place in the uploaded literature where "digital twin" is named as an FDIA target in its own right.

**Injection point in your pipeline [R]:** between producer and consumer, on the wire. This matches Lei §III-B's second insertion mode (hijacking the measurement uplink) and Caleb's DT synchronisation-link attack. Implement it as a Kafka stream processor: consume `telemetry.raw`, emit `telemetry.delivered` + `labels`.

## 6.1 Taxonomy

Each entry: *what is modified · where · formulation · required knowledge · OpenDSS consequence · AC-SE consequence · LSTM/residual signature · paper basis.*

---
### A1 — Naive / random FDIA

* **Target:** any reachable meter's `P`,`Q` (and `|V|` if published).
* **Modified:** `z_k ← z_k + a_k`, `a_k` random, no structure.
* **Where:** wire, `telemetry.raw → telemetry.delivered`.
* **Formulation:** `a ~ U(−ρ‖z‖, ρ‖z‖)` on a random subset `S`, `|S| = k`.
* **Knowledge:** none. Read/write access to `k` streams.
* **OpenDSS consequence:** operator-side twin solves a load pattern that never existed; voltage profile and losses shift arbitrarily.
* **AC-SE consequence:** large `‖r‖`; LNR and `J(x̂)` both fire.
* **LSTM/residual signature:** large `Γp`, sharp and uncorrelated across meters.
* **Basis:** Musleh §III-A ("basic FDIA", detectable by residual test); Irfan Table 16 (random attack); Sayghe §6. **Role [R]: your sanity baseline. Any detector that misses this is broken.**

### A2 — Targeted / sparse FDIA

* **Target:** a minimal set of meters chosen to bias a *specific* node's estimated state.
* **Formulation:** `α_k = min ‖Hc‖₀ s.t. H(k,:)c = 1` — `α_k` is the **security index** of measurement `k`, the minimum number of meters to compromise to spoof it [P: Sayghe eq. (38)]. Equivalently `α_k = min_c ‖Hc‖₀` [P: Liang eq. (10)]. NP-hard for arbitrary `H`; solved by MILP, matching pursuit, LASSO, or Sou et al.'s min-cut relaxation [P: Liang §III-A; Sayghe §3.1.1].
* **Knowledge:** `H` (equivalently: topology + line impedances) for the target region.
* **OpenDSS consequence:** localised voltage-profile distortion near the target node.
* **AC-SE consequence:** if the sparsity pattern satisfies stealth, residual unchanged; otherwise a small residual bump at the compromised meters.
* **LSTM signature:** `Γp` elevated on `α_k` meters only — **this is where per-meter modelling gives you localisation for free**, addressing Irfan's stated gap.
* **Basis:** Liang §III-A; Sayghe §3.1.1 (reporting Sou et al.); Liang reporting Liu et al. on the `k ≥ m − n + 1` condition [Liu, Ning & Reiter (2011) §3.2, Theorem 3.2]. **[R] Compute `α_k` for every one of your 49 meters — the resulting ranking is a deliverable in its own right and connects directly to D9-2.**

### A3 — Load-redistribution (LR) attack

* **Target:** load measurements at multiple nodes.
* **Modified:** `P_i ← P_i + δ_i`, `Q_i ← Q_i + δ^Q_i`, subject to `Σ_i δ_i = 0`.
* **Formulation:** falsify individual consumption profiles so some nodes appear lower and others higher **while the aggregate load seen by the operator is unchanged** [P: Youssef §III-B; Caleb Table 5, "shifting load `P_D` between buses while maintaining net balance"].
* **Knowledge:** Youssef §III-B is specific and worth heeding: LR attacks demand *more* than ordinary FDIA — not only complete or partial topology knowledge but also **access to state variables**, because preserving the aggregate profile requires knowing it.
* **OpenDSS consequence:** the operator's twin shows a materially wrong spatial voltage profile and wrong line loadings, while total feeder load looks correct. In a radial feeder this is the attack that most cleanly produces **hidden overload**: Liang §IV-B reports the delayed LR attack causing line overloading undetected by the control centre, leading to physical damage.
* **AC-SE consequence:** designed to be stealthy.
* **LSTM signature:** **high** — individual meters must move in coordinated, anti-correlated, physically implausible ways.
* **Power-balance-check signature:** **zero, by construction.** This is exactly why you need both detectors.
* **Basis:** Liang §IV-B (Yuan et al.); Youssef §III-B; Caleb Table 2/5. **[R] This is your single most important attack scenario** — it is the one that is realistic for AMI, stealthy to physics checks, and detectable only temporally.

**[R] POST-REVIEW STATUS — PROMOTED TO PRIMARY SCENARIO.**

**Status: SURVIVES.** No paper in the 32 constructs an attack that zero-sums across
meters to preserve the feeder-head aggregate while redistributing among individual AMI
readings. All located attack constructions (`a = Hc`, `a = h(x̂+c) − h(x̂)`, APF/SAL
variants) are state-bias injections targeting the *estimate*, not aggregate-preserving
redistributions across the metering layer.

**[R] This is now the single most important attack in the taxonomy**, for three reasons:
1. It is the only attack class unclaimed in the located literature.
2. It is realistic for AMI — falsifying individual consumption while total feeder draw
   looks normal is exactly the deductive/additive AMI threat Caleb §3.4.1 documents.
3. **It is the attack most likely to defeat the v1 detector on both arms** — see the D5
   amendment. Invisible to the power-balance check by construction; potentially invisible
   to a per-meter temporal predictor if the redistribution is slow and small per meter.

Whether A3 is detectable is therefore an *empirical question about our own architecture*,
not a foregone conclusion, and must be treated as such in the write-up.


---
### A4 — Stealthy / model-aware FDIA, DC form — NEGATIVE CONTROL

Formulation: `a = Hc` ⇒ residual unchanged, `x̂_bad = x̂ + c` [Liang eq. (9); Sayghe
eq. (36); Caleb eqs. (5)–(8); RahmanMR eq. (5); Liu, Ning & Reiter (2011) Theorem 3.1,
eq. (4), p. 13:9].

**LNR Theorem 3.1 is proven for the DC (linear) model only. AC extension: Rah26 §II.C, Cui26.**

**Status: NARROWED — construction precedented, negative-control test still open.**

Rah26 §II.C eq. (5)–(7) implements exactly this on IEEE-13 and IEEE-123 unbalanced
feeders, proves `SH = 0` for the linearised `H`, and verifies empirically (Fig. 1) that
the LNR statistic stays below threshold across 1000 samples. **Adopt that recipe
directly** rather than re-deriving it — the attack-synthesis half of A4 is no longer
novel work.

**What remains ours:** Rah26 constructs `a` and evaluates the residual using the *same*
linearisation. `SH = 0` is then a linear-algebra identity of that shared `H`, not
evidence that the attack survives a genuinely nonlinear, iteratively re-linearised
three-phase AC estimator. The question A4 poses — *does DC/linear cancellation survive
AC + unbalance under real nonlinear re-estimation?* — is untested across the 31 reviewed papers.
That test is the contribution; the construction is borrowed.

Expected outcome per Liang §III-D and Caleb §4.4: the AC residual rises. High R/X on a
distribution feeder violates every DC assumption. A publishable negative result either way.

### A5 — AC stealthy FDIA — **CONSTRUCTION IS NOT NOVEL; REFRAME AS A DETECTION RESULT**

Formulation: `a = h(x̂_bad) − h(x̂)` [Sayghe eq. (37); Liang eq. (15)].

**Status: DEAD as an attack contribution.**

Three independent implementations now exist on unbalanced three-phase AC distribution SE:
- **Cui26** eq. (9)–(14) with an algebraic proof at eq. (13) that `‖r_φ^a‖² = ‖r_φ‖² < τ`
  — exact residual invariance for full nonlinear 3-φ WLS, on IEEE-123. Not a linearised
  approximation.
- **XuIoT26** eq. (11)–(15), phase-resolved (`a_µ` for µ ∈ {a,b,c}), citing Cui26 for the
  evasion proof rather than re-deriving it.
- **Deh25** eq. (4), implemented empirically on IEEE-13 and IEEE-123 without the
  closed-form proof.

**[R] Consequences for the plan:**

1. **Do not claim novelty for the construction.** Cite Cui26 and XuIoT26 and move on.
2. **Adopt their APF + SAL parametrisation** (attack preference factor + stealth attack
   library, Cui26 §III-B) rather than inventing a targeting scheme. This frees the
   novelty budget for A6's mechanism.
3. **Verify the `J(x̂_bad) < C` framing is actually different** from Cui26/XuIoT26's
   `‖r^a‖² < τ`. As written they look equivalent. If they are, drop the claim that the
   generalized-cost framing is a differentiator.
4. **The contribution moves to the detection side:** a temporal-redundancy detector
   catching an attack *proven* to evade BDD/SNR testing on this exact feeder class. That
   is a defensible and still-unclaimed result — none of Cui26, XuIoT26 or Deh25 tests a
   GRU-predicted-residual detector against their own attack.

### A6 — Incomplete-information FDIA — **NARROWED, mechanism still ours**

Formulation: `a = H̄c = Hc + δc`; perfect iff `δc = 0` [RahmanMR eqs. (7)–(15);
Liang eqs. (11)–(13)]. Cut theorem: knowing all lines on one cut guarantees `δc = 0`
given `c` uniform per island [RahmanMR Theorem 1].

**Status: NARROWED.** Three distinct formalisations of attacker imperfection now exist,
none matching ours:
- **XuDS25** eq. (19)–(20): relaxation ball `‖a − tr{HC}‖₂ ≤ ε`, with ε = 0 recovering
  the perfect case — limited **topology knowledge**, full channel access. Closest
  conceptual precedent; usable as formal scaffolding.
- **Rah26** §II.C.3: `a = MHc` with `M` a diagonal channel-selection matrix — limited
  **channel access**, `H` exactly known. A structurally different axis; cite as a third
  formalisation, not as precedent for ours.
- **XuIoT26** §II.C, §II.G: frames full knowledge as an explicit worst case and argues
  its Huber-loss reconstruction degrades gracefully under *"partial or noisy
  information"* — engages the theme without a mechanism.

**What remains unclaimed across the 31 reviewed papers:** the shadow-model mechanism (a perturbed
second OpenDSS instance generating `δc` implicitly rather than analytically), the
two-variant design (diffuse model error vs. exact knowledge of a single branch), and the
radial-feeder graph-cut argument. **No reference list in the 15 contains anything
resembling RahmanMR's Theorem 1 or a cut-based knowledge-requirement result** — that
line of work is absent from this literature.

**Implementation:** shadow OpenDSS instance with `σ_R > σ_X` (resistance drifts with
conductor temperature, ~0.4%/°C; reactance is geometry-set and drifts less), regulator
taps frozen at nominal while the real ones move. All three of A5/A6-diffuse/A6-cut fall
out of one injector by varying where σ is nonzero:

| Variant | Shadow configuration |
|---|---|
| A5 (perfect knowledge) | σ = 0 everywhere |
| A6 diffuse | σ > 0 on every line |
| A6 cut | σ > 0 everywhere except one branch, exact; `c` uniform per island |

Sweeping σ against attack magnitude `‖c‖` yields the Teixeira proportionality
empirically — a 2-D grid of (model error × magnitude) → detected/undetected. That grid
is a figure.

**Open research question, honestly stated:** RahmanMR Theorem 1 is a **DC** result. The
cancellation works because DC flow depends only on angle *differences*, so a uniform
shift within an island leaves internal flows untouched. Under AC with magnitude coupling
and high R/X, whether it survives is genuinely unknown. Flat residual under a magnitude
sweep ⇒ transfers (alarming, novel); growing residual ⇒ breaks (defence-positive, and
thematically the same finding as A4). Either outcome is reportable.
---
---
### A7 — P/Q measurement attacks (AMI-specific)

* **Deductive (theft):** `P_i(t) = P_i(t)^act − d_t`, `d_t ∈ [d_min, d_max]` [P: Caleb §3.4.1].
* **Additive:** over-reporting consumption [P: Caleb §3.4.1].
* **Multi-parameter consistency-preserving:** simultaneously manipulate `P`, `I`, and `S` **while preserving `S_i(t) = V_i(t)·I_i(t)`**, specifically in order to degrade detector performance [P: Caleb §3.4.1, reporting its ref [204]].
* **[R] That last one is directly aimed at your architecture.** If you publish `P`, `Q`, `|V|`, `|I|` per meter, a naive intra-meter consistency check (`S = VI`) is your cheapest detector — and this documented attack defeats it exactly. Implement it as an explicit adversarial case and report that your detector survives it (or does not).

### A8 — Voltage / current / phasor measurement attacks

* **Target:** `|V|`, `|I|`, and (only if PMUs are modelled) `V∠δ`.
* **Canonical example:** Youssef §III gives a slow scaling manipulation of voltage magnitudes across a whole grid section as the archetypal stealth FDIA.
* **OpenDSS consequence:** if `|V|` is only *reported* (not used as a load input), the twin's physics is unaffected — but the estimator and any voltage-violation alarm are corrupted. **This is a clean, teachable demonstration of the Layer-2/Layer-1 boundary.**
* **[R] Include a slow voltage-scaling attack.** It is directly cited, it targets a measurement type you can add for free, and it is genuinely hard for a threshold detector.

### A9 — Line-flow attacks

* **Only meaningful once you have instantiated line sensors (D4 item 3).** Then: `P_ij, Q_ij ← P_ij + a`, targeting the redundancy those sensors provide.
* **Basis:** Youssef §II (line measurements exist to provide redundancy and support bad-data identification); RahmanMR §II-A (measurement set = injections + both-direction line flows); Xue §II (`P_km`, `Q_km` in `z`).
* **[R]** Attacking the very sensors that provide redundancy is the natural escalation and a good ablation axis.

### A10 — Topology / breaker-status attack

* **Modified:** discrete switch/breaker status bits, hence `H` itself [P: Youssef §III-B; Liang §III-C; Caleb Table 2].
* **Formulation:** the estimator uses a false topology `H̄`, `z̄ = H̄x + ē`; the attacker must modify continuous *and* discrete data simultaneously to evade both BDD and topology-error identification [P: Liang §III-C].
* **[R] SCOPE DECISION: exclude from v1.** IEEE-123 has switches and its reconfiguration is a real thing, so this is tempting. But it doubles the attack model, needs a topology estimator you do not have, and Liang §III-C notes even the transmission literature on it was recent and thin. **Name it as future work.** Note the honest counterweight: Youssef §IV observes that graph-theoretic detectors assume the topology estimator's output is ground truth, which leaves that whole detector family blind to topology attacks — so this is a real gap, just not yours to close this cycle.

## 6.2 Attacks you should NOT build

|Excluded|Why|
|-|-|
|Editing OpenDSS internal state variables|Not an FDIA; breaks the cyber/physical boundary; makes results uninterpretable|
|Market / LMP attacks|Xie, Esmalifalak, QZhang are all transmission-market. A distribution feeder twin has no LMP to attack|
|AGC / frequency attacks|Musleh §II-B2, Caleb §3.2 — generation-side, transmission, dynamic. Not a radial 4.16 kV feeder|
|Load-altering attacks (LAA)|Maleki draws the line explicitly: LAAs manipulate the *physical* load via IoT devices; you are studying measurement integrity|
|False command / dispatch injection|WangCPA — different threat class, needs a control loop you do not model|
|Adversarial ML attacks on your own detector|Caleb §3.5 and Irfan §8.4 both flag these as live threats and they are genuinely interesting — but they are a second paper, not a section of this one|

---
# DELIVERABLE 7 — Industry-relevant architecture

**Target:** real-time FDI detection in an unbalanced IEEE-123 distribution feeder digital twin.

|Layer|Component|Input|Output|Technology|Purpose|Literature basis|
|-|-|-|-|-|-|-|
|**Data**|Load-profile generator|Base household/AMI profiles, per-node nominal spot loads|49 metered `P,Q` series + pseudo-load series for unmetered nodes, scaled to feeder ratings|Python; heterogeneous base profiles|Realistic, diverse, correctly-scaled load input|Anwar §17.3(b), Sayghe §2.3 (pseudo-measurements from historical data are standard in DSSE)|
|**Data**|Feeder model|IEEE-123 OpenDSS model|3-φ unbalanced network, `Z_abc` per branch, node list, `N_phase_nodes`|OpenDSS|The `h(·)` and the ground truth|Anwar §17.3, Sayghe §2.3.1 (multi-phase modelling mandatory for distribution)|
|**Telemetry**|Ground-truth physics engine|True loads + model|`x_true` for every phase-node|OpenDSS AC solve|Layer 1; labels; impact metrics|— (definitional)|
|**Telemetry**|Sensor layer|`x_true`|`z_true = h(x_true) + e`: 49×(P,Q,\|V\|); substation head ×3φ (P,Q,\|V\|,\|I\|); 6–12 branch units; optional 1–2 PMUs|Explicit `Sensor` objects with `σ`, rate, location|Defines the attack surface precisely|Youssef §II (line meters give redundancy); Caleb §3.4.1 (smart-meter quantities); Anwar eq. 17.7 (current magnitude)|
|**Streaming**|Producer / broker|`z_true`|`telemetry.raw`|Kafka|Real-time transport; the ingest path that is itself the DT attack surface|Musleh §V (deterministic computing for real-time detection; only 10.2% of prior work tested real-time); Irfan §8.5; Caleb §3.5.5|
|**Attack**|Injector|`telemetry.raw` + scenario config|`telemetry.delivered` + `labels`|Kafka Streams processor|Implements A1–A9 as wire-level modification; emits ground truth|Lei §III-B (channel hijack); Caleb §3.5.5 (DT synchronisation-link injection)|
|**Streaming**|Consumer / time aligner|`telemetry.delivered`|Timestamp-aligned snapshot + dropout flags|Kafka consumer with window timeout|Coherent snapshots without stalling on a missing meter|Sayghe §2.3 (DS measurements often unsynchronised, infrequent); Irfan §8.7 (distinguish cyber from physical/faulty)|
|**State est.**|3-φ AC-DSSE|Snapshot + pseudo-measurements + model|`x̂`, `r = z − h(x̂)`, `J(x̂)`, LNR, per-measurement redundancy|WLS + Gauss-Newton over `h_abc`|Model-based detection arm; observability report|Anwar §17.2.3, eq. 17.9; Sayghe eqs. (5)–(13), (14)–(27); Youssef §II|
|**State est.**|Observability analyser|`H(x)`, weights|`rank`, `cond`, critical-measurement set|numpy/scipy|Proves (or disproves) observability; identifies invisible-attack measurements|Liang §V-A (basic measurement set of size `n`; `k ≥ m−n+1`); BiZhang (protected-set selection)|
|**Twin/physics**|Operator-side twin|`telemetry.delivered`|Operator's believed `V`, `I`, loading, losses|OpenDSS second instance|What the operator would act on; the impact half of the study|— (definitional; impact framing per Liang §IV)|
|**Twin/physics**|Power-balance checker|Head measurement + Σ meter P,Q + twin losses|Imbalance residual per phase|arithmetic|Cheap independent spatial constraint; catches net-load-changing attacks|Youssef §II (redundancy from line/head measurements enables bad-data identification)|
|**ML**|Temporal predictor|Per-meter windowed telemetry|One-step prediction `ŷ_k,t`|Shared-weight GRU (LSTM/TCN as ablations)|Supplies temporal redundancy where spatial redundancy is absent|Lei §III-C (GRU predicts measurement flow); ZhangYang §3.2 (LSTM over measurement history); Musleh §III-B1a (RNN family)|
|**ML**|Predicted-residual classifier|Sequence of `Γp_k,t`|Per-meter anomaly score|1-D CNN / wavelet-CNN|Classifying the residual sequence beats thresholding it|Lei §III-D (classification of predicted residuals)|
|**Detection**|Fusion + localisation|Per-meter scores ⊕ SE residual tests ⊕ balance residual|`{benign, attacked}`, suspected meter set, confidence|Rule/stacking ensemble|Layered detection; complementary blind spots; native localisation|Musleh §V (layered detection schemes; selectivity to identify affected measurements); Irfan §1 (localisation as multi-label classification)|
|**Evaluation**|Detection metrics|Decisions + labels|TPR/FPR, precision, recall, F1, detection latency, localisation accuracy|sklearn|Comparability|Irfan §7.4 (accuracy/precision/recall definitions); Musleh §IV (warns detection rates are incomparable across differing systems and attack types)|
|**Evaluation**|Physical-impact metrics|Operator twin vs. clean twin|Per-phase-node ΔV, voltage-band violations induced/masked, Δline loading, Δlosses|OpenDSS diff|Quantifies *consequence*, not just classification — the genuinely missing piece|Caleb §4.2 (vertical propagation of distribution-level falsification); Maleki §III (distribution impact is voltage-mediated)|
|**Evaluation**|Ablation harness|All of the above|Per-component contribution|scripted sweeps|Shows each component earns its place|Musleh §IV (fair comparison requires identical system and attack type)|

**Deliberately excluded, and why [R]:** market/LMP layer (no market); AGC/frequency layer (no generation dynamics); transient stability (steady-state twin); PMU-dense deployment (unjustified capex and unsupported by the papers for a feeder); topology estimator (v2); transformer telemetry (no paper support); DC-SE as a production path (only as a negative control).

---
# DELIVERABLE 8 — Literature-to-architecture traceability

|Proposed decision / component|Source paper(s)|Paper-supported concept|Our adaptation|Our novel contribution|
|-|-|-|-|-|
|**AC (not DC) modelling throughout**|Liang §III-D, §VI(1); Caleb §4.4 and §2 (ref [8]); Sayghe §2.2.2|DC-constructed attacks are impractical against an AC estimator and are easily detected; DC assumptions are a resilience blind spot; DC-effective metrics fail in AC distribution|Applied to a radial 4.16 kV feeder where the DC assumptions (negligible R, `V≈1 p.u.`, small angles) are violated by construction|Empirical demonstration that a DC-constructed `a = Hc` fails on an unbalanced AC feeder — a negative result the literature asserts but (in this set) does not show|
|**Three-phase unbalanced modelling**|Anwar §17.3; Sayghe §2.3.1 eqs. (14)–(23); Caleb §4.2|Positive-sequence modelling is invalid for LV distribution; full multi-phase with `Z_abc` required; unbalanced-feeder FDIA formulations are called for and missing|OpenDSS IEEE-123 as the `h_abc(·)` provider and ground-truth engine|Phase-resolved FDI attack construction and detection on a standard unbalanced feeder|
|**AC-FDIA construction**|Sayghe eq. (37); Liang eqs. (15)–(16); Lei eqs. (5)–(9)|`a = h(x̂_bad) − h(x̂)` preserves the residual; partitioned `a₂` form; nonlinear `a_t = Fc_t`|Adapted to `h_abc(·)`; using the **generalized** form `J(x̂_bad) < C` because Liang §III-D reports exact AC constructions are convergence-fragile|Generalized AC-FDIA on a three-phase unbalanced feeder, evaluated against a streaming detector|
|**Load-redistribution as the primary scenario**|Liang §IV-B (Yuan et al.); Youssef §III-B; Caleb Table 5|LR falsifies individual consumption while holding the aggregate constant; the delayed variant causes undetected line overloading|Applied at AMI granularity on a radial feeder rather than at transmission bus level|Quantification of LR-induced voltage-band violation and hidden loading on a real feeder model — Caleb §4.2's "vertical propagation" question, answered concretely|
|**Pseudo-measurements for unmetered nodes**|Anwar §17.3(b); Sayghe §2.1, §2.3|Pseudo-measurements from historical load data, of limited accuracy, are standard and necessary in DSSE|Unmetered IEEE-123 loads modelled as high-`σ` pseudo-measurements inside the estimator, not as silent assumptions|Explicit treatment of the metered/pseudo split as a *tunable observability parameter* rather than a fixed choice|
|**Substation-head + selective branch sensors**|Youssef §II; Anwar §17.3.1 via Sayghe eq. (17); Sayghe eq. (23)|Line meters exist to provide measurement redundancy used to validate data and identify bad points; substation P/Q/V/I and branch currents are standard DSSE measurements|Minimal set placed by the existing backbone-coverage score|Redundancy-minimal sensor augmentation for a 49-meter feeder, with the observability gain measured rather than assumed|
|**Observability / critical-measurement analysis before SE**|Liang §V-A (Bobba et al., Liu et al.); BiZhang; Sayghe §2.2.1 (convergence relies on the observability matrix rank); Liu, Ning & Reiter (2011) §3.2, Theorem 3.2|Basic measurement set has size `n`; `k ≥ m−n+1` compromised meters always admits an undetectable vector; protecting a basic set is necessary and sufficient|Rank/conditioning test on `H_abc` plus per-measurement redundancy on IEEE-123|Observability report for a representative-meter AMI configuration on an unbalanced feeder — treated as a *precondition*, not an afterthought|
|**Predicted-residual detection (LSTM/GRU placement B)**|Lei §III-C, §III-D|GRU predicts the measurement flow; `Γp_t = ‖z_t − y_t‖` is independent of currently-injected data and substitutes for the BDD residual; classify the residual *sequence*|Transferred from a 60-meter microgrid/AMI testbed to a 49-meter unbalanced feeder twin; shared-weight per-meter model for scalability|Transfer of CPR to a physics-coupled feeder twin, with the physical consequence of missed detections quantified — which Lei does not model|
|**SE-residual → LSTM (deferred arm C)**|ZhangYang §3.1–3.3|KF residual observation `o_t = ‖z_t − Hx̂_t‖/‖ω‖`, LSTM over `m` past observations, DNN classifier; motivation is that fixed thresholds trade accuracy against false alarms|Replace the linear-`H` Kalman filter with 3-φ AC WLS on a feeder|Temporal residual modelling on an *unbalanced AC distribution* estimator rather than a balanced transmission KF|
|**Layered / fused detection**|Musleh §V and Table II; Youssef §IV; Caleb §3.5.1|No algorithm is 100% secure, so layered schemes are needed; model-based and data-driven families have complementary failure modes; ML detectors underperform on unknown attacks; physics-informed hybrids address limited-observability DSSE|Exactly two complementary arms initially (temporal + power balance), with the SE arm added in phase 2|Explicit, paper-grounded complementarity argument: LR attacks are invisible to the balance check by construction and highly visible temporally, and vice versa for slow uniform drift|
|**Streaming / real-time evaluation**|Musleh §IV-A, §V; Irfan §8.5; Lei §IV-C|Only 10.2% of surveyed detection work performs real-time testing; deterministic computing is a stated criterion; the offline→online transition is an open direction; edge deployment demonstrated on ARM hardware|Kafka-based streaming twin with per-timestamp solve and detection latency measured end-to-end|Real-time FDI detection loop closed around an unbalanced feeder power-flow twin|
|**Attack injection on the transport path**|Lei §III-B; Caleb §3.5.5|Attackers hijack the measurement uplink; FDIA against digital twins works by injecting false data into the synchronisation link|Kafka stream processor between producer and consumer, emitting labels|A reproducible, wire-level FDI testbed for a distribution DT (Irfan §8.1 names ground-truth dataset scarcity as an open issue)|
|**Blind / incomplete-knowledge adversary**|RahmanMR §III–IV; Sayghe eqs. (39)–(43); Youssef §IV; Caleb §4.1|`a = Hc + δc`; perfect iff `δc = 0`; one-cut admittance knowledge suffices; SVD/PCA/ICA recover the column space of `H` from measurements alone|Radial feeder observation: every branch is a cut, so the required attacker knowledge collapses to one lateral's impedance|Feeder-topology-specific analysis of attacker knowledge requirements — sharper than the generic transmission statement|
|**Meter-placement scoring reused as attack-criticality ranking**|BiZhang; Liang §V-A; Sayghe eq. (38)|Optimal protection as a Steiner-tree problem over meters; basic-measurement-set protection; security index `α_k` = minimum meters to compromise to spoof measurement `k`|Compute `α_k` per meter on IEEE-123 and compare against the existing hybrid placement score|Joint placement–vulnerability framework: one scoring function serving observability, coverage, and attack-criticality simultaneously|
|**Physical-impact metrics (ΔV, Δloading)**|Liang §IV; Caleb §4.2; Maleki §III|FDIA impacts are voltage/loading/dispatch violations; vertical propagation from the consumer edge upward is explicitly called for; distribution impact is voltage-mediated|Per-phase-node ΔV and Δloading between clean and attacked twin runs|Direct measurement of the impact chain on a standard feeder — the step the impact literature asserts but does not quantify for distribution (see D1C)|
|**Classical BDD (LNR, `J(x̂)`, L2) as baseline**|Youssef eqs. (4),(6); Musleh eqs. (18)–(20); Sayghe §2.4; Lei Fig. 7|The standard tests, and the proof they fail against stealth FDIA|Implemented as the comparison floor, not the contribution|— (deliberately not novel; it is the control)|

---
# DELIVERABLE 9 — Research gaps and realistic novelty

## 9.1 What the literature does and does not combine

|Axis|What the set actually shows|Gap|
|-|-|-|
|Offline vs streaming| Still open. Rah26 explicitly disclaims its latency figures as *"computational throughput rather than end-to-end decision delay in a strictly online deployment"* (§IV-G); Deh25, Cui26, Zid are batch; XuIoT26 measures real communication latency on a 6LoWPAN/LoRa/MQTT testbed but for *reconstruction*, not detection. **No message-broker spine, no wire-level injection between producer and consumer anywhere in the 32**|Large — and the primary surviving axis|
|DC vs AC|Irfan Table 15 splits the reviewed corpus roughly evenly. Liang §VI(1) calls for more AC work; Caleb §4.4 calls DC a resilience blind spot; Xue argues AC is the practical case|Moderate and closing, but AC + *distribution* remains rare|
|Real multi-household load data|MAll 15 use synthetic Monte Carlo loads, except Zid (1 hr real µPMU) and Ald25 (CER Ireland, 30-min)|Open — and IDEAL at 1-min from ~232 homes is a genuine differentiator**|
|Balanced vs unbalanced|likewise. Cui26, XuIoT26, Deh25, Rah26, Sha25, XuUD26, XuDS25 all do three-phase unbalanced, most on IEEE-123 specifically|Substantially closed. Do not claim this as a gap|
|Synthetic vs realistic telemetry|Irfan §8.1: real-world measurements are extremely hard to obtain (privacy), so most data is simulated; NYISO is one of few public sources. Lei uses a real testbed + open dataset. Caleb ref [46] (Ahmed & Pathan) criticises the absence of standard datasets|Large — and it cuts *against* you: your data is synthetic. Be honest about it|
|SE vs ML|Musleh: **52.3% model-based vs 47.7% data-driven**, with complementary pros/cons (Table II)|Not a gap — an even split. Do not claim ML is under-explored|
|Physics-informed vs pure ML|Caleb §3.5.1 documents GNN+Kalman and physics-guided autoencoders for limited-observability DSSE; Maleki lists PINN-based LAA detection|Emerging; **not** unexplored. Do not overclaim novelty here|
|Static vs temporal detection|ZhangYang, Lei, Musleh's VAR/RNN entries all use temporal structure|Established. Not a gap in itself|
|FDI-only vs multiple attacks|Musleh: **79.5%** of studies consider only *basic* FDIA; stealth in only **20.5%**. Irfan §8.7: distinguishing cyber from physical is open|Moderate — attack-model *diversity* is the gap, not attack count|
|Placement/observability vs detection|XuIoT26's equivalent-electrical-distance + community-detection partitioning (eq. 1–10) is a real topology-based heuristic, but optimises subarea cohesion for reconstruction, not attack criticality. Nobody computes a minimum-unobservable-support security index for placement|Open — N2 survives across all 32|
|Digital twin + Kafka + FDIA|DT is no longer unoccupied (TF25 §III-B, Rin25, and DT-framed work throughout), but the three papers using the term mean three different things (see new §D2.5). No paper implements continuous per-timestamp ingest with transport-layer attack injection|**Narrowed to the transport layer specifically**|
|Representative-meter optimisation + FDIA|Not addressed anywhere in the set|**Open**|
|LSTM + AC-SE + OpenDSS|ZhangYang does LSTM+KF (linear, transmission); Lei does GRU without SE; Anwar §17.4 mentions OpenDSS only as a comparison baseline for a belief-propagation DSSE. **No uploaded paper combines all three<br />**|**Open**|
|AC-stealth FDIA construction on unbalanced feeders|Cui26 proves it, XuIoT26 extends it phase-resolved, Deh25 implements it|**Closed** — see A5|
|Detection at severe partial observability|Nobody works at m/n ≈ 0.18. Rah26's WLS comparator is over-determined (m/n ≈ 1.4–1.5)|Open** — do not assume Rah26/Deh25's near-ceiling AUROC transfers to our regime|
|Physical-impact quantification|Cui26 (VUF, Table XII) and XuIoT26 (VUF + THD, eq. 43) quantify *part* of the chain. Neither computes Δline loading or Δlosses; neither does a systematic per-phase-node clean-vs-attacked diff; neither uses an induced-vs-masked violation framing|Narrowed but open|


## 9.2 Five novelty directions

Each builds upward from what you already have. None requires replacing the project.

**N**N1 — Streaming FDI detection with transport-layer attack injection on an unbalanced
AC distribution twin.** *[NARROWED — rescope]* The "unbalanced AC distribution FDIA"
half is no longer a gap (§9.1). What survives is the **systems** half: a message-broker
telemetry spine, wire-level attack injection between producer and consumer, per-timestamp
solve, and detection latency measured in simulation time. Rah26 profiles latency to
*argue* online feasibility but evaluates on fixed batch splits; XuIoT26 measures real
communication latency but for reconstruction. **Neither runs a live pipeline.** Frame N1
as a systems/testbed contribution, not an algorithmic one, and state that explicitly.

**Differentiation debt:** Hallaji et al., *"A stream learning approach for real-time
identification of false data injection attacks in cyber-physical power systems,"* IEEE
TIFS 17, 2022 (cited as [49] in TF25) is explicitly stream-learning + FDI and has not
been read. **N1 cannot be finalised until it is.**


**N2 — Joint representative-meter placement and attack-surface criticality.**
Compute the security index `α_k = min ‖Hc‖₀ s.t. H(k,:)c = 1` [Sayghe eq. 38] for every metered node on IEEE-123, and compare that ranking against your existing hybrid placement score. Two publishable outcomes either way: if they correlate, your coverage heuristic is a cheap proxy for vulnerability ranking; if they diverge, you have shown that coverage-optimal metering is *not* attack-robust — a genuinely useful negative finding. Extend with BiZhang's protection framing: which subset should be hardened first. **This is the highest novelty-per-unit-effort item because the placement code already exists.**


**N3 — Quantified vertical propagation.** *[NARROWED — rescope]* Cui26 and XuIoT26
already report post-attack VUF (and THD, XuIoT26) deviation. **What remains open:** the
Δline-loading and Δlosses branches of the propagation chain, the systematic
per-phase-node ΔV diff across a clean-vs-attacked twin, and the induced-vs-masked
voltage-band-violation framing. Rescope the claim to those three specifically; cite
Cui26/XuIoT26 as partial precedent rather than claiming the whole chain.


**N4 — Attack-model stress battery: which detector fails where, and why.**
Musleh's 79.5%-basic-FDIA statistic is an indictment of evaluation practice. Run a matrix of {random, sparse, LR, DC-stealth, AC-generalized-stealth, blind/PCA, slow voltage-scaling, `S=VI`-preserving multi-parameter} × {classical BDD, predicted-residual, balance check, fusion}. The predicted headline result — that classical BDD fails on stealth, the balance check fails on LR by construction, and the temporal detector fails on blind attacks built from the same statistics — is a clean complementarity story.

**N5 — A reproducible streaming FDI benchmark for distribution feeders.**
Irfan §8.1 and §8.6 name ground-truth dataset scarcity and dataset quality as open issues; Caleb ref [46] criticises the absence of standard datasets and evaluation metrics. Release the Kafka testbed, labelled attack streams, feeder model, and metric definitions. Lowest research novelty, highest citation potential, and it costs you almost nothing beyond packaging what N1–N4 already produce.


**N6 — Meter density and placement strategy as swept parameters.** *[NEW]* Executed as
Phase-4 sub-study 4A (observability half) and Phase-12 (detection half) over a common grid of
8 meter counts × 4 placement strategies. Two results fall out that no located paper reports:
whether the hybrid placement heuristic beats random/degree/community-detection baselines at
each density, and where — if anywhere — load metering alone saturates. Ald25 fixes
~40% smart-meter coverage as a given constraint with no sensitivity sweep; nobody in the
32 treats the instrumentation fraction as a design variable. Sweeping meter count and
plotting detection AUROC against instrumentation fraction is a distinct, unclaimed angle
— and it operationalises N2 rather than leaving the security index as an assertion.
Small, cheap, and it directly answers the "does this work at realistic sparsity" question
a reviewer will ask.

## 9.3 Ranking

||Novelty|Implementation difficulty|Industry relevance|Publishability|Compatibility with existing code|
|-|-|-|-|-|-|
|**N1** streaming unbalanced AC DT detection|High|**High**|High|High|High — it *is* your pipeline, extended|
|**N2** placement ⊕ attack criticality|**High**|**Low–medium**|Medium|High|**Very high — reuses existing code**|
|**N3** vertical propagation quantification|Medium–high|Medium|**High**|Medium–high|High — needs only the second (clean) twin run|
|**N4** attack stress battery|Medium|Medium|Medium|**High** (reviewers reward it)|High|
|**N5** open benchmark|Low–medium|Low (given N1–N4)|Medium|Medium (high citations)|Very high|

**[R] Sequencing:** N2 and N3 are the cheapest real contributions and should land first; N4 is the evaluation spine that makes N1 credible; N1 is the umbrella paper; N5 is the by-product you release with it.

**[R] Do not claim:** that physics-informed detection is novel (Caleb §3.5.1 documents several); that ML for FDIA is under-explored (Musleh: ~48% of the corpus); that stealth FDIA construction is novel (Liu, Ning & Reiter, 2011, Theorem 3.1, as reported throughout and now also directly cited, §0.2); or that temporal detection is novel (ZhangYang, Lei). **Your novelty is the combination and the setting, not the primitives.** Say so plainly — reviewers respect it and punish the alternative.

---
# DELIVERABLE 10 — Implementation roadmap

Sequenced from your current code, not from scratch. Phases 0–3 are corrective and blocking; nothing downstream is valid until they pass.

---
### Phase 0 — Instrument and freeze the current twin

Task: re-run RES.md D10 Phase 0 validation against the new IDEAL corpus. This is the
blocking acceptance check for Phase 1 — Phase 1 is not complete until it passes.

The corpus is 91 per-load P,Q series at 1-minute resolution, 60 days
(2018-04-04 to 2018-06-03), generated from the IDEAL Household Energy Dataset.
It replaces the deleted Cholesky corpus entirely.

Verified structurally before handover: 91 files, 86,400 rows each, timestamps aligned
across all files, zero NaNs, aggregate mean P = 3490.0 kW and Q = 1920.0 kvar — both
exactly on nominal. Schema is timestamp, P_kW, Q_kvar. Filenames are node_<LoadName>.csv
matching the DSS Load object names.

## Setup

Repoint validation/validate_phase0.py at the new corpus directory and filename pattern.
Do not modify its logic — it is correct, it was only ever broken because the dataset it
read was deleted. If the column schema differs from what it expects, adapt the reader,
not the checks.

## Run

Full 60-day window if runtime permits; otherwise a contiguous 7-day window PLUS the
single highest-aggregate-load day, and state which you used.

Peak instantaneous feeder-head P across the corpus is 9543.2 kW = 2.73x nominal.
Locate the timestamp of that peak and include it explicitly in the run whichever window
you choose. The minimum V_pu at that timestamp is the single number that decides whether
Phase 1 passes.

Per timestamp, log: convergence flag, iteration count, V_pu min/max per phase, total
losses, feeder-head P and Q, solve wall-time.

## Acceptance — RES.md Phase 0 and Phase 1

  [ ] 100% convergence across all timestamps
  [ ] V_pu within [0.90, 1.10] at every energised node, every timestamp
  [ ] V_pu spread VISIBLY WIDER than the Phase-0 baseline of [0.980, 1.046] — a range
      this tight after the rebuild means the corpus did not take
  [ ] losses within roughly 1-5% of feeder-head load
  [ ] aggregate feeder-head P within +/-10% of 3490 kW, Q within +/-10% of 1920 kvar
  [ ] per-node mean within +/-10% of that node's own nominal kW
  [ ] solve time well under 60 s per timestamp

## Report explicitly

  - maximum instantaneous feeder-head P over the run, and its timestamp
  - minimum V_pu observed: value, node, phase, and the feeder loading at that moment
  - how many timestamps (count and percentage) breach the [0.90, 1.10] band
  - whether breaches cluster at particular nodes — if so, list them
  - peak-to-mean for the feeder aggregate, compared against the per-node range
    (4.01-7.41x, median 4.86x). Aggregation across 91 nodes should reduce this
    substantially; if it does not, the nodes are peaking in phase, which would itself
    indicate a generator problem.

## Output

reports/phase0_validation_report_ideal.txt using the standard template. Do NOT overwrite
the original Phase-0 report — the before/after comparison against the old corpus is a
Phase-1 acceptance item and both must exist.

Include a direct comparison table: old corpus vs new, for convergence rate, V_pu range,
losses %, and feeder-head P/Q against nominal.

Report per CLAUDE.md format.

### Phase 1 — Rebuild the data layer using IDEAL  ·  **STATUS: COMPLETE**

* **Goal:** physically and statistically defensible load inputs from real multi-household data.
* **Modules:** new IDEAL ingestion + aggregation generator; OpenDSS load-mapping; producer schema validation. **Delete the Cholesky generator.**
* **In:** IDEAL `electric-mains`/`electric-combined` channels + `home` metadata + auxiliary hourly propagation data; IEEE-123 nominal spot loads.
* **Out:** 91 per-load `P,Q` series (1-min) covering every `Load` object, 42 pseudo-load definitions, generator metadata.
* **Algorithm:**

  1. Filter homes: exclude home 223 (no released mains data); exclude those not covering the chosen window; exclude those failing the aggregate-coverage threshold. Record the post-filter pool size and the coverage-threshold curve that produced it.
  2. Select the time window by coverage analysis — slide candidate lengths (30/45/60/90 d) and plot window length against available home count. Target ≥60 days for a workable train/val/test time split. Exclude 2018-04-17 08:50–09:50 UTC globally (documented server error).
  3. Downsample 1 s → 1 min by **mean**, never decimation. Convert W → kW. Keep UTC throughout.
  4. Derive homes-per-node **from measured mean home power**, then tune against the correlation and spikiness diagnostics — do not hardcode (see acceptance).
  5. Build **every one of the 91 `Load` objects** by **stratified aggregation**: draw N homes without replacement within the node, stratified across `occupancy` and `hometype`; sum; apply a correction factor to that node's **own** nominal kW read from the DSS file. Generate all 91, not only the metered subset — sub-study 4A sweeps meter counts and selects a different subset at each count, and the ground-truth twin needs real values at unmetered nodes for the power flow to solve.
  5b. Handle per-home data gaps by **rescaling, not interpolation**: at each minute sum only the homes with data and rescale by the contributing count, `P_k(t) = γ_k · (N_k / n_k(t)) · Σ contributing homes`. **Persist `n_k(t)` alongside the corpus** — peak-provenance analysis and reproducibility both require it. Flag minutes where `n_k(t) < 0.7·N_k`. Filter homes on **aggregate coverage over the window**, never on worst-case single gap — over 30–60 days nearly every home has one long gap, and rejecting on that basis empties the pool.

     **[R] Retracted guidance on peak-to-mean.** An earlier draft of `IDEAL_preprocessing.md` gave 1.5–2.5× as the expected node peak-to-mean ratio. That figure was unsourced and is **wrong** for ~25 homes at 1-minute resolution — published diversified-demand curves are stated at 15-minute intervals and understate minute-scale peakiness. Measured per-node range is 4.01–7.41× (median 4.86×), and the feeder aggregate is 2.10×, confirming aggregation smooths as expected and the nodes are not peaking in phase. The corpus solves cleanly and stays in band, so this is closed — do not treat the per-node range as a defect.
  6. Derive Q: `Q_k(t) = P_k(t)·tan(φ_k^nom)`, φ from the node's own DSS nominal pf. Note the IEEE-123 model contains only **5 distinct** nominal power factors across its 91 loads — a small number of distinct tan φ values is correct here and is not evidence of a lookup-by-size error.
  7. Reject household-service-point columns at the **producer schema level** on presence, not by omission at read.
  8. Define the 42 unmetered Loads as pseudo-measurements with estimated σ_pseudo (Phase 5 amendment).
  9. Persist generator metadata: RNG seed, home-to-node map, window, correction factors, coverage threshold, achieved stratification.
* **Equations:** `P_k(t) = γ_k · Σ_{h∈S_k} p_h(t)`; `Q_k(t) = P_k(t)·tan(φ_k^nom)`.

* **Acceptance:**

  * Aggregate feeder-head **P** within ±10% of **3490 kW** nominal
  * Aggregate feeder-head **Q** within ±10% of **1170 kvar** — that is, load Q nominal (1920 kvar) **minus fixed capacitor injection (750 kvar)**: c83 at 600 kvar plus c88a/c90b/c92c at 50 kvar each, all uncontrolled shunts per model verification. **Comparing head Q against raw load Q is wrong on this feeder** — the capacitors sit upstream of the head and reduce measured reactive draw by construction. Expect measured Q to exceed 1170 by the reactive-loss margin.
  * **Per-node mean within ±10% of that node's own nominal kW** — this is the real gate. The aggregate can be right while individual nodes are wrong.
  * **Metered-subset check computed from whichever subset is selected**, not against a hardcoded figure — it varies with the 4A instrumentation fraction.
  * **Inter-node correlation decomposed, not just reported.** Fit correlation against shared-home count per node pair; report the intercept (the physical floor from shared weather and diurnal structure) separately from the overlap-driven component.
  * Per-node coefficient of variation meaningfully below a single home's (aggregation must smooth)
  * Homes-per-node N and correction factor γ reported as an explicit modelling choice, with the sensitivity sweep that justified them
  * Phase 0 checks re-run: 100% convergence; `V_pu` within [0.90, 1.10] but with **visibly wider spread than the Phase-0 [0.980, 1.046]** — a range that stays this tight means the rebuild did not take
  * Generator metadata sufficient to regenerate the corpus
  * CC BY 4.0 attribution and modification notice present in README

* **Measured result [I] — all criteria met.** Validated over the full 60-day / 86,400-timestep window (`reports/phase0_validation_report_ideal.txt`):

  | Criterion | Measured | Verdict |
  |---|---|---|
  | Convergence | 86,400 / 86,400 (100%) | PASS |
  | `V_pu` band | [0.936943, 1.099974] | PASS |
  | `V_pu` spread | 0.163 p.u. vs old 0.066 | PASS — rebuild took |
  | Losses | 2.68% of feeder-head load | PASS |
  | Feeder-head P | 3623.3 kW vs 3490 nominal (+3.8%) | PASS |
  | Feeder-head Q | 1331.9 kvar vs 1170 expected (+13.8%, reactive losses) | PASS |
  | Max solve time | 14.9 ms | PASS |

  **Gating number: minimum `V_pu` = 0.936943** at node 65.1 phase 1, 2018-05-27 08:28, clearing the 0.90 floor with margin. Zero band violations, no breach clustering. Peak feeder-head P 7591.7 kW at 2018-04-04 17:13. Feeder-aggregate peak-to-mean 2.10× against a per-node range of 4.01–7.41×, confirming aggregation smooths and the nodes are not peaking in phase.

* **Measured configuration [I]:** pool **119 homes** at a **75% aggregate-coverage threshold** (95% and 90% each returned 0 homes; full curve 85%→7, 80%→59, 75%→120, 70%→151, 65%→180). Window **60 days, 2018-04-04 to 2018-06-03**. **N = 25** homes per node, **γ mean 4.88×** (range 2.5–5.8×). Corpus covers all 91 `Load` objects; 86,400 rows each, timestamps aligned, zero NaNs.

  **Correlation decomposition:** mean pairwise inter-node P correlation **0.5605** (median 0.5554, range 0.3814–0.7775, 4095 pairs). Regressed against shared-home count: intercept **0.443**, slope **0.0221** per shared home, mean 5.31 shared homes → 0.117 overlap contribution. **79% of observed correlation is physical** (shared weather and diurnal structure, measured directly on zero-overlap pairs); 21% is an artifact of pool size. Expected overlap `N²/pool = 25²/119 = 5.25`, measured 5.31.

  **Known limitations, to state in the paper:**
  - The 119-home pool is small relative to N = 25, inflating inter-node correlation by ~0.12. Coverage threshold and N trade against each other: 65% coverage yields 180 homes and lower overlap, at the cost of more missing data per home and therefore more `N/n(t)` stretching.
  - 7.20% of node-minutes had `n(t) < 0.7·N` (worst node S79a, 7.21%), so the rescaling occasionally stretches the available homes substantially.
  - The window is **spring only** (April–June). No winter heating or summer cooling peak; the detector will not have seen seasonal extremes.

* **Basis:** Anwar §17.3(b), Sayghe §2.3 (pseudo-measurements); Youssef §IV (do not manufacture correlation); IDEAL dataset documentation.
* **Operational spec:** the step-by-step ingestion/aggregation procedure, filter thresholds, validation checklist and CC BY 4.0 attribution text live in the companion document **`IDEAL_preprocessing.md`**. RES.md defines *what must be true*; `IDEAL_preprocessing.md` defines *how to produce it*. If the two conflict, RES.md governs and the handover doc is corrected.




### Phase 2 — Define the measurement model

* **Goal:** a formal `h(·)` and an explicit sensor registry. **This is the gate on everything else.**
* **Modules:** new `sensors/` module; telemetry extractor.
* **In:** `x_true`. **Out:** `z_true = h(x_true) + e` with a machine-readable registry `{id, node, phase, quantity, σ, rate}`.
* **Algorithm:** implement `h` per quantity from Sayghe eqs. (14)–(15) (3-φ `P_i^p`, `Q_i^p`), Anwar eqs. (17.5)–(17.7) (flows, current magnitude), and voltage magnitude directly.
* **Acceptance:** every published field traces to exactly one registry entry; no OpenDSS quantity reaches Kafka without one; noise-free `h(x_true)` reproduces the OpenDSS solve to solver tolerance.
* **Basis:** Sayghe §2.3.1; Anwar §17.2.2; the Layer-1/Layer-2 rule (D2).

### Phase 3 — Restructure the streaming layer

* **Goal:** topics that can carry an attack and its ground truth.
* **Modules:** producer, new injector, consumer.
* **Out:** `telemetry.raw`, `telemetry.delivered`, `labels`, `state`.
* **Algorithm:** move the sensor layer into the producer; add a pass-through injector; add window timeout + missing-data policy to the consumer.
* **Acceptance:** with a null attack, `telemetry.delivered ≡ telemetry.raw` and end-to-end results are byte-identical to Phase 1; dropping one meter no longer stalls the pipeline; dropout events are logged distinctly from attacks.
* **Basis:** Lei §III-B; Caleb §3.5.5; Irfan §8.7; Sayghe §2.3.

### Phase 4 — Observability analysis

* **Goal:** determine, do not assume, what is estimable.
* **In:** sensor registry + model. **Out:** observability report.
* **Algorithm:** build `H_abc(x) = ∂h/∂x` at flat start; compute `dim(x)` per §4.2(a)
  (**553** — three fixed source-bus angles; recompute and confirm, do not assume),
  `rank(H)`, `cond(HᵀWH)`; sweep sensor configurations {49 P,Q} → {+|V|} → {+head} →
  {+branch} → {+zero-injection} → {+pseudo}; identify critical measurements.
* **Acceptance:** full-rank `H` with acceptable conditioning for the chosen configuration;
  the critical-measurement set is enumerated; the *minimum* configuration achieving
  observability is identified; zero-injection phase-nodes enumerated explicitly.
* **Expected:** 49×(P,Q) is far from observable (D4); zero-injection constraints are the
  decisive addition; head + branch units then supply the redundancy that makes residuals
  meaningful.
* **Basis:** Liang §V-A; Youssef §II; Sayghe §2.2.1; BiZhang; Sha25 (angle reference,
  Lagrangian zero-injection).

#### Sub-study 4A — meter-density sweep × placement strategy

1. **RESOLVED — meter count now honoured.** Per-size-class quota rounding
   undershot at 15/45/70/80/85 on this feeder's data (49 was unaffected —
   the earlier "returns 48" figure assumed a size-class split that does not
   occur here). Top-up pass added; `len(selected) == target_meters` asserted.
Corpus-independent: computed from the network model and meter placement alone, so it runs
before Phase 1 delivers load data.

* **Grid:** 8 meter counts × 4 placement strategies (hybrid, random with multiple seeds,
  degree-based, XuIoT26 community-detection).
* **Counts:** 15 / 30 / 45 / 49 / 60 / 70 / 80 / 85. **The ceiling is 85, not 91** — model
  verification found 6 of the 91 loads share buses (S49a/b/c, S65a/b/c, S76a/b/c are
  per-phase splits of three 3-phase spot loads), and the placement algorithm enforces one
  meter per bus. Counts above 85 are unreachable and would silently return 85.
* **Per-cell metrics:** `rank(H)/dim(x)`, `cond(HᵀWH)`, critical-measurement count,
  downstream-load coverage, phase coverage, security index `α_k` distribution.
    **[I] Downstream-load coverage is not a placement-quality discriminator on a radial
  feeder.** The Tier-1 sweep found it saturates to 1.00 for every deterministic strategy
  at every count — correct behaviour, since any meter near the head covers nearly all
  load below it on a tree. Retain it only as a deterministic-vs-random sanity check and
  label it as such; do not report it as a strategy comparison.
* **Expected headline result [R]:** at saturation, `m = 85 × 2 = 170` measurements against
  `dim(x) = 553` — still short by a factor of 3.3. Load metering alone cannot reach
  observability at any density. Zero-injection constraints, not meters, close the gap.
* **Phase-coverage baseline [I]:** the feeder's own distribution is A:99 / B:84 / C:95
  phase-nodes and A:1400 / B:952.5 / C:1137.5 kW. **Decide which the placement balances
  against** — node count or connected kW. Balancing on kW is the better argument for
  detection (observe where the load is); state the choice explicitly.
* **Degenerate cells to expect:** at 85 the strategies converge toward one configuration;
  at 15 the between-seed variance of random placement will be large. Report both rather
  than hiding them.

* **Meter-set construction — decide explicitly; both curves are cheap [R].**
  1. **Independent:** re-run the placement algorithm targeting each count separately.
     **This is what the current implementation does by default** — `target_meters` is a
     constructor parameter, and because `used_buses` accumulates across size classes, a
     larger quota changes what remains available to later classes, so results are **not**
     nested across counts.
  2. **Nested:** instrument the 85-meter run to record the **order** in which loads are
     appended to `selected`, then take the first *k* per count. Preserve selection order
     rather than re-sorting by score — re-sorting discards the round-robin branch-diversity
     mechanism that is the algorithm's actual contribution.
  Divergence between (1) and (2) is a property of the algorithm worth reporting. Baselines
  must use the same construction as the strategy they are compared against.

* **Phase coverage is a required metric, and the current algorithm cannot compete on it
  [I/R].** Report the phase A/B/C distribution of selected meters per cell against the
  baseline above. **Known gap:** `_balance_phases` and `_balance_connections` are no-op
  placeholders and the scoring function has no phase term, so placement is presently
  phase-blind. On an unbalanced feeder — the premise of this project — random placement may
  beat the hybrid algorithm on this metric by accident. Either implement balancing (a
  post-selection swap pass suffices) or delete the placeholders and drop the claim that the
  algorithm balances phases. **Do not run the sweep with stubs in place.**
    **[I] Design decision — the algorithm optimises for ranking validity, not for a single
  count.** Phase balancing was originally a post-hoc swap pass over the completed
  target-meters set. That produced a *set* optimal at one count but a *ranking* with no
  phase awareness at any prefix, so nested and independent construction diverged by 4–8
  percentage points. Balancing is now applied incrementally at each selection against
  running per-phase kW. Divergence fell to under 3.18pp (mostly <2pp) and does not close
  to zero — no single ranking can be simultaneously optimal at every prefix length.

  The trade-off is real and must be stated in the paper: independent-construction phase
  balance degraded at 6 of 8 counts (count=49: 0.58pp → 3.16pp). The incremental version
  is retained because 4A and Phase 12 both sweep meter count, so a ranking valid at any
  budget is the correct artifact; a set optimal only at 49 is not.

* **Three knees to locate; their disagreement is the result [R].**

  | Knee | Curve | Question | Phase |
  |---|---|---|---|
  | **Observability** | `rank(H)/dim(x)` | is the state estimable? | 4 |
  | **Placement quality** | hybrid advantage over random / degree / community-detection | *does our algorithm earn its keep?* | 4 |
  | **Detection** | AUROC at a fixed operating point | is the attack detectable? | 12 |

  The **placement-quality knee** evaluates our own contribution and is easiest to overlook.
  At very low counts every strategy picks the trunk; at saturation all converge; so the
  hybrid algorithm can only demonstrate value in the middle. **Report where that advantage
  opens and closes**, using the random baseline's between-seed distribution, not one draw.

  **If the placement-quality knee and the detection knee fall at different counts** — e.g.
  the heuristic peaks at 30 while detection keeps improving to 60 — that is a standalone
  finding: coverage-optimal placement is not detection-optimal. Same conclusion N2 reaches
  from the security-index side, arrived at independently.

* **Implementation prerequisites before running the grid [I].**
  1. **Meter count is not honoured.** Per-size-class quotas use
     `max(1, round(target · n_class / total))`; overshoot is truncated but **undershoot is
     never topped up**. Asking for 49 returns 48, and 45 returns 44. Add a top-up pass and
     assert `len(selected) == target_meters` before returning.
  2. **RESOLVED — the ceiling is 85.** Model verification confirms 85 distinct load buses
     across 91 Load objects. Counts above 85 are unreachable.
  3. **Debug output must be gated.** 8 counts × 4 strategies = 32 runs, each currently
     printing the full topology graph, all bus loads and every load's branch assignment.

### Phase 5 — Add the minimum sensor set + finalise the pseudo-measurements

* **Goal:** implement exactly what Phase 4 says is needed, and no more.
* **Out:** substation head ×3φ, 6–12 branch units, `|V|` at meters, optional 1–2 PMUs.
* **Acceptance:** Phase-4 metrics recomputed and now passing; sensor count justified line-by-line against the D4 table.
* **Basis:** Youssef §II; Anwar §17.3.1 via Sayghe eq. (17); Sayghe eq. (23).


**[R] Pseudo-measurement design: keep static, but justify it and estimate σ.**

Ald25 generates pseudo-measurements with a WaveNet-LSTM from ~40% smart-meter coverage,
solar irradiance, and **live main-branch flow currents**, on modified IEEE-123 unbalanced
— the same test system. It materially outperforms a GMM baseline on irregular loads
(MAE 0.059 vs 0.241), at 585,699 parameters and 104 s/epoch.

**Two findings change our design:**

1. **σ_pseudo must be estimated, not asserted.** Ald25 derives it from Monte Carlo
dropout at inference (rate 0.2, Gal 2016) — sample variance around the predictive
mean. A fixed hand-picked σ_pseudo is the first thing a reviewer familiar with this
line will attack. **Minimum acceptable:** empirical per-node, per-time-bucket variance
from held-out historical data. This costs almost nothing and does not require adopting
a learned generator.
2. **Our static design has an unstated security property — write it down.** Ald25's
generator consumes live branch-flow currents as an input feature. If an attacker
manipulates those upstream measurements, the generator launders attacker-influenced
signal into what the estimator treats as an *independent* pseudo-measurement. That is
a circularity channel, and Ald25 does not consider it (the paper is non-adversarial
throughout). **Our offline-computed historical pseudo-measurements are structurally
immune to it** — they neither retrain nor re-condition on live data at inference.
Frame this as a deliberate security choice, not an accuracy shortfall. Same design,
different framing, and the framing is what gets evaluated. Residual exposure: stale-
model drift, which should be acknowledged.

**Scaling caveat:** Ald25 trains on one nonconventional and one conventional bus and does
not state whether one model per node is required. At 42 unmetered loads, that
generalisation question is unresolved in the source — a further reason not to adopt the
learned approach as the primary path.

### Phase 6 — Generate the normal-operation corpus

* **Goal:** clean labelled baseline for training and for impact differencing.
* **Out:** ≥ 30 days of 1-min clean telemetry + full `x_true` + derived quantities.
* **Acceptance:** train/validation/test split by *time*, never randomly (temporal leakage would invalidate every ML result); stationarity and per-meter correlation reported.
* **Basis:** Musleh §IV-C (overfitting and training-set selection are the principal data-driven risks); Irfan §8.6.

### Phase 7 — Build the state estimator

* **Goal:** 3-φ AC WLS DSSE with classical BDD.
* **Algorithm:** Gauss-Newton on `min (z−h(x))ᵀW(z−h(x))`; `H_k = ∂h/∂x` at `x_k`; iterate to tolerance; compute `r`, `J(x̂)`, normalised residuals, LNR loop.
* **Equations:** Sayghe eqs. (5)–(8), (28)–(31); Anwar eq. (17.9); Musleh eqs. (18)–(20).
* **Acceptance:** on clean data, `‖x̂ − x_true‖_∞` within noise-consistent bounds; `J(x̂)` follows χ² with the expected DoF; the LNR loop correctly identifies a single deliberately corrupted measurement.
* **Basis:** as cited.


**[R] Amendments from the DSSE tranche:**

1. **Angle reference:** do **not** pin one angle and estimate the other two at the
reference bus (Sha25 §III: rank-deficient Jacobian; §V.A: flat-start sensitivity;
Table II: slower convergence). Fix all three source-bus angles at their known
OpenDSS `Vsource` values. Recompute and report `dim(x)` accordingly — see §4.2(a).
2. **Zero-injection:** enforce as **hard Lagrangian equality constraints** (Sha25 §II
eq. 1–6), not as high-weight pseudo-measurements. Sha25 adopts this specifically to
avoid the conditioning damage that large weight-ratio disparities cause — which is
exactly the regime we would be in with real measurements at 1% σ sitting beside
pseudo-measurements at 15%+.
3. **Numerical conditioning:** apply Sha25 eq. (6)'s scaling constant
`α = 1/max(diag(W))` to the Gauss-Newton state update. One line, directly adoptable.
4. **Expected failure mode is *not* XuUD26's.** XuUD26 §IV.C demonstrates covariance loss
of positive-definiteness triggered by *"several highly-correlated measurements with
very low uncertainty"* — too much correlated high-precision information. Ours is the
opposite: too little information plus an extreme dynamic range in `R`. Both are
ill-conditioning of `G = HᵀR⁻¹H`, so the mitigation *category* (factorised/square-root
propagation) transfers, but **XuUD26's experiment does not validate our failure mode**
— cite it as an analogy, not as evidence.
5. **Convergence expectations:** Sha25 Table II gives IEEE-123 at 4 iterations / 3.46 s
with fixed angular references. No paper reports a divergence *rate* on IEEE-123.
Sha25 §V.C notes, citing the feeder documentation itself, that the 123-node feeder is
*"noted for its consistent behavior and convergence difficulties"* — reputational,
second-hand, but worth heeding.
6. **Fallback:** if the classical arm will not converge reliably at our redundancy ratio,
the documented alternatives are learned estimators (DKL25, TAPINE, BMAML) — but none
embeds physics as rigorously as WLS+BDD, and adopting one changes what the model-based
arm *means*. Decide deliberately rather than drifting into it.

**Implementation status against the three acceptance criteria above (2026-10-10,
IEEE-123 multiphase model, 30-point stratified sweep, seed=777, n=60 calls):**

- **Feasibility — PASS.** 60/60 (100%) converge; `constraint_residual_inf` stays well
  under the 1.0 kW/kvar tolerance (mean 0.177, max 0.468) across the whole sample.
- **Accuracy — UNRESOLVED, and the spec itself has a gap.** Mean `‖x̂ − x_true‖_∞`
  (voltage magnitude) is ~4.5% pu (range 3.7%–6.25%) across all 60 points, and this
  number did **not** move under any of today's fixes (LM-damping symmetry, M042_Q and
  the broader sigma recalibration, the REGZI_83 fix) — consistent with it being a
  genuine estimation/model-fit limit, not a solver-convergence or measurement-weighting
  artifact. **Flagging an open gap in this document, not just in the implementation:**
  "within noise-consistent bounds" above was never given a numeric definition. The
  single relevant sensor sigma we do have (V_magnitude, σ=0.005 pu) makes 4.5% mean
  error ~9× that sigma, which fails under almost any plausible multiplier (0/60 pass at
  1σ, 3σ, or 5σ; only 55/60 pass at a generous, otherwise-unjustified 10σ). Until this
  document states what bound was intended, "accuracy" cannot be scored as pass/fail —
  only reported as a raw number.
- **Statistical consistency — FAILS, 0/60.** `J(x̂)` was supposed to follow χ²(dof=57)
  (95% CI ≈ [38, 80]). After the full sigma recalibration pass (every legitimate P/Q
  channel's sigma widened via `sigma_new = sqrt(sigma_old² + (sensitivity × typical
  error)²)`, plus the REGZI_83_A/B_Q fix), mean `J` dropped from ~8.6e5 to ~6.0e4 — a
  real ~14× reduction — but the mean `J`/dof ratio only fell from ~15,000× to **~1,050×**
  expected. **State plainly: this is model bias, not sensor-noise mis-specification.**
  The ~4.5% voltage error above is the direct cause — sigma recalibration can only ever
  rescale how much a fixed residual counts against a fixed noise budget; it cannot
  shrink a residual that isn't noise. **Further sigma tuning will not close this gap.**
  If pursued further, this needs to be solved at the estimator/model level (e.g. an
  actual model-fit defect, a missing network element, or a linearization limit of the
  Gauss-Newton formulation at this redundancy ratio), not by reweighting measurements.
  One known loose end if that investigation resumes: `M040_P` (bus 60) showed a
  1.15-billion× sigma-inflation ratio under the recalibration formula despite not being
  regulator-adjacent by direct check — excluded from the fix pending a real diagnosis,
  not explained by anything in this section.



### Phase 8 — Build the attack injector

**[R] Amendments:**

1. **Adopt Rah26 §II.C eq. (5)–(7) directly for A4's attack synthesis.** Verified,
published, on IEEE-123. Do not re-derive.
2. **Adopt Cui26's APF + SAL parametrisation (§III-B) for A5/A6 targeting.** Attack
preference factor + stealth attack library is a clean, citable way to express
"attacker targets specific nodes by preference." Spend the novelty budget on A6's
shadow-model mechanism instead.
3. **NEW ACCEPTANCE CRITERION — pre-evaluation stealth verification.** Before any attack
is used to stress-test a detector, confirm **empirically** that it evades classical
BDD in *our* implementation: run the full sample set through the LNR and χ² tests and
show the statistics stay below threshold, as Rah26 Fig. 1 does for its own samples.
Applies to A1, A3, A4, A5, A6 alike. Without it, "our detector caught what BDD missed"
is unsupported — a reviewer will ask whether the attack actually evaded BDD in our code
or only on paper.
4. **A3 is promoted to primary scenario** (see D6 amendment) and its detectability is an
open empirical question about our own architecture, not an assumption.

**Step 1 baseline (measured, n=330 clean):**

- Textbook thresholds: chi2 (J > 75.62) and LNR (> 3) both give 100% false alarms.
- Empirical clean J: mean 64,597; min 27,594; max 260,651; 99th pct 207,272.6. Empirical
  clean LNR: mean 98.46; max 164.51; 99th pct 158.72.
- 16 of 364 channels have zero redundancy and are excluded from LNR (10 pseudo rows plus
  real meters M004_P, M004_Q, M040_P, HEAD_Q_A/B/C). Attacks on the 6 real ones are
  residual-invisible by construction; report them separately, never as method evidence.
- Unattackable rows: 442, rank 324 of 553 states, nullity 229 (first-order, checked at 3
  points only).
- **Rule:** stealth is judged by paired dJ and dLNR against a same-size random-attack
  control, not by the empirical threshold alone.
- **Rule:** report attack size against the ~0.045 pu estimator error floor.



### Phase 9 — Temporal predictor and predicted-residual detector

* **Goal:** detection arm B.
* **Algorithm:** shared-weight GRU over a `w`-step window per meter → `ŷ_k,t`; `Γp_k,t = ‖z_k,t − ŷ_k,t‖`; 1-D CNN over the `Γp` sequence → per-meter score.
* **Equations:** Lei eqs. (10)–(15).
* **Acceptance:** clean-data RMSE reported (replicate Lei's GRU/LSTM/CNN comparison as the ablation); detection at attack magnitudes comparable to Lei's; per-meter localisation accuracy measured; **and a correlation-ablation showing performance does not collapse when the synthetic inter-meter correlation is removed.**
* **Basis:** Lei §III-C/D; ZhangYang §3.2; Musleh §III-B1a.


**[R] Amendments:**

1. **Add a cross-meter channel** (see D5 amendment). The per-meter-only design has no
spatial view; Deh25 §III.C shows 2-D spatiotemporal CNNs beating temporal-only by
17–22 pp AUC on this attack family. Minimum viable: joint 2-D convolution or a
pairwise-correlation feature over the 49-vector of `Γp` scores.
2. **Class imbalance: prefer under-sampling ensembles to synthetic oversampling.** Deh25
benchmarks an under-sampled CNN ensemble against WCGAN oversampling and the ensemble
wins (AUC 0.961/0.983 vs 0.955/0.972 on IEEE-13/123), explicitly because *"with
limited historical data, the GAN cannot generate high-quality synthetic records
effectively."* Our attack windows will be rare relative to normal streaming volume and
our labelled attack data limited — the same conditions.
3. **Do not assume near-ceiling performance transfers.** Rah26 and Deh25 report
AUROC ≈ 99.98–100% and AUC ≈ 0.96–0.98 respectively, both at near-complete
instrumentation. We are at m/n ≈ 0.18. Budget for materially lower numbers and treat
the gap as a finding rather than a failure.

**Report precision alongside TPR/FPR, at a stated attack prevalence, at a fixed
operating point.** Precision is a function of experimental design: the same detector at
90% TPR / 5% FPR yields 94.7% precision at 50% prevalence and 1.2% at one attack per
simulated day. Lead with the prevalence-independent metrics; quote precision with its base rate attached.

### Phase 10 — Physics cross-check and fusion

* **Goal:** the second, complementary arm, then the combiner.
* **Algorithm:** per phase, `Δ_head = P_head − (Σ_k P_k + P_loss^twin)`; threshold on the clean-data distribution. Fuse `{per-meter Γp scores, Δ_head, J(x̂), LNR}` → decision + suspected meter set.
* **Acceptance:** fusion strictly dominates each arm alone on the D9-N4 attack matrix; each arm's specific blind spot is demonstrated (balance check blind to LR; temporal detector weakest on blind/PCA attacks); false-alarm rate on clean data stated at a fixed operating point.
* **Basis:** Musleh §V (layered schemes, selectivity, robustness); Youssef §IV (ML detectors underperform on unknown attacks).

### Phase 11 — Impact quantification

* **Goal:** consequence, not just classification.
* **Algorithm:** diff clean twin vs. attacked twin per timestamp: per-phase-node ΔV, induced/masked voltage-band violations, Δline loading, Δlosses; report against detection latency (impact accrued before detection).
* **Acceptance:** impact reported per attack class; masked violations counted separately from induced ones.
* **Basis:** Caleb §4.2; Liang §IV; Maleki §III.

### Phase 12 — Ablations and evaluation

* **Goal:** show every component earns its place.
* **Axes:** sensor configuration; metered/pseudo ratio; meter count (49 → higher, per your design intent); predictor architecture; window length `w`; with/without balance check; with/without SE arm; correlation ablation; attack magnitude sweep.
* **Acceptance:** removing any recommended component measurably degrades a metric; if one does not, **delete it** — that is Rule 11 applied to your own results.
* **Basis:** Musleh §IV (fair comparison demands identical system and attack type — hold both fixed across ablations).


**[R] Additional required axes:**

* **Instrumentation-fraction sweep** (N6) — **reuse the Phase-4 sub-study 4A grid** (counts
  15/30/45/49/60/70/80/85 × placement strategies) and add the detection half: AUROC and AUPRC
  at a **fixed operating point**, per cell. The Phase-4 half answers *"is the state estimable?"*
  and *"does the placement algorithm earn its keep?"*; this half answers *"is the attack
  detectable?"*. All three curves share the same meter-count axis and **need not share a knee** —
  that divergence is itself the reportable result (see Phase 4, sub-study 4A). Nobody in the 31 reviewed papers sweeps instrumentation density against
  detection performance; it operationalises N2 and directly answers the sparsity question.
* **Cross-meter channel on/off** — proves the D5 amendment earns its place.
* **Homes-per-node N and correction factor γ** — the aggregation ratio is a modelling
choice with no precedent in the literature (nobody uses real multi-household data at
this scale), so its sensitivity must be reported rather than assumed.
* **Baselines to compare against, concretely:**

  * XuIoT26's equivalent-electrical-distance + community-detection partitioning
(eq. 1–10) — the defensible topology-based heuristic baseline for N2
  * Random placement and degree-based placement — the naive floors
  * Classical BDD (LNR, χ²) — the detection floor, per D3

**Consider adding IEEE-13 as a secondary feeder.** Four of the five attack-focused
papers report both 13- and 123-node results; cross-scale comparability is how that
literature demonstrates scalability. We currently test only IEEE-123.


### Phase 13 — Dashboard and packaging

* **Goal:** operator-facing view + release.
* **Out:** live feeder map with per-meter scores and per-node voltage deviation; released benchmark (N5).
* **Basis:** Musleh §V (selectivity — identify the affected measurements, not just raise a flag); Irfan §1 (localisation).

**[R] Why this ordering beats the one you sketched.** Yours starts with "stabilize the twin" then goes straight to the measurement model — correct — but it places observability *after* the measurement model without a data-correctness gate, and it adds sensors before proving they are needed. The reordering above inserts Phase 1 (data correctness) as a blocking gate, because every voltage, loss, and residual downstream is meaningless if the feeder is loaded an order of magnitude below nominal; and it makes Phase 4 (observability) *determine* Phase 5 (sensors) rather than the reverse, which is what Rule 11 requires.

---
# DELIVERABLE 11 — Final verdict

## Is the current implementation a good base?

### **PARTIALLY — with one conceptual correction that is not optional.**

**Why.** The three architectural commitments you have already made are the three the literature says are needed and mostly missing: an **unbalanced AC distribution feeder** (Anwar §17.3, Sayghe §2.3.1, Caleb §4.2 all demand it); **streaming rather than batch** (Musleh: 89.8% of prior detection work never tested real-time); and **realistically sparse metering** (Anwar §17.3(e): very low measurement redundancy is a defining property of distribution). Those are hard to retrofit and you have them. Against that: the pipeline **cannot currently detect anything**, and the reason is structural, not a missing model — a power-flow twin driven by its own measurements is an exact input→output map with no redundancy, so an attacker who moves the inputs gets a perfectly self-consistent output. Detection requires redundancy that is either spatial (`m > n`) or temporal (prediction from history). You have neither yet. Add both, in that order of cheapness: temporal first, then a small amount of spatial.

### Preserve

* OpenDSS IEEE-123 unbalanced AC model, and the decision to solve per timestamp.
* The Kafka streaming spine and timestamp synchronisation (with a timeout added).
* The hybrid representative-meter placement algorithm — and its design to scale past 49.
* The commitment to AC over DC throughout.

### Must change

1. **Stop treating the datasets' `Voltage` / `Global_intensity` / `Sub_metering_*` columns as feeder telemetry.** Voltage and current at a metered node come from the OpenDSS solve, never from the source dataset. (3.1.2 — the single most consequential correction here.)
2. **Rebuild load magnitudes by aggregation, not rescaling.** Household kW substituted for feeder spot loads under-loads **the metered subset by ~14.7×** (Phase 0 measured), appearing as only ~2× at feeder aggregate because static unmetered load dilutes it. Both figures are correct at their own denominator; the fix must be sized to the former. Reactive power additionally requires derivation from nodal power factor (household pf ≈ 0.995 vs feeder nominal ≈ 0.876), not proportional rescaling. (3.1.1)
3. **Replace the data source; delete the Cholesky generator.** Correlated copies of one household differ by *noise*, not behaviour, and a temporal predictor will learn the artifact. IDEAL (CC BY 4.0, ~232 homes) supplies genuine multi-household diversity; the generator is removed, not patched. Correlation ablation remains a Phase 9 acceptance criterion. (3.1.3; `IDEAL_preprocessing.md`)
4. **Declare the 42 unmetered Loads as pseudo-measurements with an *estimated* σ_pseudo** — empirical per-node variance from held-out history, not a hand-picked constant (Ald25). Keep them static and offline-computed, and state the reason explicitly: unlike generators conditioned on live measurements, static pseudo-measurements cannot launder attacker-influenced signal into the estimator. Distinguish these from true zero-injection nodes, which take hard Lagrangian equality constraints instead (Sha25). (3.1.4; §4.3; Phase 5)
5. **Define the measurement model before extracting telemetry.** No OpenDSS quantity reaches Kafka without a registered sensor.
6. **Insert the attack on the wire**, between producer and consumer, with ground-truth labels on a separate topic.
7. **Add a window timeout** so one missing meter cannot silently halt the twin.

### Missing

* Sensor abstraction layer (`h(·)` + noise + registry).
* Feeder-head and selective branch measurements — the redundancy that makes any residual meaningful.
* Observability analysis (`dim(x)`, `rank(H)`, critical measurements).
* Attack injector with a formal `a` and labels.
* Clean/attacked dual twin run — without it you have no labels and no impact metric.
* The detectors themselves.

### Implement next (first three phases only)

**Phase 0 is complete** (see D10 Phase 0). **Phase 0** (validate the twin) → **Phase 1** (fix load scaling and data diversity) → **Phase 2** (formal measurement model). Nothing downstream is valid until these pass, and Phase 1 will change your Phase 0 numbers substantially.

### Explicitly do NOT build

* **DC state estimation or DC-FDIA as a production path** — only as the negative control that demonstrates Caleb §4.4's blind spot.
* **Any attack that edits OpenDSS internal state variables.** That is a physical attack, not an FDIA.
* **Market / LMP attack modelling** (Xie, Esmalifalak, QZhang are transmission-market; a feeder has no LMP).
* **AGC / frequency / transient-stability layers.**
* **Load-altering attacks** — a different threat class (Maleki), not measurement integrity.
* **False command / dispatch injection** (WangCPA) — different class again.
* **Transformer telemetry as an attack surface** — no paper support in this set.
* **Dense PMU deployment.** Two PMUs at most; the 1/3-of-buses result is transmission economics, and GPS spoofing bypasses it anyway (Liang §V-B).
* **Topology / breaker-status attacks in v1.** Real and interesting; name as future work.
* **A separate LSTM per meter.** It will not scale to your stated intent to increase meter count.
* **An LSTM justified by "the project is AI-based."** The justification must remain: *it supplies the temporal redundancy that a sparsely-metered unbalanced feeder cannot supply spatially.*

## Final recommended architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ DATA          IEEE-123 OpenDSS model  ·  49 IDEAL-aggregated meter        │
│               profiles  ·  pseudo-loads (σ_pseudo) for unmetered nodes    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ GROUND TRUTH  OpenDSS AC 3-φ solve on TRUE loads  →  x_true               │──┐
└───────────────────────────────┬──────────────────────────────────────────┘  │
                                ▼                                              │
┌──────────────────────────────────────────────────────────────────────────┐  │
│ SENSORS       z_true = h(x_true) + e                                      │  │
│               49×(P,Q,|V|) · head×3φ(P,Q,|V|,|I|) · 6–12 branch · [2 PMU] │  │
└───────────────────────────────┬──────────────────────────────────────────┘  │
                                ▼                                              │
┌──────────────────────────────────────────────────────────────────────────┐  │
│ ATTACK        z_delivered = z_true + a     {A1 random · A2 sparse ·       │  │
│ INJECTOR      A3 LR · A4 DC-stealth(control) · A5 AC-generalized ·        │  │
│               A6 blind · A7 AMI P/I/S · A8 slow V-scaling · A9 line}      │  │
│               → labels                                                    │  │
└───────────────────────────────┬──────────────────────────────────────────┘  │
                                ▼                                              │
┌──────────────────────────────────────────────────────────────────────────┐  │
│ KAFKA         telemetry.raw │ telemetry.delivered │ labels │ state        │  │
│               timestamp align + window timeout + dropout flags            │  │
└───┬───────────────────────┬──────────────────────────┬───────────────────┘  │
    ▼                       ▼                          ▼                       │
┌────────────┐   ┌──────────────────────┐   ┌────────────────────────┐        │
│ TEMPORAL   │   │ AC-DSSE (3-φ WLS)    │   │ OPERATOR TWIN          │        │
│ shared GRU │   │ x̂, r, J(x̂), LNR      │   │ OpenDSS on delivered   │        │
│ Γp = ‖z−ŷ‖ │   │ + observability rpt   │   │ + head power balance   │        │
│ → CNN      │   │ [PHASE 2 ARM]        │   │   Δ_head per phase     │        │
│ per-meter s│   │                      │   │                        │        │
└─────┬──────┘   └──────────┬───────────┘   └───────────┬────────────┘        │
      └─────────────────────┼───────────────────────────┘                      │
                            ▼                                                  │
              ┌───────────────────────────────────┐                            │
              │ FUSION DETECTOR                    │                            │
              │ temporal ⊕ balance ⊕ SE residual   │                            │
              │ → decision · localisation · score  │                            │
              └───────────────┬───────────────────┘                            │
                              ▼                                                │
              ┌───────────────────────────────────┐                            │
              │ EVALUATION                         │◀───────────────────────────┘
              │ TPR/FPR/P/R/F1 · latency ·         │   clean-vs-attacked diff
              │ localisation acc · ΔV per node ·   │
              │ induced+masked V violations ·      │
              │ Δloading · full ablation matrix    │
              └────────────────────────────────────┘
```

**The one-sentence version.** Keep the unbalanced AC twin, the Kafka spine, and the meter-placement algorithm; fix the load scaling and stop publishing the source datasets' voltage and current as telemetry; add pseudo-measurements plus a feeder-head and a handful of branch sensors to create the redundancy that makes detection possible at all; put the LSTM on the per-meter telemetry stream to manufacture temporal redundancy (Lei's predicted-residual scheme) and pair it with a feeder-head power-balance check whose blind spot is exactly complementary; add the AC state estimator only after observability is proven; and build nothing else.


**[R] Post-literature-review amendment.**

The architectural commitments still hold — AC over DC, three-phase unbalanced, streaming,
realistically sparse metering — and the 2025–26 tranche largely *agrees* with them. What
changed is the novelty framing, not the architecture.

**No longer claimable:** unbalanced three-phase AC distribution FDIA as a gap; the
AC-stealth attack construction (A5) as novel; "digital twin" as unoccupied.

**Still claimable, and now more sharply:**
- Transport-layer streaming with wire-level attack injection and simulation-time
  detection latency (N1, rescoped to systems) — **pending the Hallaji et al. read**
- Joint meter placement ⊕ attack-surface criticality (N2) — unclaimed across the 31 reviewed papers
- Δline-loading / Δlosses / induced-vs-masked propagation (N3, rescoped)
- Load redistribution at AMI granularity (A3) — unclaimed, and the sharpest test of our
  own architecture
- The A6 shadow-model + radial-cut mechanism — no reference list in the 15 contains
  anything resembling RahmanMR's Theorem 1
- Real multi-household load data (IDEAL) at 1-min feeder scale — nobody else does this
- Detection at m/n ≈ 0.18 — nobody else works this sparse
- Instrumentation-fraction sweep (N6)

**The honest one-line framing:** the primitives are borrowed and should be cited as such;
the contribution is the streaming systems layer, the sparsity regime, the real-data corpus,
and the attack classes nobody has run. Say this plainly — reviewers respect it and punish
the alternative.
---