# Claude Code — FDI Digital Twin: Persistent Project Instructions

Implementation agent for: **AI-Based Cyber-Physical Digital Twin for FDI Detection on an IEEE-123 Bus Unbalanced Distribution Feeder.**

`RES.md` is the sole research/architecture authority. Read the relevant section before every phase.

## Authority
- Follow RES.md's architecture, sequencing, and threat model.
- Redesign only if: the current code makes the spec impossible, an assumption is invalid, or evidence contradicts RES.md. If so — state the issue in 1–2 sentences, propose the smallest fix, continue. Never silently substitute a different design.

## Scope per turn
- Implement only the phase/task named in this message. No later phases, no unrelated fixes, no style-only refactors.
- Sequence: read RES.md section → inspect affected code → implement fully → preserve untouched functionality → run tests → fix regressions you caused → report.
- No placeholders, TODOs, fake outputs, or hardcoded results to simulate completion.

## Cyber/physical boundary (hard constraint)
```
L1 Physical truth      — OpenDSS ground-truth state
L2 Cyber measurements  — sensor output via Kafka
L3 Inference            — estimation / residuals / anomaly scores
```
- FDI = `z_delivered = z_true + a`, injected at L2 (Kafka telemetry) only.
- Never modify OpenDSS internals to simulate an attack. If an OpenDSS quantity becomes attackable, model it as: sample → noise model → publish.
- Out of scope unless a phase explicitly says otherwise: load-altering attacks, market/LMP attacks, AGC/frequency attacks, dispatch injection, adversarial-ML attacks, topology attacks, direct OpenDSS-internal edits.

## Traceability
Cite the paper/section/equation already identified in RES.md when implementing derived methods. Never invent citations.

## Respect current pipeline
```
49 synthetic datasets → representative-meter placement → Kafka producer
→ smartgrid.telemetry → consumer → timestamp buffering
→ 49 OpenDSS load updates → 1 AC solve/timestamp → digital-twin telemetry
```
Don't break this while extending it.

## Data caution
Household V/I ≠ feeder measurements — `Voltage`/`Global_intensity`/`Sub_metering_*` must be dropped from the telemetry path entirely, not merely scaled. Voltage/current at a metered node come only from the OpenDSS solve. Rescale each meter's P/Q to its node's nominal spot load, preserving shape. Diversify base profiles per meter and report a correlation ablation — don't let the imposed Cholesky correlation become a detector shortcut. Represent unmetered feeder loads explicitly as pseudo-measurements with an explicit σ_pseudo, not a silent assumption.

## Roadmap index (RES.md Deliverable 10 — authoritative; read the full section before each phase)
```
0  Instrument & validate current twin   — BLOCKING, do first
1  Fix data layer (rescale, drop V/I columns, diversify, pseudo-measurements) — BLOCKING
2  Define measurement model (sensor registry, formal h(·))
3  Restructure streaming (raw/delivered/labels/state topics, injector, window timeout)
4  Observability analysis (rank/cond of H, critical measurements)
5  Add minimum sensor set (per Phase 4 result only)
6  Generate normal-operation corpus (≥30 days, time-split)
7  Build 3-φ AC-DSSE + classical BDD
8  Build attack injector (A1–A9)
9  Temporal predictor + predicted-residual detector (primary detection arm)
10 Physics cross-check (feeder-head balance) + fusion
11 Impact quantification (ΔV, Δloading vs. clean twin)
12 Ablations
13 Dashboard & packaging
```
Do not skip 0 or 1 to reach later phases — every downstream voltage/loss/residual number is invalid until both pass their acceptance criteria.

## Communication
One short sentence before substantial tool use, otherwise silent until done. Interrupt only for: a blocking error, an architectural conflict with RES.md, or a genuine direction change.

## Report format (every phase, nothing else)
```
Implemented:
Files changed:
Run:
Result:
Known limitation / next step:
```

## Reasoning
Multi-file, multi-step work — think it through fully before editing; don't shortcut verification. Keep the final report concise regardless of how much thinking it took.
