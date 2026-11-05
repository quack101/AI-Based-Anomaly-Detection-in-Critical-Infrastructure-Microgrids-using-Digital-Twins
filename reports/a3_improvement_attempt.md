# A3: attempts to raise detection beyond the current system (5 Nov 2026)

Starting point: the combined system V | P | AE detects A3 on 69% of held-out benchmark windows, at 1.13% false alarms. In the live demo, A3 was caught, but the power peer check hovered right at its threshold.

## What was tried

Script: `validation/a3_ppeer_temporal.py`. Results: `validation/a3_ppeer_temporal_summary.json`; ratio cache and grid search in a scratch run.

All variants were compared at the **same clean false-alarm rate (1.13%)**. Thresholds were tuned on the even-id half of the benchmark and scored on the odd-id ("report") half.

| variant | idea | A3 (report half) |
|---|---|---|
| current V \| P1 \| AE | per-step power peer error | 69.0% |
| + PT (trailing means 15/30, clipped) | the V-peer design applied to P: averages away noise, keeps a persistent shift | 69.0% (at equal FAR) |
| + PL10 / PL15 / PL30 / PL60 | single long windows | 69.0–70.7% |
| + PT pair statistic | largest up-move + largest down-move (load moved between two groups) | identical to PT |
| + jump detector D / D4 (step change in peer error, 1-step alarm) | A3 switches on abruptly | no change |
| joint grid (base scale × PT × jump), tuned on the tune half | | best = current system |

PT alone, at its own 99% calibration, does catch slightly more A3 (72.4%). But it also raises false alarms to 2.85% and loses A4/A5 when spikes are clipped, so the gain is not real.

## Why nothing helps: the noise floor

- On the **missed** A3 windows, the per-step power score peaks at a median of 0.85× its threshold, and many peak below 0.7×. These misses are mostly 2–5% and 15-minute windows.
- Trailing means do not lift those windows either (median 0.75×).
- Adding the same meter's and its peers' readings from 1 h and 24 h earlier to the peer regression leaves the prediction-error spread unchanged (ratio 1.00). What remains after the peer prediction is the meters' own measurement noise.

So the smallest and shortest A3 attacks are **below the measurement-noise floor** of these meters. A better statistic on the same readings cannot separate them from clean data at about 1% false alarms. This matches the theory of stealthy attacks: a power-flow-consistent A3 with full access is only detectable through its implausibility relative to the other meters.

## What would actually move A3

These change the information available, not the detector:

1. **Protected (authenticated) meters.** If the feeder head and a few key meters cannot be tampered with, a consistent A3 must leave a mismatch at those meters. The literature shows that protecting a basis set of meters makes stealthy injection impossible (Bobba et al. 2010; Kim & Poor 2011). This is a natural extension of the meter-placement work.
2. **Moving-target defence.** Small, unpredictable changes to the network (capacitor or regulator switching, D-FACTS in transmission) make the attacker's model stale, so the injected readings stop being consistent (Morrow et al. 2012; Lakshminarayana & Yau, cost-benefit analysis of MTD).
3. **Lower-noise sensing.** Better meter accuracy (for example PMU-class channels at key buses) lowers the noise floor directly.

For the paper: report A3 at 69% (64–71% across levels), with this explanation of the limit and items 1–3 as future work.
