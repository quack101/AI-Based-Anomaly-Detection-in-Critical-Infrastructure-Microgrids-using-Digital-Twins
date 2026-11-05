# Containers and fault-tolerant Kafka

The live system now starts with one command, and its stream survives the loss of a Kafka broker. With a broker hard-killed twice during a full A1-A5 run, no message was lost or duplicated and detector scores matched an offline re-score to floating-point precision.

```
docker compose up --build
```

## What changed

| Before | Now |
|---|---|
| One broker, `apache/kafka:latest` | Three brokers and three controllers, `apache/kafka:4.3.2` pinned by digest |
| One partition per topic, one copy of each | Telemetry 6 partitions, labels/alarms/state 3; every partition on all three brokers, `min.insync.replicas=2` |
| Default producer acknowledgement | `acks=all` with idempotence; a record that cannot be delivered stops the stage |
| Stages started by a host script | One Dockerfile per stage, in `docker/`, with pinned package versions |
| Windows closed by a 5 s timer | Windows closed by what has been read (`streaming/windowing.py`) |

The window change was required by the partitions. One timestep now spreads over six partitions, and one of them can pause during a failover. A timer cannot tell that pause from a dropped meter, so it would close the window early and score an incomplete step. The direct `ingest()` path and its timeout are unchanged, and the existing tests still cover them.

## Results

Both are from `streaming/docker-compose.yml`: start 80440, 520 steps, S1 noise, all five attack classes.

| | No failures (`runC_docker`) | Two broker kills (`runC_failover`) |
|---|---|---|
| Steps scored, gaps | 511, 0 | 511, 0 |
| `telemetry.raw` / `telemetry.delivered` messages | 26,000 / 26,000 | 26,000 / 26,000 |
| Duplicates, steps missing a message | 0, 0 | 0, 0 |
| Channels changed on clean steps | 0 | 0 |
| Changed channels outside the labelled targets | 0 | 0 |
| Clean-stretch false alarms | 0/126 | 0/126 |
| Attack classes detected | 5/5 | 5/5 |

**The kills.** `docker compose kill`, no clean shutdown, 45 s down each:

1. `kafka-2`, the injector group's coordinator, at step 166 while the producer and injector were writing. The stream reached step 341 before the broker came back.
2. `kafka-3`, the detector group's coordinator, at alarm 317 while the detector was catching up.

All partitions were back to three in-sync replicas 6-7 s after each restart. The record is in `streaming/demo_runs/runC_failover/broker_kill.json`.

**Did the kills affect detection?** No. Re-scoring that run's delivered telemetry offline, in timestamp order and with no Kafka, gave the same result as the live detector. The largest score difference was 7e-14, and no alarm flag differed.

**Re-reads after a failover.** In the kill run the injector dropped 4,050 messages and the operator twin 5,880 as already-processed. These are re-reads after the group coordinator died, not losses: the message counts above are exact, and the no-failure run dropped none.

**Scratch-topic check.** `python -m streaming.failover_check` kills a broker under load on a scratch topic. It passed with 3,000 sent and 3,000 received, with none missing, duplicated or out of order (`reports/kafka_failover_check.json`).

**Two brokers down.** Two of the three brokers were killed under load, on a scratch topic, using the stages' own publishing code. This loses both the write quorum and the controller quorum.

| Outage | Longest write wait | Outcome |
|---|---|---|
| 60 s | 76 s | Resumed by itself; 15,000 of 15,000 messages, none missing, duplicated or out of order |
| 150 s | 155 s | Producer stopped with `KafkaTimeoutError` at the 120 s delivery timeout; every acknowledged message was still there |

**Presentation setup.** A third full run used the containers with `python -m streaming.live_view` on the host, killing `kafka-3`, the coordinator of the viewer's own consumer group, at alarm 152. The window stayed up and reached the last step, and the run scored 511/511 steps with no gaps and 0/126 clean-stretch false alarms. All five attack classes were detected.

## Detection compared with the committed host run

| Class | `runC_all` (committed, Windows host) | `runC_docker` (containers) | `runC_failover` (containers) | Rehearsal (containers) |
|---|---|---|---|---|
| A1 delay, alarm steps | 0, 60 | 0, 60 | 0, 60 | 0, 59 |
| A2 delay, alarm steps | 17, 43 | 16, 41 | 16, 44 | 23, 37 |
| A3 delay, alarm steps | 3, 27 | 4, 13 | 5, 10 | 6, 6 |
| A3 caught by | fusion, V-peer, P-peer | P-peer | P-peer | P-peer |
| A4 delay, alarm steps | 0, 30 | 0, 30 | 0, 30 | 0, 30 |
| A5 delay, alarm steps | 0, 30 | 0, 30 | 0, 30 | 0, 30 |

A3 is weaker in all three container runs: 13, 10 and 6 alarm steps of 60, always from P-peer alone, compared with 27 steps and all three detectors in the committed host run. `reports/demo_full.md` already records A3 as the weakest case, with an earlier host run at 9/60 from P-peer alone, so the 6 is the lowest seen so far.

- **The broker kills are not the cause.** The re-score above matched the live detector.
- **The detector does not differ between platforms.** Scoring the `runC_docker` telemetry on Windows matched its Linux scores, with a largest difference of 2e-7 in the fusion score and no alarm flag different, including 13/60 on A3.
- **A3 sits at the threshold.** In `runC_docker` the P-peer score averaged 0.99 of its threshold across the A3 window, and V-peer peaked at 1.005. Small changes in the telemetry can therefore change the alarm count a lot, which fits the spread of 6-13 across container runs with unseeded noise.
- **Not ruled out: the telemetry itself.** The sensor noise is unseeded, and the Linux and Windows OpenDSS solves have not been compared directly, so noise and a producer-side platform difference cannot yet be separated. A noise-free run on each platform would settle it. `runC_all` kept no `alarms.csv`, so its per-step scores cannot be compared either.

Quote the benchmark numbers for A3, as `reports/demo_full.md` already advises.

## Limits

- **One broker down is the design limit.** With two down, the pipeline pauses for up to 120 s. A longer outage stops the producing stage, and the run has to be started again.
- **Broker failure only.** A stage that crashes is restarted (the producer excepted), but it loses its in-memory window and the detector's history, so some steps go unscored. This was not tested.
- **The cluster runs on one machine.** It demonstrates broker failure, not machine failure.
- **No live window from containers.** The `viewer` container records a GIF because it has no display. Run `python -m streaming.live_view` on the host for the window.
- **Detector speed sets the run length.** The producer finishes in about 2 minutes and the detector catches up at just under 2 s per step, so its alarms trail the stream by minutes in a default run.
