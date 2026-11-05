"""
Kafka settings shared by every stage: where the brokers are, how each topic is
laid out, and how a stage publishes so that a broker failure loses nothing.

The cluster is three brokers (streaming/docker-compose.yml). A stage on the host
reaches them through the three published ports below; a stage in a container gets
the internal listeners through KAFKA_BOOTSTRAP_SERVERS.
"""

import os

BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092,localhost:9094,localhost:9096"
).split(",")

# RES.md D10 Phase 3 topic split -- replaces the single smartgrid.telemetry
# topic (kept below, deprecated, for reference only; nothing publishes to
# it anymore as of Phase 3).
TOPIC_RAW = "telemetry.raw"            # z_true = h(x_true) + e, from the sensor layer
TOPIC_DELIVERED = "telemetry.delivered"  # z_delivered = z_true + a, what the consumer reads
TOPIC_LABELS = "labels"                 # {timestamp, attacked_meters, attack_class, magnitude, stealth_flag}
TOPIC_STATE = "state"                   # solved operator-twin state per timestamp
TOPIC_ALARMS = "alarms"                 # {timestamp, fusion_score, alarm, top_channel, J, maxLNR} from the detector

TOPIC = "smartgrid.telemetry"  # deprecated (pre-Phase-3), unused

# Every partition has a copy on all three brokers. A write is acknowledged once
# two copies hold it, so one broker can die with no data loss and no write outage;
# with two brokers down, writes wait rather than risk being lost.
REPLICATION_FACTOR = 3
MIN_INSYNC_REPLICAS = 2

# Telemetry is keyed by meter_id, so each meter's readings stay in order. The 49
# meters + HEAD hash onto all six partitions, none left empty (checked by
# tests/test_kafka_cluster.py). Labels, alarms and state are keyed by timestamp;
# every reader indexes or sorts them by timestamp, so order across partitions
# does not matter.
TOPIC_PARTITIONS = {
    TOPIC_RAW: 6,
    TOPIC_DELIVERED: 6,
    TOPIC_LABELS: 3,
    TOPIC_ALARMS: 3,
    TOPIC_STATE: 3,
}


def new_topics():
    """The NewTopic definitions of every live topic."""
    from kafka.admin import NewTopic
    return [
        NewTopic(name, partitions, REPLICATION_FACTOR,
                 topic_configs={"min.insync.replicas": str(MIN_INSYNC_REPLICAS)})
        for name, partitions in TOPIC_PARTITIONS.items()
    ]


def reliable_producer(**config):
    """A KafkaProducer that survives a broker failure: acks="all" waits for the
    in-sync replicas, and idempotence stops a retry after a failover from
    duplicating or reordering a message."""
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",
        enable_idempotence=True,
        **config,
    )


def send_all(producer, records):
    """Publish (topic, key, value) records in order and wait until every one is
    acknowledged. A record that cannot be delivered raises, so a stage stops
    instead of silently continuing with a gap in the stream."""
    futures = [producer.send(topic, key=key, value=value) for topic, key, value in records]
    producer.flush()
    for future in futures:
        future.get()
