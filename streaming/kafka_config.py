BOOTSTRAP_SERVERS = "localhost:9092"

# RES.md D10 Phase 3 topic split -- replaces the single smartgrid.telemetry
# topic (kept below, deprecated, for reference only; nothing publishes to
# it anymore as of Phase 3).
TOPIC_RAW = "telemetry.raw"            # z_true = h(x_true) + e, from the sensor layer
TOPIC_DELIVERED = "telemetry.delivered"  # z_delivered = z_true + a, what the consumer reads
TOPIC_LABELS = "labels"                 # {timestamp, attacked_meters, attack_class, magnitude, stealth_flag}
TOPIC_STATE = "state"                   # solved operator-twin state per timestamp

TOPIC = "smartgrid.telemetry"  # deprecated (pre-Phase-3), unused
