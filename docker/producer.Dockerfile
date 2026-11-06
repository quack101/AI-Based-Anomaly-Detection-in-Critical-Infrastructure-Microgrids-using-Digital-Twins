# L1 + L2: the OpenDSS ground-truth solve and the sensor layer, publishing telemetry.raw.
# Mount: dataset/node_loads_91 (read-only), streaming/demo_runs.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas opendssdirect.py kafka-python
COPY 123Bus/ 123Bus/
COPY config/ config/
COPY sensors/ sensors/
COPY simulation/ simulation/
COPY streaming/__init__.py streaming/kafka_config.py streaming/producer.py streaming/
CMD ["python", "-m", "streaming.producer"]
