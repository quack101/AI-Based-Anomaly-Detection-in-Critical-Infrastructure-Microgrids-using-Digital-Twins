# L3c operator twin: applies delivered telemetry to its own feeder model and publishes
# each AC solve to the state topic. Never sees ground truth.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas opendssdirect.py kafka-python
COPY 123Bus/ 123Bus/
COPY config/ config/
COPY sensors/ sensors/
COPY simulation/ simulation/
COPY streaming/__init__.py streaming/kafka_config.py streaming/windowing.py streaming/consumer.py streaming/
CMD ["python", "-m", "streaming.consumer"]
