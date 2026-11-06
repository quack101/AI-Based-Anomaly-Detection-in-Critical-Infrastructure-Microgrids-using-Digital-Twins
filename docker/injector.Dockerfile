# L2: the FDI seam, z_delivered = z_true + a, from telemetry.raw to telemetry.delivered
# and labels. The attack library imports OpenDSS at module level, so the package is
# installed here, but the injector only ever adds stored or computed bias to readings.
# Mount: simulation/phase6_corpus (read-only, train-split caps), streaming/demo_runs.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas scipy opendssdirect.py kafka-python
COPY 123Bus/ 123Bus/
COPY attacks/ attacks/
COPY config/ config/
COPY observability/ observability/
COPY placement/ placement/
COPY sensors/ sensors/
COPY simulation/ simulation/
COPY streaming/demo_attacks/ streaming/demo_attacks/
COPY streaming/__init__.py streaming/kafka_config.py streaming/windowing.py streaming/injector.py streaming/demo_schedule.py streaming/
CMD ["python", "-m", "streaming.injector"]
