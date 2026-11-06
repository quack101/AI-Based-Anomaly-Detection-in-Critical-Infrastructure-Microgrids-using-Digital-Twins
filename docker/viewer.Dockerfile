# Read-only viewer of alarms + labels. Headless in a container: it records a GIF.
# Mount: streaming/demo_runs.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas matplotlib pillow kafka-python
COPY streaming/__init__.py streaming/kafka_config.py streaming/live_view.py streaming/detector_fit_cache.npz streaming/
CMD ["python", "-m", "streaming.live_view", "--headless", "--save-gif", "streaming/demo_runs/live_demo.gif"]
