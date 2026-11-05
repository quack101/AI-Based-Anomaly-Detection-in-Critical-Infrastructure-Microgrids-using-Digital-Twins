# Topic setup before a run (kafka-init) and the results after it (report), both through
# streaming/demo_tools.py. Mount: streaming/demo_runs.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas matplotlib kafka-python
COPY streaming/__init__.py streaming/kafka_config.py streaming/windowing.py streaming/demo_tools.py streaming/detector_fit_cache.npz streaming/
CMD ["python", "-m", "streaming.demo_tools", "reset"]
