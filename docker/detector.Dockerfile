# L3: state estimator + BDD, AE-Mahalanobis and the peer checks, publishing alarms. The
# frozen model files are part of the image; the first run fits the peer detectors from
# dataset/fdia_v1 and keeps them on a named volume.
# Mount: dataset/fdia_v1, dataset/node_loads_91 and simulation/phase6_corpus (read-only),
# the fit-cache volume.
FROM python:3.14.6-slim
WORKDIR /app
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt numpy pandas scipy opendssdirect.py kafka-python
COPY 123Bus/ 123Bus/
COPY config/ config/
COPY estimation/ estimation/
COPY observability/ observability/
COPY placement/ placement/
COPY sensors/ sensors/
COPY simulation/ simulation/
COPY validation/__init__.py validation/detect_ae.py validation/detect_eval.py validation/detect_mahalanobis.py validation/detect_a2_peer.py validation/detect_peer_power.py validation/detect_ae_weights.npz validation/detect_final_summary.json validation/
COPY streaming/__init__.py streaming/kafka_config.py streaming/windowing.py streaming/detector_consumer.py streaming/detector_core.py streaming/peer_monitor.py streaming/detector_fit_cache.npz streaming/
CMD ["python", "-m", "streaming.detector_consumer"]
