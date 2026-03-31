# Digital Twin — Microgrid Anomaly Detection

AI-based anomaly detection workflow for critical infrastructure microgrids using a Digital Twin backend and a Three.js visualization shell.

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd cappy

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create local environment file
copy .env.example .env

# 5. Run API server
uvicorn main:app --reload --port 8000
```

## Data File

- The backend reads data from `CSV_PATH` (default: `data/processed_data.csv`).
- Keep your cleaned dataset in that location or change `CSV_PATH` in `.env`.

## Backend Endpoints

| Method | Path           | Description                                |
| ------ | -------------- | ------------------------------------------ |
| GET    | `/health`      | Health check                               |
| GET    | `/api/history` | Recent anomaly logs with graceful fallback |
| WS     | `/ws/stream`   | Real-time twin frames                      |

## Observability Behavior

- Every streamed frame can be logged to Supabase table `twin_frames` (configurable).
- Anomaly events are logged to `anomaly_logs` (configurable).
- Frame logging uses an in-memory async batch policy:
  - Batch size: `FRAME_LOG_BATCH_SIZE`
  - Flush interval: `FRAME_LOG_FLUSH_INTERVAL_SECONDS`
  - Retry count: `FRAME_LOG_RETRY_COUNT`
  - Backoff base: `FRAME_LOG_RETRY_BACKOFF_SECONDS`
- If Supabase write/query fails, stream/API behavior degrades gracefully instead of crashing.

## Frontend Visualization (Step-1 Shell)

- Open `frontend/index.html` using any static server.
- Example with Python:

```bash
python -m http.server 5500
```

- Then open `http://localhost:5500/frontend/index.html`.
- The page connects to `ws://localhost:8000/ws/stream` by default.

## Environment Variables

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
CSV_PATH=data/processed_data.csv

EMA_WINDOW=30
ANOMALY_THRESHOLD=3.0
WS_FRAME_DELAY=1.0

ENABLE_FRAME_LOGGING_TO_SUPABASE=true
TWIN_FRAMES_TABLE_NAME=twin_frames
ANOMALY_LOGS_TABLE_NAME=anomaly_logs

FRAME_LOG_BATCH_SIZE=100
FRAME_LOG_FLUSH_INTERVAL_SECONDS=1.0
FRAME_LOG_RETRY_COUNT=3
FRAME_LOG_RETRY_BACKOFF_SECONDS=0.5

HISTORY_DEFAULT_LIMIT=50
HISTORY_MAX_LIMIT=500
```

## Tests

```bash
pytest -q
```

## Project Structure

```text
.
├── main.py
├── config.py
├── routes/
├── services/
├── models/
├── frontend/
│   ├── index.html
│   └── manual-checklist.md
├── tests/
└── data/
```
