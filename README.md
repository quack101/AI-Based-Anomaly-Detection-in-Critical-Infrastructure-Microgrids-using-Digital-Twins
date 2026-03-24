# Digital Twin — Microgrid Anomaly Detection

AI-Based Anomaly Detection in Critical Infrastructure Microgrids using Digital Twins.

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd cool

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env with your Supabase credentials

# 5. Place the UCI dataset
# Download from: https://archive.ics.uci.edu/dataset/235
# Save as: data/household_power_consumption.csv

# 6. Run the server
uvicorn main:app --reload --port 8000
```

## Project Structure

```
├── main.py                    # FastAPI app entry point
├── config.py                  # Environment settings (Pydantic)
├── requirements.txt
├── .env.example
├── models/
│   ├── sensor_data.py         # SensorReading model (UCI dataset row)
│   └── twin_state.py          # TwinState model (WebSocket JSON schema)
├── services/
│   ├── data_ingestion.py      # CSV loader [Track A]
│   ├── twin_engine.py         # Digital Twin engine [Track A]
│   ├── supabase_client.py     # Supabase connection [Track B]
│   └── logger.py              # Anomaly log writer [Track B]
├── routes/
│   ├── api.py                 # REST endpoints [Track B]
│   └── ws.py                  # WebSocket endpoint [Track B]
├── tests/
└── data/                      # UCI dataset (not tracked by git)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/history` | Recent anomaly logs |
| WS | `/ws/stream` | Real-time twin state stream |

## Work Split

- **Track A** (Data + Engine): `data_ingestion.py`, `twin_engine.py`
- **Track B** (API + Infra): `supabase_client.py`, `logger.py`, `api.py`, `ws.py`

## Team

- Digital Twin Engine + Three.js Visualization: [Your Name]
- ML Layer (LSTM / RL): [Teammate Names]
