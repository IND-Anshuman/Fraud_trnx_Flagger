# FastAPI Fraud Detector

This repository exposes an IsolationForest + XGBoost fraud detector via FastAPI.

## Quickstart

1. Create venv: `python -m venv .venv && source .venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Run server: `uvicorn fastapi_fraud_api:app --reload --port 8000`
4. Health: `GET /health`  
5. Predict: `POST /predict` with JSON array of transactions.

**Notes:** Do not commit large model artifacts; use `models/` and store them externally or use Git LFS.
