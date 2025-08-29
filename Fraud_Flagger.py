"""
fastapi_fraud_api.py

Expose the IsolationForest + XGBoost fraud detection pipeline as a FastAPI REST service.

Endpoints:
- POST /predict : Accept a JSON array of transaction objects, return fraud scores
- POST /train   : (Optional) retrain pipeline on uploaded CSV
- GET  /health  : basic health check

Run with:
uvicorn fastapi_fraud_api:app --reload --port 8000

Requires prior training and saved artifacts in the `models/` directory (see isoforest_xgboost_fraud_detector.py)

"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import shutil

from isoforest_xgboost_fraud_detector import (
    load_artifacts,
    inference,
    basic_preprocess,
    train_pipeline,
    save_artifacts
)

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model artifacts at startup
ARTIFACTS_PATH = "models"
artifacts = None
if os.path.exists(ARTIFACTS_PATH):
    try:
        artifacts = load_artifacts(ARTIFACTS_PATH)
        print("Loaded artifacts successfully")
    except Exception as e:
        print("Could not load artifacts:", e)


class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    customer_id: Optional[str] = None
    merchant: Optional[str] = None
    amount: float
    timestamp: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": artifacts is not None}


@app.post("/predict")
def predict(transactions: List[Transaction]):
    if artifacts is None:
        return {"error": "Model artifacts not loaded. Train first."}

    # Convert transactions to DataFrame
    df = pd.DataFrame([t.dict() for t in transactions])
    df = basic_preprocess(df)

    preds = inference(df, artifacts)

    return preds.to_dict(orient="records")


@app.post("/train")
def train(file: UploadFile = File(...), label_col: str = "is_fraud"):
    """Upload a CSV to retrain pipeline."""
    filepath = f"uploaded_{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(filepath)
    df = basic_preprocess(df)
    artifacts_new, _, _, _ = train_pipeline(df, label_col=label_col)

    save_artifacts(artifacts_new, path=ARTIFACTS_PATH)
    global artifacts
    artifacts = load_artifacts(ARTIFACTS_PATH)

    return {"message": "Training complete and artifacts updated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_fraud_api:app", host="0.0.0.0", port=8000, reload=True)
