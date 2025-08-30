import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import shutil
import io
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fraud_model import (
    load_artifacts,
    inference,
    basic_preprocess,
    train_pipeline,
    save_artifacts,
    evaluate_model
)

app = FastAPI(title="Fraud Detection API", version="1.0")

ARTIFACTS_PATH = "models"
artifacts = None

def reload_artifacts():
    global artifacts
    if os.path.exists(ARTIFACTS_PATH):
        try:
            artifacts = load_artifacts(ARTIFACTS_PATH)
            print("Loaded artifacts successfully")
        except Exception as e:
            print("Could not load artifacts:", e)

reload_artifacts()

class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    customer_id: Optional[str] = None
    merchant: Optional[str] = None
    amount: float
    timestamp: Optional[str] = None
    # NOTE: isFraud is strictly NOT expected from clients

@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": artifacts is not None}

@app.post("/predict/")
def predict_fraud(transaction: Transaction):
    if artifacts is None:
        raise HTTPException(status_code=400, detail="Model artifacts not loaded. Train first.")
    # Build dataframe, explicitly drop any accidental label columns
    data = transaction.dict()
    input_df = pd.DataFrame([data])
    # No isFraud in input, never used in prediction
    preds = inference(input_df, artifacts)
    fraud_probability = preds["xgb_proba"].iloc
    threshold = 0.3  # threshold can be parameterized
    return {
        "fraud_probability": float(fraud_probability),
        "is_flagged": bool(fraud_probability > threshold)
    }

@app.post("/model_accuracy")
async def model_accuracy(file: UploadFile = File(...), label_col: str = Query(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if label_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Label column '{label_col}' not found in uploaded file.")
        reload_artifacts()
        # Separate label and preprocess only for features
        y = df[label_col]
        X = df.drop(columns=[label_col])
        # Process features (never includes label)
        X_processed = basic_preprocess(X, drop_identifiers=False, reference_cols=artifacts["base_feature_names"])
        X_processed = inference(X_processed, artifacts)  # runs scaling, iso_score, xgb_proba
        y_pred = (X_processed["xgb_proba"] > 0.5).astype(int)
        # Compute metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "total_samples": len(y),
            "frauds_detected": int(y_pred.sum()),
            "frauds_actual": int(y.sum()),
            "missed_frauds": int(((y == 1) & (y_pred == 0)).sum()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {e}")

@app.post("/train")
async def train(file: UploadFile = File(...), label_col: str = "isFraud", sample_size: Optional[int] = None):
    """Upload a CSV to retrain pipeline."""
    try:
        filepath = f"uploaded_{file.filename}"
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        df = pd.read_csv(filepath)
        # Training occurs with label_col reserved for evaluation only
        artifacts_new, _ = train_pipeline(df, label_col=label_col, sample_size=sample_size)
        save_artifacts(artifacts_new, path=ARTIFACTS_PATH)
        reload_artifacts()
        return {"message": f"Training complete and artifacts updated (sample_size={sample_size})"}
    except Exception as e:
        import traceback
        print("Training failed:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), label_col: str = "isFraud"):
    """
    Upload a CSV test dataset, run fraud detection,
    and return overall evaluation stats.
    """
    if artifacts is None:
        raise HTTPException(status_code=400, detail="Model artifacts not loaded. Train first.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # Separate label if present
        y = None
        if label_col in df.columns:
            y = df[label_col]
            X = df.drop(columns=[label_col])
        else:
            X = df
        # preprocess for inference, consistent with feature columns
        X_processed = basic_preprocess(X, reference_cols=artifacts["base_feature_names"])
        preds = inference(X_processed, artifacts)
        preds["isFraud_pred"] = (preds["xgb_proba"] > 0.5).astype(int)
        total_cases = len(preds)
        fraud_cases = preds["isFraud_pred"].sum()
        nonfraud_cases = total_cases - fraud_cases
        response = {
            "total_cases": total_cases,
            "fraud_cases": int(fraud_cases),
            "nonfraud_cases": int(nonfraud_cases),
            "fraud_examples": preds[preds["isFraud_pred"] == 1].head(5).to_dict(orient="records")
        }
        # If label is present, add evaluation metrics
        if y is not None:
            y_pred = preds["isFraud_pred"]
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            response.update({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_fraud_api:app", host="0.0.0.0", port=8000, reload=True)

