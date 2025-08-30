import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------
# Preprocessing
# -------------------------
def basic_preprocess(df: pd.DataFrame, drop_identifiers=True, reference_cols=None, label_col="isFraud") -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Drop identifier columns
    for col in ["nameOrig", "nameDest", "transaction_id", "customer_id", "merchant", "timestamp"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop label column if present (avoid data leakage)
    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    # One-hot encode 'type' if present
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Convert numerics
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encode other categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill NaNs
    df = df.fillna(0)

    # Align with reference columns if provided
    if reference_cols is not None:
        missing_cols = set(reference_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[reference_cols]

    return df

# -------------------------
# Feature Alignment
# -------------------------
def align_features_for_inference(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    X = df.copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0.0
    extra = [c for c in X.columns if c not in feature_list]
    if extra:
        X = X.drop(columns=extra)
    X = X[feature_list]
    return X

# -------------------------
# Training Pipeline
# -------------------------
def train_pipeline(df: pd.DataFrame,
                  label_col: str = "isFraud",
                  iso_contamination: float = 0.005,
                  sample_size: int = None,
                  test_size: float = 0.2,
                  random_state: int = 42):
    # Optionally sample
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # Separate label (do before preprocessing)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    y = df[label_col].astype(int).values

    # Preprocess (does NOT use label_col for any processing)
    X_all = basic_preprocess(df, label_col=label_col)

    # Ensure all columns are numeric before scaling
    non_numeric = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        for col in non_numeric:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
        still_non_numeric = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
        if still_non_numeric:
            X_all = X_all.drop(columns=still_non_numeric)
    X_all = X_all.fillna(0)

    base_feature_names = X_all.columns.tolist()

    # Standard scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=iso_contamination, random_state=random_state)
    iso.fit(X_scaled)
    raw_scores = iso.score_samples(X_scaled)
    iso_score = -raw_scores

    # Add iso_score as a feature
    X_model = pd.DataFrame(X_scaled, columns=base_feature_names, index=df.index)
    X_model["iso_score"] = iso_score

    # Train/test split
    stratify = y if len(np.unique(y)) > 1 and (y.sum() > 1 and (len(y) - y.sum()) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, stratify=stratify, random_state=random_state)

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state
    }
    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtrain, "train"), (dtest, "eval")],
                    early_stopping_rounds=20, verbose_eval=False)

    # Evaluate
    y_pred_proba = bst.predict(dtest)
    auc_score = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else float("nan")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    eval_results = evaluate_model(bst, X_test, y_test)

    artifacts = {
        "scaler": scaler,
        "isolation_forest": iso,
        "xgboost_model": bst,
        "feature_list": list(X_model.columns),
        "base_feature_names": base_feature_names,
        "training_metrics": {
            "roc_auc": float(auc_score),
            "pr_auc": float(pr_auc),
            **{k: float(v) for k, v in eval_results.items() if isinstance(v, (int, float))}
        }
    }

    return artifacts, (X_test, y_test, y_pred_proba, eval_results)

# -------------------------
# Inference
# -------------------------
def inference(df_new: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df_new.copy()
    df = basic_preprocess(df, label_col=None)  # Don't remove label here unless present

    # Align to base features then scale
    base_names = artifacts["base_feature_names"]
    X_base = pd.DataFrame({c: df.get(c, 0.0) for c in base_names}, index=df.index)
    X_scaled = artifacts["scaler"].transform(X_base)

    # Iso scoring
    iso = artifacts["isolation_forest"]
    raw_scores = iso.score_samples(X_scaled)
    iso_score = -raw_scores

    # Build model input aligning with training feature order
    X_model = pd.DataFrame(X_scaled, columns=base_names, index=df.index)
    X_model["iso_score"] = iso_score

    # Reorder/align columns as used during training
    X_model = align_features_for_inference(X_model, artifacts["feature_list"])

    dmat = xgb.DMatrix(X_model)
    proba = artifacts["xgboost_model"].predict(dmat)

    out = df.copy().reset_index(drop=True)
    out["iso_score"] = iso_score
    out["xgb_proba"] = proba

    return out

def evaluate_model(bst, X_test, y_test):
    """
    Evaluate fraud detection model using accuracy, precision, recall, f1.
    """
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = bst.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return results

# -------------------------
# Save & Load
# -------------------------
def save_artifacts(artifacts, path="models"):
    os.makedirs(path, exist_ok=True)

    joblib.dump(artifacts["scaler"], os.path.join(path, "scaler.pkl"))
    joblib.dump(artifacts["isolation_forest"], os.path.join(path, "iso.pkl"))
    artifacts["xgboost_model"].save_model(os.path.join(path, "xgboost.json"))
    joblib.dump(artifacts["feature_list"], os.path.join(path, "feature_list.pkl"))
    joblib.dump(artifacts["base_feature_names"], os.path.join(path, "base_feature_names.pkl"))

    if "training_metrics" in artifacts:
        joblib.dump(artifacts["training_metrics"], os.path.join(path, "metrics.pkl"))

def load_artifacts(path="models"):
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    iso = joblib.load(os.path.join(path, "iso.pkl"))
    bst = xgb.Booster()
    bst.load_model(os.path.join(path, "xgboost.json"))

    feature_list = joblib.load(os.path.join(path, "feature_list.pkl"))
    base_feature_names = joblib.load(os.path.join(path, "base_feature_names.pkl"))

    metrics_path = os.path.join(path, "metrics.pkl")
    training_metrics = {}
    if os.path.exists(metrics_path):
        training_metrics = joblib.load(metrics_path)

    artifacts = {
        "scaler": scaler,
        "isolation_forest": iso,
        "xgboost_model": bst,
        "feature_list": feature_list,
        "base_feature_names": base_feature_names,
        "training_metrics": training_metrics
    }
    return artifacts
