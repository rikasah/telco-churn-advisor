"""Load the champion churn model from MLflow and score customers from Postgres."""
import os

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text

from features import BOOLEAN, FEATURE_COLUMNS

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/churn_db"
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = "telco_churn_model"
MODEL_ALIAS = "champion"

_engine = create_engine(DATABASE_URL)
_model = None
_model_version = None


class CustomerNotFound(Exception):
    pass


def _load_model():
    global _model, _model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    if _model is None or _model_version != str(version.version):
        _model = mlflow.sklearn.load_model(
            f"models:/{REGISTERED_MODEL_NAME}/{version.version}"
        )
        _model_version = str(version.version)
    return _model


def get_model_info() -> dict:
    _load_model()
    return {"name": REGISTERED_MODEL_NAME, "version": _model_version}


def get_customer_row(customer_id: str) -> pd.DataFrame:
    with _engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM customers WHERE customer_id = :cid"),
            {"cid": customer_id},
        ).mappings().first()
    if row is None:
        raise CustomerNotFound(customer_id)

    data = dict(row)
    for col in BOOLEAN:
        data[col] = int(data[col])
    return pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])


def risk_level(probability: float) -> str:
    if probability >= 0.5:
        return "high"
    if probability >= 0.25:
        return "medium"
    return "low"


def predict_churn(customer_id: str) -> dict:
    model = _load_model()
    X = get_customer_row(customer_id)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "customer_id": customer_id,
        "model": get_model_info(),
        "churn_probability": round(probability, 4),
        "risk_level": risk_level(probability),
    }
