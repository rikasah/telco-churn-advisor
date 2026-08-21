"""Load the champion churn model from MLflow and score customers from Postgres."""
import os

import mlflow.sklearn
import pandas as pd
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


class CustomerNotFound(Exception):
    pass


def _load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        )
    return _model


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
        "churn_probability": round(probability, 4),
        "risk_level": risk_level(probability),
    }
