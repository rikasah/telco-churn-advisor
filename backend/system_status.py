"""System-level status for the dashboard: data freshness, ingestion health,
champion model info, PR-AUC trend across training runs, and the current
highest-risk customers (batch-scored on demand from the live champion
model) -- all additive endpoints, none of this touches /predict or /chat.
"""
import os

import mlflow.sklearn
import pandas as pd
import requests
from mlflow.tracking import MlflowClient
from sqlalchemy import text

import model as model_module
from features import BOOLEAN, FEATURE_COLUMNS

INGESTION_URL = os.environ.get("INGESTION_URL", "http://ingestion:8001")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "telco_churn"
REGISTERED_MODEL_NAME = "telco_churn_model"


def get_data_status() -> dict:
    with model_module._engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM customers")).scalar()
        last_ingested = conn.execute(text("SELECT max(ingested_at) FROM customers")).scalar()

    ingestion_job = None
    try:
        resp = requests.get(f"{INGESTION_URL}/status", timeout=5)
        resp.raise_for_status()
        ingestion_job = resp.json()
    except requests.RequestException:
        pass

    return {
        "total_customers": total,
        "last_ingested_at": last_ingested.isoformat() if last_ingested else None,
        "last_ingestion_job": ingestion_job,
    }


def get_champion_info() -> dict | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
    except Exception:
        return None
    return {
        "version": mv.version,
        "run_name": run.data.tags.get("mlflow.runName"),
        "pr_auc": run.data.metrics.get("pr_auc"),
        "recall_churn": run.data.metrics.get("recall_churn"),
    }


def get_model_trend() -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []

    runs = client.search_runs([experiment.experiment_id], order_by=["start_time ASC"])
    trend = []
    for run in runs:
        pr_auc = run.data.metrics.get("pr_auc")
        if pr_auc is None:
            continue
        trend.append(
            {
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                "pr_auc": pr_auc,
                "start_time": run.info.start_time,
            }
        )
    return trend


def _score_all_customers() -> pd.DataFrame:
    """Batch-score every customer in the database against the live champion
    model. Shared by get_top_risk_customers and get_risk_summary so both
    the dashboard's top-risk table and the agent's aggregate-query tools
    stay consistent with each other.
    """
    model = model_module._load_model()
    with model_module._engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM customers", conn)

    for col in BOOLEAN:
        df[col] = df[col].astype(int)

    X = df[FEATURE_COLUMNS]
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["risk_level"] = df["churn_probability"].apply(model_module.risk_level)
    return df


def get_top_risk_customers(n: int = 10) -> list[dict]:
    df = _score_all_customers()
    top = df.sort_values("churn_probability", ascending=False).head(n)
    cols = ["customer_id", "churn_probability", "risk_level", "contract", "tenure", "monthly_charges"]
    return top[cols].round({"churn_probability": 4}).to_dict(orient="records")


def get_risk_summary() -> dict:
    df = _score_all_customers()
    counts = df["risk_level"].value_counts().to_dict()
    return {
        "total_customers": int(len(df)),
        "risk_counts": {level: int(counts.get(level, 0)) for level in ["high", "medium", "low"]},
    }
