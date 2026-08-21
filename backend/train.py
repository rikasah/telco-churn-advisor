"""Train churn models on data from Postgres, log runs to MLflow, register
the best one (by PR-AUC) as the champion in the Model Registry.
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

from features import BOOLEAN, CATEGORICAL, NUMERIC, TARGET

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/churn_db"
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "telco_churn"
REGISTERED_MODEL_NAME = "telco_churn_model"

MODEL_CONFIGS = [
    {
        "run_name": "logreg_balanced",
        "model": LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0),
        "params": {"model_type": "LogisticRegression", "class_weight": "balanced", "C": 1.0},
    },
    {
        "run_name": "logreg_balanced_c01",
        "model": LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1),
        "params": {"model_type": "LogisticRegression", "class_weight": "balanced", "C": 0.1},
    },
    {
        "run_name": "random_forest_200",
        "model": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "params": {"model_type": "RandomForestClassifier", "n_estimators": 200, "class_weight": "balanced"},
    },
    {
        "run_name": "random_forest_500_depth10",
        "model": RandomForestClassifier(
            n_estimators=500, max_depth=10, class_weight="balanced", random_state=42
        ),
        "params": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 10,
            "class_weight": "balanced",
        },
    },
    {
        "run_name": "gradient_boosting",
        "model": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "params": {"model_type": "GradientBoostingClassifier", "n_estimators": 200},
    },
]


def load_data() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM customers", engine)
    engine.dispose()
    return df


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": average_precision_score(y_test, y_proba),
        "recall_churn": recall_score(y_test, y_pred, pos_label=1),
        "precision_churn": precision_score(y_test, y_pred, pos_label=1),
        "f1_churn": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = load_data()
    df[TARGET] = df[TARGET].astype(int)
    for col in BOOLEAN:
        df[col] = df[col].astype(int)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    for cfg in MODEL_CONFIGS:
        with mlflow.start_run(run_name=cfg["run_name"]) as run:
            pipeline = build_pipeline(cfg["model"])
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_params(cfg["params"])
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"[{cfg['run_name']}] pr_auc={metrics['pr_auc']:.4f} recall_churn={metrics['recall_churn']:.4f}")
            results.append({"run_id": run.info.run_id, "pr_auc": metrics["pr_auc"]})

    best = max(results, key=lambda r: r["pr_auc"])
    print(f"Best run: {best['run_id']} (pr_auc={best['pr_auc']:.4f})")

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
    print(f"Registered {REGISTERED_MODEL_NAME} v{mv.version} as @champion")


if __name__ == "__main__":
    main()
