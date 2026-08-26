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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
CV_FOLDS = 5

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


def cross_validate_estimator(estimator, X_train, y_train) -> dict:
    """Stratified k-fold CV on the training split only (X_test stays an
    untouched holdout). A single random split can land on a fold with an
    unrepresentative churn ratio -- with churn at ~26% of the data, that's
    enough to swing PR-AUC/recall by several points purely by luck of the
    split, which is what "probability tidak sesuai harapan" looks like in
    practice. Averaging across folds gives a mean +/- std that reflects the
    model's real stability on imbalanced data instead of one split's luck.
    """
    pipeline = build_pipeline(estimator)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring={"pr_auc": "average_precision", "recall": "recall"},
    )
    return {
        "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
        "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
        "cv_recall_churn_mean": float(scores["test_recall"].mean()),
        "cv_recall_churn_std": float(scores["test_recall"].std()),
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
            cv_metrics = cross_validate_estimator(cfg["model"], X_train, y_train)

            pipeline = build_pipeline(cfg["model"])
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            mlflow.log_params(cfg["params"])
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(cv_metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(
                f"[{cfg['run_name']}] cv_pr_auc={cv_metrics['cv_pr_auc_mean']:.4f}"
                f"(+/-{cv_metrics['cv_pr_auc_std']:.4f}) holdout_pr_auc={metrics['pr_auc']:.4f} "
                f"recall_churn={metrics['recall_churn']:.4f}"
            )
            results.append(
                {
                    "run_id": run.info.run_id,
                    "pr_auc": metrics["pr_auc"],
                    "cv_pr_auc_mean": cv_metrics["cv_pr_auc_mean"],
                }
            )

    # Selection uses the cross-validated mean, not the single holdout score --
    # a 5-fold average is far less sensitive to which rows happened to land in
    # one particular split than a single train/test PR-AUC number is.
    best = max(results, key=lambda r: r["cv_pr_auc_mean"])
    print(
        f"Best run this session (by CV PR-AUC): {best['run_id']} "
        f"(cv_pr_auc={best['cv_pr_auc_mean']:.4f}, holdout_pr_auc={best['pr_auc']:.4f})"
    )

    # Champion/challenger gate: only promote if the new best actually beats
    # the current champion, re-evaluated on this run's own test split so the
    # comparison is apples-to-apples even if the underlying data has grown.
    champion_pr_auc = None
    try:
        champion_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
        champion_proba = champion_model.predict_proba(X_test)[:, 1]
        champion_pr_auc = average_precision_score(y_test, champion_proba)
        print(f"Current champion re-evaluated on this test split: pr_auc={champion_pr_auc:.4f}")
    except Exception:
        print("No existing champion found -- this run's best becomes champion by default.")

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    if champion_pr_auc is None or best["pr_auc"] > champion_pr_auc:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
        print(f"Promoted {REGISTERED_MODEL_NAME} v{mv.version} to @champion "
              f"(challenger pr_auc={best['pr_auc']:.4f} > champion pr_auc={champion_pr_auc})")
    else:
        print(f"Challenger did NOT beat champion (challenger pr_auc={best['pr_auc']:.4f} "
              f"<= champion pr_auc={champion_pr_auc:.4f}) -- @champion alias unchanged. "
              f"New version v{mv.version} registered but not promoted.")


if __name__ == "__main__":
    main()
