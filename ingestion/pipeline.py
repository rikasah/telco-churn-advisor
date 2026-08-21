"""Fetch, clean, validate, and load the Telco churn dataset into Postgres."""
import io
import logging
import os

import pandas as pd
import requests
from sqlalchemy import create_engine, text

logger = logging.getLogger("ingestion.pipeline")

DATA_SOURCE_URL = os.environ.get(
    "DATA_SOURCE_URL",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/churn_db"
)

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen BOOLEAN,
    partner BOOLEAN,
    dependents BOOLEAN,
    tenure INTEGER,
    phone_service BOOLEAN,
    multiple_lines TEXT,
    internet_service TEXT,
    online_security TEXT,
    online_backup TEXT,
    device_protection TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    contract TEXT,
    paperless_billing BOOLEAN,
    payment_method TEXT,
    monthly_charges DOUBLE PRECISION,
    total_charges DOUBLE PRECISION,
    churn BOOLEAN,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

UPSERT_SQL = """
INSERT INTO customers (
    customer_id, gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges, churn, ingested_at
) VALUES (
    :customer_id, :gender, :senior_citizen, :partner, :dependents, :tenure,
    :phone_service, :multiple_lines, :internet_service, :online_security,
    :online_backup, :device_protection, :tech_support, :streaming_tv,
    :streaming_movies, :contract, :paperless_billing, :payment_method,
    :monthly_charges, :total_charges, :churn, now()
)
ON CONFLICT (customer_id) DO UPDATE SET
    gender = EXCLUDED.gender,
    senior_citizen = EXCLUDED.senior_citizen,
    partner = EXCLUDED.partner,
    dependents = EXCLUDED.dependents,
    tenure = EXCLUDED.tenure,
    phone_service = EXCLUDED.phone_service,
    multiple_lines = EXCLUDED.multiple_lines,
    internet_service = EXCLUDED.internet_service,
    online_security = EXCLUDED.online_security,
    online_backup = EXCLUDED.online_backup,
    device_protection = EXCLUDED.device_protection,
    tech_support = EXCLUDED.tech_support,
    streaming_tv = EXCLUDED.streaming_tv,
    streaming_movies = EXCLUDED.streaming_movies,
    contract = EXCLUDED.contract,
    paperless_billing = EXCLUDED.paperless_billing,
    payment_method = EXCLUDED.payment_method,
    monthly_charges = EXCLUDED.monthly_charges,
    total_charges = EXCLUDED.total_charges,
    churn = EXCLUDED.churn,
    ingested_at = now();
"""

YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


def fetch_raw() -> pd.DataFrame:
    resp = requests.get(DATA_SOURCE_URL, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def validate_raw(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df.empty:
        raise ValueError("source dataset is empty")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customerID", keep="last").copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    dropped = before - len(df)
    if dropped:
        logger.info("dropped %d rows with invalid TotalCharges", dropped)

    df = df[(df["tenure"] >= 0) & (df["MonthlyCharges"] >= 0)]

    for col in YES_NO_COLUMNS:
        df[col] = df[col].map({"Yes": True, "No": False})

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

    df = df.rename(
        columns={
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churn",
        }
    )
    return df


def load(df: pd.DataFrame) -> int:
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        records = df.to_dict(orient="records")
        for record in records:
            conn.execute(text(UPSERT_SQL), record)
    engine.dispose()
    return len(records)


def run() -> dict:
    logger.info("ingestion run started (source=%s)", DATA_SOURCE_URL)
    raw = fetch_raw()
    validate_raw(raw)
    cleaned = clean(raw)
    n = load(cleaned)
    logger.info("ingestion run finished: %d rows upserted", n)
    return {"rows_upserted": n}
