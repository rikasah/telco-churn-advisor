"""Feature schema shared between train.py and the serving code in app.py."""

CATEGORICAL = [
    "gender", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract", "payment_method",
]
BOOLEAN = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]
NUMERIC = ["tenure", "monthly_charges", "total_charges"] + BOOLEAN
TARGET = "churn"

FEATURE_COLUMNS = CATEGORICAL + NUMERIC
