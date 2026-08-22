"""SHAP-based explanation for a single customer's churn prediction.

Treats the full sklearn Pipeline (preprocessing + classifier) as a black box
so SHAP values come back in terms of the original raw features
(e.g. "contract", "tenure") instead of one-hot encoded columns -- these are
what actually get shown to the user via the explain_prediction agent tool.

SHAP's tabular masker needs an all-numeric matrix, so categorical columns are
encoded to integer codes here and decoded back to their original string
values inside predict_fn before handing rows to the real sklearn pipeline.
"""
import numpy as np
import pandas as pd
import shap

from features import BOOLEAN, CATEGORICAL, FEATURE_COLUMNS, NUMERIC
from model import _engine, _load_model, get_customer_row

N_BACKGROUND = 50
MAX_EVALS = 100

FEATURE_LABELS = {
    "gender": "jenis kelamin",
    "multiple_lines": "multiple lines",
    "internet_service": "jenis layanan internet",
    "online_security": "online security",
    "online_backup": "online backup",
    "device_protection": "device protection",
    "tech_support": "tech support",
    "streaming_tv": "streaming TV",
    "streaming_movies": "streaming movies",
    "contract": "jenis kontrak",
    "payment_method": "metode pembayaran",
    "tenure": "lama berlangganan (bulan)",
    "monthly_charges": "biaya bulanan",
    "total_charges": "total biaya sejak awal",
    "senior_citizen": "status lansia",
    "partner": "punya pasangan",
    "dependents": "punya tanggungan",
    "phone_service": "layanan telepon",
    "paperless_billing": "paperless billing",
}

_background_raw = None
_categories: dict[str, list] = {}
_explainer = None


def _get_background_raw() -> pd.DataFrame:
    global _background_raw
    if _background_raw is None:
        df = pd.read_sql(
            f"SELECT * FROM customers ORDER BY random() LIMIT {N_BACKGROUND}", _engine
        )
        for col in BOOLEAN:
            df[col] = df[col].astype(int)
        _background_raw = df[FEATURE_COLUMNS].reset_index(drop=True)
    return _background_raw


def _encode(df: pd.DataFrame) -> np.ndarray:
    """Raw feature DataFrame -> all-numeric matrix (categoricals as codes)."""
    cols = []
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL:
            codes = _categories[col]
            cols.append(df[col].map({v: i for i, v in enumerate(codes)}).astype(float))
        else:
            cols.append(df[col].astype(float))
    return np.column_stack(cols)


def _decode(x: np.ndarray) -> pd.DataFrame:
    """Numeric matrix -> raw feature DataFrame the pipeline expects."""
    data = {}
    for j, col in enumerate(FEATURE_COLUMNS):
        if col in CATEGORICAL:
            codes = _categories[col]
            idx = np.clip(np.round(x[:, j]).astype(int), 0, len(codes) - 1)
            data[col] = [codes[i] for i in idx]
        else:
            data[col] = x[:, j]
    return pd.DataFrame(data)


def _get_explainer():
    global _explainer
    if _explainer is None:
        background = _get_background_raw()
        for col in CATEGORICAL:
            _categories[col] = sorted(background[col].unique().tolist())

        model = _load_model()

        def predict_fn(x: np.ndarray) -> np.ndarray:
            df = _decode(x)
            return model.predict_proba(df)[:, 1]

        background_encoded = _encode(background)
        _explainer = shap.explainers.Permutation(predict_fn, background_encoded, seed=42)
    return _explainer


def explain_churn(customer_id: str, top_n: int = 4) -> dict:
    X = get_customer_row(customer_id)
    explainer = _get_explainer()
    x_encoded = _encode(X)
    shap_values = explainer(x_encoded, max_evals=MAX_EVALS)

    row = X.iloc[0]
    contributions = sorted(
        zip(FEATURE_COLUMNS, shap_values.values[0]), key=lambda t: -abs(t[1])
    )[:top_n]

    factors = []
    for feature, impact in contributions:
        factors.append(
            {
                "faktor": FEATURE_LABELS.get(feature, feature),
                "nilai_pelanggan": str(row[feature]),
                "arah": "meningkatkan risiko" if impact > 0 else "menurunkan risiko",
                "besar_pengaruh": round(float(abs(impact)), 4),
            }
        )

    return {"customer_id": customer_id, "faktor_utama": factors}
