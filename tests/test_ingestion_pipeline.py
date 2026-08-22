import pandas as pd
import pytest
from pipeline import REQUIRED_COLUMNS, clean, validate_raw


def _base_row(**overrides):
    row = {
        "customerID": "C0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.5,
        "TotalCharges": "666.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_validate_raw_rejects_missing_columns():
    df = pd.DataFrame([{"customerID": "C0001"}])
    with pytest.raises(ValueError):
        validate_raw(df)


def test_validate_raw_rejects_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError):
        validate_raw(df)


def test_validate_raw_accepts_well_formed_data():
    df = pd.DataFrame([_base_row()])
    validate_raw(df)  # should not raise


def test_clean_drops_rows_with_blank_total_charges():
    df = pd.DataFrame(
        [_base_row(customerID="C0001", TotalCharges=" "), _base_row(customerID="C0002")]
    )
    result = clean(df)
    assert "C0001" not in result["customer_id"].values
    assert "C0002" in result["customer_id"].values


def test_clean_drops_duplicates_keeping_last():
    df = pd.DataFrame(
        [_base_row(customerID="C0001", tenure=1), _base_row(customerID="C0001", tenure=99)]
    )
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]["tenure"] == 99


def test_clean_converts_yes_no_to_boolean():
    df = pd.DataFrame([_base_row(Churn="Yes", Partner="No")])
    result = clean(df)
    assert result.iloc[0]["churn"] == True  # noqa: E712
    assert result.iloc[0]["partner"] == False  # noqa: E712


def test_clean_renames_columns_to_snake_case():
    df = pd.DataFrame([_base_row()])
    result = clean(df)
    assert "customer_id" in result.columns
    assert "monthly_charges" in result.columns
    assert "customerID" not in result.columns
