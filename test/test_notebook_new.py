"""
Use this file to define pytest tests that verify the outputs of the Telecom Customer Churn Prediction task.
This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# ==============================================================================
# CONFIGURATION - File paths inside the container
# ==============================================================================

NOTEBOOK_DIR = Path("/workspace")
TEST_DIR = Path("/tests")
VERIFIER_DIR = Path("/logs/verifier")

# Where the harness saves captured notebook variables
NOTEBOOK_VARS_PATH = VERIFIER_DIR / "notebook_variables.json"

# ==============================================================================
# EXPECTED VALUES
# ==============================================================================

# Expected metrics based on Random Forest (Random State 42, 70/30 Split)
EXPECTED_MODEL_QUALITY = {
    "f1_score": 0.57889,
    "auc_score": 0.83881,
}

# Expected counts of classes in the test set predictions
EXPECTED_CHURN_COUNTS = {
    0: 1692,
    1: 421
}

REQUIRED_MODEL_METRICS_KEYS = ["f1_score", "auc_score"]

# Expected features after Data Prep and One-Hot Encoding
EXPECTED_FEATURE_KEYS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 
    'TenureChargeInteraction', 'ShortTenureFlag', 'SeniorCitizen', 
    'PaperlessBilling_No', 'PaperlessBilling_Yes', 
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
    'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes', 
    'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'
]


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def notebook_variables() -> dict:
    """Load and return the variables assigned in the notebook."""
    if not NOTEBOOK_VARS_PATH.exists():
        pytest.skip("notebook_variables.json not found")
    with open(NOTEBOOK_VARS_PATH, "r") as f:
        return json.load(f)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def reconstruct_dataframe_from_dict(data: dict) -> pd.DataFrame:
    """
    Reconstruct a pandas DataFrame from a dictionary.
    Handles dictionaries with 'split' orientation (from to_dict(orient='split')).
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        if "columns" in data and "data" in data:
            return pd.DataFrame(data["data"], columns=data["columns"])
        # Fallback for simple dicts
        return pd.DataFrame(data)
    raise ValueError(f"Cannot reconstruct DataFrame from type: {type(data)}")


def normalize_feature_importance(data) -> dict:
    """
    Helper to extract feature importance as a simple dict {feature: importance},
    handling cases where the user returned a DataFrame or a serialized DataFrame.
    """
    # Case 1: Already a simple dict
    if isinstance(data, dict) and not ("columns" in data and "data" in data):
        # check if values are scalar
        if all(isinstance(v, (int, float)) for v in data.values()):
            return data
            
    # Case 2: Serialized DataFrame (orient='split') or raw DataFrame structure
    try:
        df = reconstruct_dataframe_from_dict(data)
        # Expected DataFrame structure: Index=Features OR Column=Feature, Column=Importance
        # Attempt to coerce to dict
        if df.shape[1] == 1: # Index contains feature names
             return df.iloc[:, 0].to_dict()
        elif df.shape[1] == 2: # Two columns: FeatureName, Importance
             # Assume non-numeric column is keys, numeric is values
             numeric_cols = df.select_dtypes(include=['number']).columns
             if len(numeric_cols) == 1:
                 val_col = numeric_cols[0]
                 key_col = [c for c in df.columns if c != val_col][0]
                 return dict(zip(df[key_col], df[val_col]))
    except:
        pass
    
    # Return as-is if normalization fails, letting tests fail naturally
    return data


# ==============================================================================
# BASIC EXISTENCE TESTS
# ==============================================================================

def test_notebook_exists() -> None:
    """Test that the notebook exists."""
    notebook_path = NOTEBOOK_DIR / "notebook.ipynb"
    assert notebook_path.exists(), (
        f"Notebook 'notebook.ipynb' not found in {NOTEBOOK_DIR}"
    )


# ==============================================================================
# VARIABLE EXISTENCE TESTS
# ==============================================================================

def test_feature_importance_dict_exists(notebook_variables: dict) -> None:
    """Test that feature_importance_dict exists in the environment."""
    assert "feature_importance_dict" in notebook_variables, (
        "feature_importance_dict must exist in the environment"
    )
    # Accepts Dict (preferred) or Serialized DataFrame
    assert isinstance(notebook_variables["feature_importance_dict"], dict), (
        "feature_importance_dict must be a dictionary (or serialized DataFrame)"
    )


def test_model_metrics_exists(notebook_variables: dict) -> None:
    """Test that model_metrics exists in the environment."""
    assert "model_metrics" in notebook_variables, (
        "model_metrics must exist in the environment"
    )
    assert isinstance(notebook_variables["model_metrics"], dict), (
        "model_metrics must be a dictionary"
    )


def test_churn_counts_exists(notebook_variables: dict) -> None:
    """Test that churn_counts exists in the environment."""
    assert "churn_counts" in notebook_variables, (
        "churn_counts must exist in the environment"
    )
    assert isinstance(notebook_variables["churn_counts"], dict), (
        "churn_counts must be serialized as a dictionary using to_dict(orient='split')"
    )


# ==============================================================================
# KEYS AND STRUCTURE TESTS
# ==============================================================================

def test_model_metrics_keys_correct(notebook_variables: dict) -> None:
    """Test that model_metrics has f1_score and auc_score keys."""
    model_metrics = notebook_variables.get("model_metrics", {})
    actual_keys = set(model_metrics.keys())
    expected_keys = set(REQUIRED_MODEL_METRICS_KEYS)

    assert actual_keys == expected_keys, (
        f"model_metrics must contain exactly these keys: {REQUIRED_MODEL_METRICS_KEYS}. "
        f"Got: {sorted(actual_keys)}"
    )


def test_feature_importance_keys_correct(notebook_variables: dict) -> None:
    """Test that feature_importance_dict has all required feature keys."""
    raw_data = notebook_variables.get("feature_importance_dict", {})
    feature_dict = normalize_feature_importance(raw_data)
    
    actual_keys = set(feature_dict.keys())
    expected_keys = set(EXPECTED_FEATURE_KEYS)

    # Allow for minor differences if student dropped extra columns, but critical ones must exist
    missing = expected_keys - actual_keys
    
    assert not missing, (
        f"feature_importance_dict is missing required encoded features: {sorted(missing)}. "
        "Did you perform One-Hot Encoding correctly on all categorical variables?"
    )


# ==============================================================================
# VALUE ACCURACY TESTS (MODEL QUALITY)
# ==============================================================================

def test_auc_within_tolerance(notebook_variables: dict) -> None:
    """Test that AUC matches expected value within tolerance."""
    model_metrics = notebook_variables.get("model_metrics", {})
    auc = model_metrics.get("auc_score")

    assert auc is not None, "auc_score key not found in model_metrics"

    expected = EXPECTED_MODEL_QUALITY["auc_score"]
    tolerance = 0.05  # Allowing variance for RF randomness/environment

    assert abs(auc - expected) <= tolerance, (
        f"AUC Score should be {expected} ± {tolerance}, got {auc}"
    )


def test_f1_within_tolerance(notebook_variables: dict) -> None:
    """Test that F1 matches expected value within tolerance."""
    model_metrics = notebook_variables.get("model_metrics", {})
    f1 = model_metrics.get("f1_score")

    assert f1 is not None, "f1_score key not found in model_metrics"

    expected = EXPECTED_MODEL_QUALITY["f1_score"]
    tolerance = 0.05  # Allowing variance for RF randomness/environment

    assert abs(f1 - expected) <= tolerance, (
        f"F1 Score should be {expected} ± {tolerance}, got {f1}"
    )


def test_metrics_values_reasonable_bounds(notebook_variables: dict) -> None:
    """Test that metrics are valid probabilities in [0, 1]."""
    model_metrics = notebook_variables.get("model_metrics", {})
    auc = model_metrics.get("auc_score", 0)
    f1 = model_metrics.get("f1_score", 0)

    assert 0.0 <= auc <= 1.0, f"AUC must be between 0 and 1, got {auc}"
    assert 0.0 <= f1 <= 1.0, f"F1 must be between 0 and 1, got {f1}"


# ==============================================================================
# churn_counts VALIDATION
# ==============================================================================

def test_churn_counts_structure(notebook_variables: dict) -> None:
    """Test churn_counts reconstructs to a DataFrame and contains expected info."""
    churn_counts = reconstruct_dataframe_from_dict(notebook_variables["churn_counts"])

    assert churn_counts.shape[0] > 0, "churn_counts must not be empty"
    # Expecting columns for Class and Count (or Index=Class, Col=Count)
    assert churn_counts.size >= 2, "churn_counts must contain data"


def test_churn_counts_values(notebook_variables: dict) -> None:
    """Test that churn_counts values match the expected predictions."""
    churn_counts_df = reconstruct_dataframe_from_dict(notebook_variables["churn_counts"])
    
    # Normalize to a dict {class: count}
    # Attempt to identify the count column (numeric)
    try:
        numeric_cols = churn_counts_df.select_dtypes(include=['number'])
        if numeric_cols.shape[1] > 0:
            # Summing just in case structure is weird, but usually it's one row per class
            count_values = numeric_cols.iloc[:, 0].tolist() 
            # Check if values are close to expected [1692, 421]
            # We sort to compare sets of values regardless of order
            actual_values = sorted(count_values)
            expected_values = sorted(EXPECTED_CHURN_COUNTS.values())
            
            # Allow small variance (e.g. +/- 5 predictions)
            assert all(abs(a - e) <= 5 for a, e in zip(actual_values, expected_values)), (
                 f"Churn counts do not match expected distribution. "
                 f"Expected approx {expected_values}, got {actual_values}"
            )
    except Exception as e:
        pytest.fail(f"Could not validate churn_counts values: {e}")