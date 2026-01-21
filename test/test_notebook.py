import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------------------------
# Fixtures (Required to load the agent's code and data)
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor_class():
    """Dynamically imports ChurnPredictor from /results/utils.py"""
    source_path = Path("/results/utils.py")
    if not source_path.exists():
        pytest.fail(f"CRITICAL: Source file '{source_path}' missing.")
    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)
        return module.ChurnPredictor
    except AttributeError:
        pytest.fail("CRITICAL: Class 'ChurnPredictor' not found in utils.py")
    except Exception as e:
        pytest.fail(f"CRITICAL: Import failed. {e}")

@pytest.fixture(scope="module")
def student_data():
    """Loads blind test features and hidden ground truth."""
    X_test_path = Path("/workspace/data/test.csv")
    y_test_path = Path("/eval_data/ground_truth.csv")
    train_path = Path("/workspace/data/train.csv")

    if not all(p.exists() for p in [X_test_path, y_test_path, train_path]):
        pytest.fail("CRITICAL: Data missing. Check /eval_data for ground truth.")

    # Load Data
    X_test = pd.read_csv(X_test_path)
    # Target is now 'Churn' based on the Telecom Customer Churn task
    y_test = pd.read_csv(y_test_path)["Churn"]
    df_train = pd.read_csv(train_path)

    X_train = df_train.drop("Churn", axis=1)
    y_train = df_train["Churn"]

    if "Churn" in X_test.columns:
        pytest.fail("SECURITY: Target leaked in test.csv")

    return X_train, y_train, X_test, y_test

# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------

def test_performance_roc_auc(predictor_class, student_data):
    """Test 1: ROC-AUC must be > 0.83 on blind test set."""
    ChurnPredictor = predictor_class
    X_train, y_train, X_test, y_test = student_data

    model = ChurnPredictor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert y_test to numeric labels if needed
    if y_test.dtype == object:
        y_test_bin = y_test.map({"No": 0, "Yes": 1})
    else:
        y_test_bin = y_test

    roc_auc = roc_auc_score(y_test_bin, y_pred)

    # Strict threshold based on task goal
    assert roc_auc > 0.83, f"FAIL: ROC-AUC is {roc_auc:.4f}, expected > 0.83"

def test_robustness_garbage_columns(predictor_class, student_data):
    """Test 2: Model must handle extra 'garbage' columns without crashing."""
    ChurnPredictor = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = ChurnPredictor()
    model.fit(X_train, y_train)

    # Create a copy of test data with a random garbage column
    X_dirty = X_test.copy()
    X_dirty["random_garbage_999"] = np.random.random(len(X_dirty))

    try:
        # This should not raise an error (KeyError, ValueError, etc.)
        model.predict(X_dirty)
    except Exception as e:
        pytest.fail(f"FAIL: Model crashed when input contained extra columns. Error: {e}")

    