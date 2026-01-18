import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# Fixture: Handle Dynamic Import of the Model
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor_class():
    """
    Locates /results/utils.py, dynamically imports it as a module,
    and returns the ChurnPredictor class.
    Fails the test suite immediately if the file or class is missing.
    """
    source_path = Path("/results/utils.py")

    # 1. Check file existence
    if not source_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Source file '{source_path}' does not exist.")

    # 2. Dynamic Import
    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)

        # 3. Check for Class Existence
        if not hasattr(module, "ChurnPredictor"):
            pytest.fail(f"CRITICAL FAILURE: Class 'ChurnPredictor' not found in {source_path}")

        return module.ChurnPredictor

    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Failed to import model from {source_path}. Error: {e}")

# ----------------------------------------------------------------------
# Fixture: Handle Data Loading
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def churn_data():
    """
    Loads churn train/test datasets as full dataframes (including 'Churn').
    """
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")

    if not train_path.exists() or not test_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Data files missing. Train: {train_path}, Test: {test_path}")

    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if 'Churn' not in df_train.columns:
            pytest.fail("CRITICAL FAILURE: 'Churn' column missing from train.csv")

        return df_train, df_test

    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Error processing data files: {e}")

# ----------------------------------------------------------------------
# Fixture: Trained Model Instance
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model(predictor_class, churn_data):
    """
    Creates and trains a model instance for use in multiple tests.
    """
    ChurnPredictor = predictor_class
    df_train, df_test = churn_data

    model = ChurnPredictor()
    model.fit(df_train)

    return model, df_train, df_test

# ----------------------------------------------------------------------
# Test: File Existence
# ----------------------------------------------------------------------

def test_file_exists():
    """
    Verify that the utils.py file exists at the expected location.
    """
    source_path = Path("/results/utils.py")
    assert source_path.exists(), f"Source file '{source_path}' does not exist."

# ----------------------------------------------------------------------
# Test: Class Structure
# ----------------------------------------------------------------------

def test_class_structure(predictor_class):
    """
    Verify that the ChurnPredictor class has the required methods.
    """
    ChurnPredictor = predictor_class

    # Test instantiation
    model = ChurnPredictor()

    # Check required methods exist
    assert hasattr(model, 'fit'), "Model must have 'fit' method"
    assert hasattr(model, 'predict'), "Model must have 'predict' method"
    assert hasattr(model, 'predict_proba'), "Model must have 'predict_proba' method"

    # Check methods are callable
    assert callable(model.fit), "'fit' must be callable"
    assert callable(model.predict), "'predict' must be callable"
    assert callable(model.predict_proba), "'predict_proba' must be callable"

# ----------------------------------------------------------------------
# Test: Model Training
# ----------------------------------------------------------------------

def test_model_training(predictor_class, churn_data):
    """
    Verify that the model can be trained without errors.
    """
    ChurnPredictor = predictor_class
    df_train, df_test = churn_data

    model = ChurnPredictor()

    # Training should complete without exception
    try:
        model.fit(df_train)
    except Exception as e:
        pytest.fail(f"Model training failed with error: {e}")

# ----------------------------------------------------------------------
# Test: Predict Method
# ----------------------------------------------------------------------

def test_predict_method(trained_model):
    """
    Verify that the predict() method returns valid binary predictions.
    """
    model, df_train, df_test = trained_model

    # Get predictions
    y_pred = model.predict(df_test.drop('Churn', axis=1))

    # Check output shape
    assert len(y_pred) == len(df_test), f"Prediction length {len(y_pred)} doesn't match test data length {len(df_test)}"

    # Check binary output
    unique_values = set(y_pred)
    assert unique_values.issubset({0, 1}), f"Predictions must be binary (0 or 1), got {unique_values}"

    # Check output type
    assert isinstance(y_pred, (np.ndarray, list)), "Predictions must be array-like"

# ----------------------------------------------------------------------
# Test: Predict Proba Method
# ----------------------------------------------------------------------

def test_predict_proba_method(trained_model):
    """
    Verify that the predict_proba() method returns valid probability arrays.
    """
    model, df_train, df_test = trained_model

    # Get probability predictions
    y_pred_proba = model.predict_proba(df_test.drop('Churn', axis=1))

    # Check output shape
    assert y_pred_proba.shape == (len(df_test), 2), \
        f"predict_proba must return shape (n_samples, 2), got {y_pred_proba.shape}"

    # Check probabilities are in valid range [0, 1]
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), \
        "All probabilities must be between 0 and 1"

    # Check probabilities sum to 1 for each sample
    prob_sums = y_pred_proba.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), \
        "Probabilities for each sample must sum to 1"

# ----------------------------------------------------------------------
# Test: Model Performance
# ----------------------------------------------------------------------

def test_model_performance(trained_model):
    """
    Verify that the model achieves the required ROC AUC performance (>= 0.83).
    """
    model, df_train, df_test = trained_model

    # Map target to binary (Yes → 1, No → 0)
    y_true = (df_test['Churn'] == 'Yes').astype(int)

    # Get predicted probabilities (positive class = churn)
    y_pred_proba = model.predict_proba(df_test.drop('Churn', axis=1))[:, 1]

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    target_roc_auc = 0.83

    print(f"Model ROC AUC Score: {roc_auc:.4f}")  # Visible with pytest -s

    # Check ROC AUC is in valid range
    assert 0.0 <= roc_auc <= 1.0, f"ROC AUC must be in [0, 1], got {roc_auc:.4f}"

    # Check ROC AUC meets minimum threshold
    assert roc_auc >= target_roc_auc, \
        f"Performance Failure: Model ROC AUC ({roc_auc:.4f}) is below the " \
        f"required threshold of {target_roc_auc} (83%)"