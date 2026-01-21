import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Fixtures
@pytest.fixture(scope="module")
def predictor_class():
    """Import ChurnPredictor class from /results/utils.py"""
    source_path = Path("/results/utils.py")
    
    if not source_path.exists():
        pytest.fail(f"File '{source_path}' does not exist.")
    
    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, "ChurnPredictor"):
            pytest.fail(f"Class 'ChurnPredictor' not found in {source_path}")
        
        return module.ChurnPredictor
    except Exception as e:
        pytest.fail(f"Failed to import model: {e}")

@pytest.fixture(scope="module")
def churn_data():
    """Load training and test data"""
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")
    
    if not train_path.exists() or not test_path.exists():
        pytest.fail(f"Data files missing. Train: {train_path}, Test: {test_path}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    if 'Churn' not in df_train.columns:
        pytest.fail("'Churn' column missing from train.csv")
    
    return df_train, df_test

@pytest.fixture(scope="module")
def trained_model(predictor_class, churn_data):
    """Train model for testing"""
    ChurnPredictor = predictor_class
    df_train, df_test = churn_data
    
    model = ChurnPredictor()
    model.fit(df_train)
    
    return model, df_test

# Tests
def test_file_exists():
    """Check that utils.py exists"""
    assert Path("/results/utils.py").exists(), "File /results/utils.py does not exist"

def test_class_structure(predictor_class):
    """Check that class has required methods"""
    model = predictor_class()
    
    assert hasattr(model, 'fit'), "Missing 'fit' method"
    assert hasattr(model, 'predict'), "Missing 'predict' method"
    assert hasattr(model, 'predict_proba'), "Missing 'predict_proba' method"
    
    assert callable(model.fit), "'fit' must be callable"
    assert callable(model.predict), "'predict' must be callable"
    assert callable(model.predict_proba), "'predict_proba' must be callable"

def test_model_training(predictor_class, churn_data):
    """Test that model can be trained"""
    ChurnPredictor = predictor_class
    df_train, _ = churn_data
    
    model = ChurnPredictor()
    model.fit(df_train)
    
    assert model.pipeline is not None, "Model pipeline not created after training"

def test_predict_method(trained_model):
    """Test predict() returns binary predictions (0 or 1)"""
    model, df_test = trained_model
    
    y_pred = model.predict(df_test.drop('Churn', axis=1))
    
    assert len(y_pred) == len(df_test), "Prediction length mismatch"
    assert set(y_pred).issubset({0, 1}), f"Predictions must be 0 or 1, got {set(y_pred)}"

def test_model_performance(trained_model):
    """Test that model achieves ROC AUC >= 0.83"""
    model, df_test = trained_model
    
    y_true = (df_test['Churn'] == 'Yes').astype(int)
    y_pred_proba = model.predict_proba(df_test.drop('Churn', axis=1))[:, 1]
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nModel ROC AUC Score: {roc_auc:.4f}")
    
    assert 0.0 <= roc_auc <= 1.0, f"ROC AUC must be in [0, 1], got {roc_auc:.4f}"
    assert roc_auc >= 0.83, f"ROC AUC ({roc_auc:.4f}) is below required threshold of 0.83"