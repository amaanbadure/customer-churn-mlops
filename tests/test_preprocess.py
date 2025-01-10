import pytest
import pandas as pd
import numpy as np
from src.preprocess import DataPreprocessor

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'tenure': [1, 2, 3],
        'MonthlyCharges': [50.0, 60.0, 70.0],
        'TotalCharges': ['100.0', '200.0', '300.0'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer'],
        'Churn': ['Yes', 'No', 'No']
    })

def test_load_data(preprocessor, sample_data):
    # Test data loading and cleaning
    df = preprocessor.load_data("https://raw.githubusercontent.com/nikhilsthorat03/Telco-Customer-Churn/refs/heads/main/telco.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df['TotalCharges'].isnull().any()

def test_preprocess_features(preprocessor, sample_data):
    # Test feature preprocessing
    X, feature_names = preprocessor.preprocess_features(sample_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert all(X.columns == feature_names)

def test_prepare_target(preprocessor, sample_data):
    # Test target preparation
    y = preprocessor.prepare_target(sample_data)
    assert isinstance(y, pd.Series)
    assert y.dtype == np.int64
    assert set(y.unique()) == {0, 1}