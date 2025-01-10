import pytest
import pandas as pd
import numpy as np
import pickle
from src.train import train_model
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    model = train_model()
    assert isinstance(model, RandomForestClassifier)

def test_model_prediction():
    # Load model, encoder, and features
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Available features:", feature_names)
    
    # Create sample data with all features
    sample_data = pd.DataFrame(columns=feature_names)
    sample_data.loc[0] = 0  # Initialize with zeros
    
    # Set numeric features
    numeric_features = {
        'tenure': 36.0,
        'MonthlyCharges': 65.0,
        'TotalCharges': 2340.0
    }
    
    for feat, value in numeric_features.items():
        if feat in feature_names:
            sample_data[feat] = value
    
    # Set categorical features
    categorical_features = {
        'InternetService_DSL': 1,
        'Contract_Two year': 1,
        'PaymentMethod_Credit card (automatic)': 1
    }
    
    for feat, value in categorical_features.items():
        if feat in feature_names:
            sample_data[feat] = value
    
    # Print model information
    print(f"Model classes: {model.classes_}")
    
    # Make prediction
    prediction = model.predict_proba(sample_data)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction values: {prediction}")
    
    # Assertions
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 2), f"Expected shape (1, 2), got {prediction.shape}"
    assert all(0 <= p <= 1 for p in prediction[0]), "Probabilities should be between 0 and 1"
    assert np.isclose(sum(prediction[0]), 1), "Probabilities should sum to 1"

if __name__ == "__main__":
    test_model_prediction()