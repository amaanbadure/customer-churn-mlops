import pytest
import pandas as pd
import pickle
from src.train import train_model
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    model = train_model()
    assert isinstance(model, RandomForestClassifier)
    
def test_model_prediction():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Test sample
    sample_input = pd.DataFrame({
        'tenure': [36],
        'MonthlyCharges': [65.0],
        'TotalCharges': [2340.0],
        'InternetService_Fiber optic': [1],
        'InternetService_No': [0],
        'Contract_One year': [0],
        'Contract_Two year': [1],
        'PaymentMethod_Credit card (automatic)': [1],
        'PaymentMethod_Electronic check': [0],
        'PaymentMethod_Mailed check': [0]
    })
    
    prediction = model.predict_proba(sample_input)
    assert len(prediction[0]) == 2  # Binary classification
    assert 0 <= prediction[0][1] <= 1  # Probability between 0 and 1