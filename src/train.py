# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("config/model_config.yaml") as f:
        return yaml.safe_load(f)

def train_model():
    # Load configuration
    config = load_config()
    
    # Load data
    df = pd.read_csv("https://raw.githubusercontent.com/nikhilsthorat03/Telco-Customer-Churn/refs/heads/main/telco.csv")
    
    # Preprocess
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Print unique values in Churn column
    logger.info(f"Unique values in Churn column: {df['Churn'].unique()}")
    logger.info(f"Target distribution before encoding:\n{df['Churn'].value_counts()}")
    
    # Convert target variable using LabelEncoder for consistent encoding
    le = LabelEncoder()
    y = le.fit_transform(df['Churn'])
    
    logger.info(f"Target distribution after encoding:\n{pd.Series(y).value_counts()}")
    logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Feature engineering with explicit dummy variable creation
    features = config['model_params']['features']
    dummy_features = pd.get_dummies(df[features])
    
    # Split data while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        dummy_features, y, 
        test_size=config['model_params']['test_size'],
        random_state=config['model_params']['random_state'],
        stratify=y
    )
    
    # Verify class distribution in training set
    logger.info(f"Training set class distribution: {np.bincount(y_train)}")
    
    # Train model
    model = RandomForestClassifier(
        random_state=config['model_params']['random_state'],
        n_estimators=100,
        class_weight='balanced'
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Verify model classes
    logger.info(f"Model classes: {model.classes_}")
    
    # Test predictions
    test_pred = model.predict_proba(X_test[:1])
    logger.info(f"Test prediction shape: {test_pred.shape}")
    logger.info(f"Sample prediction probabilities: {test_pred[0]}")
    
    # Save model and encoder
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open('features.pkl', 'wb') as f:
        pickle.dump(list(dummy_features.columns), f)
    
    logger.info(f"Model training completed. Features: {list(dummy_features.columns)}")
    return model

if __name__ == "__main__":
    train_model()