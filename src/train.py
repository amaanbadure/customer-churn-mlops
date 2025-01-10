import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("config/model_config.yaml") as f:
        return yaml.safe_load(f)

def train_model():
    # Load configuration
    config = load_config()
    
    # Load data (using Telco Customer Churn dataset)
    df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-dataset/master/Telco-Customer-Churn.csv")
    
    # Preprocess
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Feature engineering
    features = config['model_params']['features']
    X = pd.get_dummies(df[features], drop_first=True)
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model_params']['test_size'],
        random_state=config['model_params']['random_state']
    )
    
    # Train model
    model = RandomForestClassifier(random_state=config['model_params']['random_state'])
    model.fit(X_train, y_train)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    with open('features.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    logger.info("Model training completed")
    return model

if __name__ == "__main__":
    train_model()