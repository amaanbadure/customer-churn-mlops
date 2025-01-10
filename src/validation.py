import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import json
from typing import Dict, Any
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model_path: str = 'model.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
    def validate_data(self, X: pd.DataFrame, expected_features: list) -> bool:
        """Validate input data structure"""
        if not all(feature in X.columns for feature in expected_features):
            logger.error("Missing required features in input data")
            return False
        return True
        
    def validate_predictions(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate and return validation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Save metrics to file
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        logger.info(f"Validation metrics: {metrics}")
        return metrics
        
    def validate_model_performance(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.7) -> bool:
        """Validate model performance meets minimum threshold"""
        y_pred = self.model.predict(X)
        metrics = self.validate_predictions(y, y_pred)
        
        return metrics['f1'] >= threshold

if __name__ == "__main__":
    from preprocess import DataPreprocessor
    
    # Load and preprocess test data
    preprocessor = DataPreprocessor()
    X, y, features = preprocessor.process_pipeline(
        "https://raw.githubusercontent.com/nikhilsthorat03/Telco-Customer-Churn/refs/heads/main/telco.csv"
    )
    
    # Validate model
    validator = ModelValidator()
    is_valid = validator.validate_model_performance(X, y)
    logger.info(f"Model validation {'passed' if is_valid else 'failed'}")