import pandas as pd
import numpy as np
from typing import Tuple, List
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.features = self.config['model_params']['features']
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and perform initial cleaning of data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Handle missing values and conversions
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        return df
        
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Preprocess features and return processed dataframe and feature names"""
        logger.info("Preprocessing features")
        
        # Select features
        X = df[self.features]
        
        # Create dummy variables
        X_processed = pd.get_dummies(X, drop_first=True)
        
        return X_processed, X_processed.columns.tolist()
        
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable"""
        logger.info("Preparing target variable")
        return (df['Churn'] == 'Yes').astype(int)
        
    def process_pipeline(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Run complete preprocessing pipeline"""
        df = self.load_data(data_path)
        X, feature_names = self.preprocess_features(df)
        y = self.prepare_target(df)
        
        return X, y, feature_names

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X, y, features = preprocessor.process_pipeline(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-dataset/master/Telco-Customer-Churn.csv"
    )
    logger.info(f"Preprocessing complete. Features shape: {X.shape}")