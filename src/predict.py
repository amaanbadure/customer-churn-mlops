from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
import yaml
import uvicorn

app = FastAPI()

# Load model and features
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    Contract: str
    PaymentMethod: str

@app.post("/predict")
async def predict(customer: CustomerData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # Create dummy variables
        input_processed = pd.get_dummies(input_df)
        
        # Align features with training data
        for feature in features:
            if feature not in input_processed.columns:
                input_processed[feature] = 0
        
        input_processed = input_processed[features]
        
        # Make prediction
        prediction = model.predict_proba(input_processed)[0][1]
        
        return {
            "churn_probability": float(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def start_server():
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port']
    )

if __name__ == "__main__":
    start_server()