import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn": bool(prediction),
        "probability": float(probability)
    }