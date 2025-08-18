from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Load your trained model
model = joblib.load("house_price_model.pkl")

app = FastAPI(title="House Price Prediction API", description="API for predicting house prices")

# Define input schema
class PredictionInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

class PredictionOutput(BaseModel):
    prediction: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to model format
        features = np.array([[input_data.area, input_data.bedrooms, input_data.bathrooms, input_data.stories,
                              input_data.mainroad, input_data.guestroom, input_data.basement, input_data.hotwaterheating,
                              input_data.airconditioning, input_data.parking, input_data.prefarea, input_data.furnishingstatus]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Return prediction
        return PredictionOutput(prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Linear Regression",
        "problem_type": "regression",
        "features": ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
    }
