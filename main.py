from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import logging
import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    confidence_score: Optional[float] = None
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None

# Define input schema for batch prediction
class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# Feature encoding functions
def encode_categorical_features(input_data: PredictionInput):
    """Convert categorical string features to numerical values for model prediction"""
    
    # Binary categorical mappings
    binary_mapping = {'yes': 1, 'no': 0}
    
    # Furnishing status mapping
    furnishing_mapping = {
        'furnished': 2,
        'semi-furnished': 1, 
        'unfurnished': 0
    }
    
    # Create feature array in the correct order
    features = [
        float(input_data.area),
        int(input_data.bedrooms),
        int(input_data.bathrooms),
        int(input_data.stories),
        binary_mapping.get(input_data.mainroad.lower(), 0),
        binary_mapping.get(input_data.guestroom.lower(), 0),
        binary_mapping.get(input_data.basement.lower(), 0),
        binary_mapping.get(input_data.hotwaterheating.lower(), 0),
        binary_mapping.get(input_data.airconditioning.lower(), 0),
        int(input_data.parking),
        binary_mapping.get(input_data.prefarea.lower(), 0),
        furnishing_mapping.get(input_data.furnishingstatus.lower(), 0)
    ]
    
    return np.array([features])

@app.get("/")
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "message": "House Price Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        logger.info("Prediction endpoint called with input: %s", input_data)
        
        # Encode categorical features properly
        features = encode_categorical_features(input_data)
        logger.info("Encoded features: %s", features)

        # Make prediction
        prediction = model.predict(features)[0]

        logger.info("Prediction made: %f", prediction)

        # Return prediction
        return PredictionOutput(prediction=prediction)

    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionOutput)
def batch_predict(batch_input: BatchPredictionInput):
    try:
        logger.info("Batch prediction endpoint called with inputs: %s", batch_input)
        predictions = []
        for input_data in batch_input.inputs:
            # Encode categorical features properly
            features = encode_categorical_features(input_data)

            # Make prediction
            prediction = model.predict(features)[0]
            predictions.append(PredictionOutput(prediction=prediction))

        logger.info("Batch predictions made: %s", predictions)

        # Return batch predictions
        return BatchPredictionOutput(predictions=predictions)

    except Exception as e:
        logger.error("Error during batch prediction: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Linear Regression",
        "problem_type": "regression",
        "features": ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
    }

@app.get("/test-form", response_class=HTMLResponse)
def get_test_form():
    """Serve a simple HTML form for testing the prediction API"""
    logger.info("Test form endpoint called")
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Prediction Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            input, select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .result.success {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .result.error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            @media (max-width: 600px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> House Price Prediction Test</h1>
            <form id="predictionForm">
                <div class="grid">
                    <div class="form-group">
                        <label for="area">Area (sq ft):</label>
                        <input type="number" id="area" name="area" required step="0.01" placeholder="e.g., 7420">
                    </div>
                    <div class="form-group">
                        <label for="bedrooms">Bedrooms:</label>
                        <input type="number" id="bedrooms" name="bedrooms" required min="1" placeholder="e.g., 4">
                    </div>
                    <div class="form-group">
                        <label for="bathrooms">Bathrooms:</label>
                        <input type="number" id="bathrooms" name="bathrooms" required min="1" placeholder="e.g., 1">
                    </div>
                    <div class="form-group">
                        <label for="stories">Stories:</label>
                        <input type="number" id="stories" name="stories" required min="1" placeholder="e.g., 3">
                    </div>
                    <div class="form-group">
                        <label for="mainroad">Main Road:</label>
                        <select id="mainroad" name="mainroad" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="guestroom">Guest Room:</label>
                        <select id="guestroom" name="guestroom" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="basement">Basement:</label>
                        <select id="basement" name="basement" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="hotwaterheating">Hot Water Heating:</label>
                        <select id="hotwaterheating" name="hotwaterheating" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="airconditioning">Air Conditioning:</label>
                        <select id="airconditioning" name="airconditioning" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="parking">Parking Spaces:</label>
                        <input type="number" id="parking" name="parking" required min="0" placeholder="e.g., 2">
                    </div>
                    <div class="form-group">
                        <label for="prefarea">Preferred Area:</label>
                        <select id="prefarea" name="prefarea" required>
                            <option value="">Select...</option>
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="furnishingstatus">Furnishing Status:</label>
                        <select id="furnishingstatus" name="furnishingstatus" required>
                            <option value="">Select...</option>
                            <option value="furnished">Furnished</option>
                            <option value="semi-furnished">Semi-furnished</option>
                            <option value="unfurnished">Unfurnished</option>
                        </select>
                    </div>
                </div>
                <button type="submit">🔮 Predict House Price</button>
            </form>
            <div id="result" class="result"></div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {};
                
                // Convert form data to the expected format
                data.area = parseFloat(formData.get('area'));
                data.bedrooms = parseInt(formData.get('bedrooms'));
                data.bathrooms = parseInt(formData.get('bathrooms'));
                data.stories = parseInt(formData.get('stories'));
                data.mainroad = formData.get('mainroad');
                data.guestroom = formData.get('guestroom');
                data.basement = formData.get('basement');
                data.hotwaterheating = formData.get('hotwaterheating');
                data.airconditioning = formData.get('airconditioning');
                data.parking = parseInt(formData.get('parking'));
                data.prefarea = formData.get('prefarea');
                data.furnishingstatus = formData.get('furnishingstatus');
                
                const resultDiv = document.getElementById('result');
                const button = e.target.querySelector('button');
                
                // Show loading state
                button.textContent = ' Predicting...';
                button.disabled = true;
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3> Prediction Result</h3>
                            <p><strong>Predicted House Price:</strong> $${result.prediction.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
                            ${result.confidence_score ? `<p><strong>Confidence Score:</strong> ${(result.confidence_score * 100).toFixed(2)}%</p>` : ''}
                            ${result.prediction_interval_lower && result.prediction_interval_upper ? `
                                <p><strong>Price Range:</strong> $${result.prediction_interval_lower.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})} - $${result.prediction_interval_upper.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
                            ` : ''}
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `
                            <h3> Error</h3>
                            <p>${result.detail || 'An error occurred while making the prediction.'}</p>
                        `;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <h3> Network Error</h3>
                        <p>Failed to connect to the API. Please make sure the server is running.</p>
                        <p><small>Error: ${error.message}</small></p>
                    `;
                } finally {
                    button.textContent = '🔮 Predict House Price';
                    button.disabled = false;
                    resultDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content
