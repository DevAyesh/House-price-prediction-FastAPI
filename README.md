# House Price Prediction with FastAPI

## Problem Description
This project aims to predict house prices based on various features such as area, number of bedrooms, bathrooms, and more. The dataset used for this project is `Housing.csv`, which contains 545 entries with 13 features.

## Model Choice Justification
Three models were trained: Random Forest Regressor, Linear Regression, and Decision Tree Regressor. The best model, based on R² score, was chosen for deployment. Linear Regression was selected as the best model due to its highest R² score, indicating better predictive performance on the test data.

## API Usage Examples
The FastAPI application provides several endpoints:
- `/predict`: Accepts a JSON payload with house features and returns the predicted price.
- `/health`: Returns the health status of the API.
- `/model-info`: Provides information about the model.
- `/sample-input`: Returns a sample input for the `/predict` endpoint.

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3, "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes", "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}'
```

## How to Run the Application
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI application using Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
3. Access the API documentation at `http://localhost:8000/docs`. 
