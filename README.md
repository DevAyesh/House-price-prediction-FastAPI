# House Price Prediction API

## 📁 Project Structure
```
house_price_prediction/
├── main.py                              # FastAPI application with all endpoints
├── house_price_model.pkl                # Trained machine learning model
├── Housing.csv                          # Dataset (545 entries, 13 features)
├── House_Price_Prediction_Complete.ipynb # Complete ML pipeline notebook
├── requirements.txt                     # Python dependencies
├── PROJECT_REPORT.md                   # Detailed project analysis
├── README.md                           # This file
└── api.log                             # API request logs
```

## 🏠 Problem Description
This project predicts house prices based on various features such as area, number of bedrooms, bathrooms, and more. The dataset contains 545 entries with 13 features including both numerical and categorical variables.

## 🤖 Model Information
- **Algorithm**: Linear Regression (selected based on highest R² score)
- **Alternative models tested**: Random Forest Regressor, Decision Tree Regressor
- **Features**: 12 input features with proper categorical encoding
- **Performance**: Best R² score among tested models

## 🚀 API Endpoints

### Core Endpoints
- **`GET /`** - Health check endpoint
- **`POST /predict`** - Single house price prediction
- **`POST /batch-predict`** - Multiple house price predictions
- **`GET /model-info`** - Model metadata and feature information
- **`GET /test-form`** - Interactive HTML form for testing

### 🌐 Web Interface
Access the interactive test form at: `http://localhost:8000/test-form`

The form includes:
- All 12 input fields with proper validation
- Real-time prediction results
- Error handling and user feedback
- Responsive design for mobile and desktop

## 📊 API Features

### Enhanced Logging
- Structured logging with timestamps
- File-based logging (`api.log`)
- Request/response tracking
- Error monitoring

### Data Processing
- Automatic categorical feature encoding
- Input validation and sanitization
- Proper error handling with detailed messages

### Batch Processing
- Process multiple predictions in a single request
- Efficient bulk operations
- Consistent response format

## 🔧 Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 7420,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 3,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
      },
      {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 1,
        "stories": 2,
        "mainroad": "no",
        "guestroom": "yes",
        "basement": "yes",
        "hotwaterheating": "yes",
        "airconditioning": "no",
        "parking": 1,
        "prefarea": "no",
        "furnishingstatus": "unfurnished"
      }
    ]
  }'
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd house_price_prediction
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. **Access the application**
   - API Documentation: `http://localhost:8000/docs`
   - Interactive Form: `http://localhost:8000/test-form`
   - Health Check: `http://localhost:8000/`

## 📋 Input Features

| Feature | Type | Values | Description |
|---------|------|---------|-------------|
| area | float | > 0 | House area in square feet |
| bedrooms | int | ≥ 1 | Number of bedrooms |
| bathrooms | int | ≥ 1 | Number of bathrooms |
| stories | int | ≥ 1 | Number of stories |
| mainroad | string | "yes"/"no" | Connected to main road |
| guestroom | string | "yes"/"no" | Has guest room |
| basement | string | "yes"/"no" | Has basement |
| hotwaterheating | string | "yes"/"no" | Has hot water heating |
| airconditioning | string | "yes"/"no" | Has air conditioning |
| parking | int | ≥ 0 | Number of parking spaces |
| prefarea | string | "yes"/"no" | In preferred area |
| furnishingstatus | string | "furnished"/"semi-furnished"/"unfurnished" | Furnishing status |

## 📈 Response Format

### Single Prediction Response
```json
{
  "prediction": 4500000.50,
  "confidence_score": null,
  "prediction_interval_lower": null,
  "prediction_interval_upper": null
}
```

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "prediction": 4500000.50,
      "confidence_score": null,
      "prediction_interval_lower": null,
      "prediction_interval_upper": null
    },
    {
      "prediction": 3200000.25,
      "confidence_score": null,
      "prediction_interval_lower": null,
      "prediction_interval_upper": null
    }
  ]
}
```

## 🔍 Monitoring & Logs
- All API requests are logged to `api.log`
- Logs include timestamps, request details, and error information
- Monitor the log file for debugging and performance analysis

## 📚 Additional Resources
- **Complete Analysis**: See `House_Price_Prediction_Complete.ipynb` for detailed EDA and model training
- **Project Report**: See `PROJECT_REPORT.md` for comprehensive project documentation
- **API Documentation**: Visit `/docs` endpoint for interactive Swagger documentation 
