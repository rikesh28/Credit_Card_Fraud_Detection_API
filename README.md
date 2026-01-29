# Fraud Detection API

Real-time fraud detection API built with FastAPI and XGBoost.

## Features

- Single transaction prediction
- Batch CSV processing
- Risk level classification (Low/Medium/High/Critical)
- Optimal threshold from ML tuning
- Interactive API documentation (Swagger UI)

## Quick Start

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/rikesh28/Credit_Card_Fraud_Detection
cd fraud-api

# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Run Locally

\`\`\`bash
python -m app.main
\`\`\`

API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
\`\`\`
GET /
\`\`\`

### Single Prediction
\`\`\`
POST /predict
Content-Type: application/json

{
  "TransactionAmt": 150.50,
  "ProductCD": "W",
  "card1": 13926,
  ...
}
\`\`\`

### Batch Prediction
\`\`\`
POST /predict/batch
Content-Type: multipart/form-data

Upload CSV file with transaction data
\`\`\`

### Model Info
\`\`\`
GET /model/info
\`\`\`

## Interactive Documentation

Visit `http://localhost:8000/docs` for full interactive API documentation.

## Model Performance

- "precision": "12.51%", 
- "recall": "67.32%",
- "f1_score": "21.10%",
- "roc_auc": "82.42%"

## Tech Stack

- FastAPI
- XGBoost
- Pandas
- Pydantic
- Uvicorn
\`\`\`
