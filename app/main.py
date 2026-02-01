from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, PredictionResponse, HealthResponse
from app.model import fraud_model
import pandas as pd
from io import StringIO

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for e-commerce transactions using XGBoost",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("Loading fraud detection model...")
    fraud_model.load_model()
    print("✓ API ready!")

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Check if API is running and model is loaded"""
    return {
        "status": "healthy",
        "ml_model_loaded": fraud_model.model_loaded,
        "ml_model_type": "XGBoost (Production Optimized)"
    }

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud for a single transaction
    
    Returns:
    - is_fraud: Boolean indicating if transaction is fraudulent
    - fraud_probability: Probability score (0-1)
    - risk_level: Low/Medium/High/Critical
    - message: Human-readable result
    """
    try:
        if hasattr(transaction, 'model_dump'):
            transaction_dict = transaction.model_dump()
        else:
            transaction_dict = transaction.dict()
        
        # Get prediction
        result = fraud_model.predict(transaction_dict)
        
        return result
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict fraud for multiple transactions from CSV file
    
    Upload a CSV with transaction data, get predictions for all rows
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        print(f"Received {len(df)} transactions for batch prediction")
        
        # Make predictions for each row
        predictions = []
        for idx, row in df.iterrows():
            transaction_dict = row.to_dict()
            result = fraud_model.predict(transaction_dict)
            predictions.append({
                'transaction_id': idx,
                **result
            })
        
        return {
            "total_transactions": len(predictions),
            "fraud_detected": sum(1 for p in predictions if p['is_fraud']),
            "predictions": predictions
        }
        
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "XGBoost Classifier",
        "version": "2.0 (API-Ready)",
        "features_count": len(fraud_model.feature_names) if fraud_model.feature_names else "Unknown",
        "optimal_threshold": fraud_model.optimal_threshold,
        "training_date": "2025-01-20",  
        "performance": {
            "precision": "12.51%", 
            "recall": "67.32%",
            "f1_score": "21.10%",
            "roc_auc": "82.42%"
        },
        "note": "Optimized for real-time inference (<100ms)"
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)