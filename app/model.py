import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class FraudDetectionModel:
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.optimal_threshold = 0.6
        self.model_loaded = False
        
    def load_model(self):
        """Load model and feature names"""
        try:
            model_path = Path("models/api_model_xgb.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            feature_path = Path("models/api_feature_names.csv") 
            features_df = pd.read_csv(feature_path)
            self.feature_names = features_df['feature'].tolist()
            
            self.model_loaded = True
            print(f"✓ Model loaded: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def preprocess_input(self, transaction_data: dict) -> pd.DataFrame:
        """
        Preprocess transaction for API model
        """
        # Convert to dataframe
        df = pd.DataFrame([transaction_data])
        
        # Amount features
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
            df['is_round_amount'] = (df['TransactionAmt'] % 10 == 0).astype(int)
        
        # Email features
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            # Fill NaN first
            df['P_emaildomain'].fillna('unknown', inplace=True)
            df['R_emaildomain'].fillna('unknown', inplace=True)
            
            df['email_domain_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
            
            common_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
            df['P_email_is_common'] = df['P_emaildomain'].isin(common_providers).astype(int)
            df['R_email_is_common'] = df['R_emaildomain'].isin(common_providers).astype(int)
            df['has_P_email'] = (df['P_emaildomain'] != 'unknown').astype(int)
            df['has_R_email'] = (df['R_emaildomain'] != 'unknown').astype(int)
        
        # === HANDLE CATEGORICAL ===
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna('unknown', inplace=True)
            # Simple label encoding (in production, save encoders from training!)
            df[col] = df[col].astype('category').cat.codes
        
        # === FILL MISSING NUMERICAL ===
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col].fillna(-999, inplace=True)
        
        # === ENSURE ALL FEATURES EXIST ===
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = -999
        
        # === SELECT ONLY MODEL FEATURES IN CORRECT ORDER ===
        df = df[self.feature_names]
        
        return df
    
    def predict(self, transaction_data: dict) -> dict:
        """Make prediction"""
        if not self.model_loaded:
            raise Exception("Model not loaded")
        
        # Preprocess
        X = self.preprocess_input(transaction_data)
        
        # Predict
        fraud_probability = float(self.model.predict_proba(X)[0][1])
        is_fraud = fraud_probability >= self.optimal_threshold
        
        # Risk level
        if fraud_probability < 0.3:
            risk_level = "Low"
        elif fraud_probability < 0.6:
            risk_level = "Medium"
        elif fraud_probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        # Message
        if is_fraud:
            message = f"⚠️ Transaction flagged as FRAUD (confidence: {fraud_probability*100:.1f}%)"
        else:
            message = f"✓ Transaction appears legitimate (fraud risk: {fraud_probability*100:.1f}%)"
        
        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
            "message": message
        }

# Global instance
fraud_model = FraudDetectionModel()