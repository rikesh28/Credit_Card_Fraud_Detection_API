import pandas as pd
from datetime import datetime

def calculate_fraud_stats(predictions: list) -> dict:
    """Calculate summary statistics from predictions"""
    
    total = len(predictions)
    fraud_count = sum(1 for p in predictions if p['is_fraud'])
    
    risk_distribution = {
        'Low': sum(1 for p in predictions if p['risk_level'] == 'Low'),
        'Medium': sum(1 for p in predictions if p['risk_level'] == 'Medium'),
        'High': sum(1 for p in predictions if p['risk_level'] == 'High'),
        'Critical': sum(1 for p in predictions if p['risk_level'] == 'Critical')
    }
    
    avg_fraud_prob = sum(p['fraud_probability'] for p in predictions) / total if total > 0 else 0
    
    return {
        'total_transactions': total,
        'fraud_detected': fraud_count,
        'fraud_rate': f"{(fraud_count/total*100):.2f}%" if total > 0 else "0%",
        'average_fraud_probability': f"{avg_fraud_prob:.4f}",
        'risk_distribution': risk_distribution,
        'timestamp': datetime.now().isoformat()
    }

def validate_transaction_data(df: pd.DataFrame) -> tuple:
    """Validate transaction dataframe"""
    
    required_cols = ['TransactionAmt', 'ProductCD']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for negative amounts
    if (df['TransactionAmt'] < 0).any():
        return False, "Transaction amounts cannot be negative"
    
    return True, "Valid"