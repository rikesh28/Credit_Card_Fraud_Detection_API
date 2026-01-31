from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class TransactionInput(BaseModel):
    """Schema for single transaction prediction"""
    
    TransactionAmt: float = Field(..., description="Transaction amount in USD", ge=0)
    ProductCD: str = Field(..., description="Product code (W, C, H, S, R)")
    card1: int = Field(..., description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card4: Optional[str] = Field(None, description="Card type")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    card6: Optional[str] = Field(None, description="Card category")
    addr1: Optional[float] = Field(None, description="Address 1")
    addr2: Optional[float] = Field(None, description="Address 2")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TransactionAmt": 150.50,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 150.0,
                "card3": 150.0,
                "card4": "discover",
                "card5": 226.0,
                "card6": "debit",
                "addr1": 315.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com"
            }
        }
    )

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str

class HealthResponse(BaseModel):
    """Schema for health check"""
    
    status: str
    ml_model_loaded: bool  
    ml_model_type: str     
    
    model_config = ConfigDict(protected_namespaces=())  