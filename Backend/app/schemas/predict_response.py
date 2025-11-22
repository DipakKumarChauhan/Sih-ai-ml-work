"""
Response schemas for prediction API
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionPoint(BaseModel):
    """Single prediction data point"""
    year: float = Field(..., description="Year")
    month: float = Field(..., description="Month")
    day: float = Field(..., description="Day")
    hour: float = Field(..., description="Hour")
    O3_target: float = Field(..., description="Predicted O3 concentration")
    NO2_target: float = Field(..., description="Predicted NO2 concentration")


class PredictResponse(BaseModel):
    """Response model for prediction"""
    success: bool = Field(..., description="Whether prediction was successful")
    site_id: Optional[int] = Field(None, description="Site ID used for prediction")
    forecast_hours: int = Field(..., description="Number of hours forecasted")
    predictions: List[PredictionPoint] = Field(..., description="List of predictions")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "site_id": 1,
                "forecast_hours": 24,
                "predictions": [
                    {
                        "year": 2024.0,
                        "month": 5.0,
                        "day": 5.0,
                        "hour": 0.0,
                        "O3_target": 45.2,
                        "NO2_target": 68.5
                    }
                ],
                "message": "Predictions generated successfully"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")


