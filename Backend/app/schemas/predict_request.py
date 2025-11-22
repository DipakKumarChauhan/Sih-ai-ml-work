"""
Request schemas for prediction API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class FeatureData(BaseModel):
    """Single feature data point"""
    year: float = Field(..., description="Year")
    month: float = Field(..., description="Month (1-12)")
    day: float = Field(..., description="Day of month (1-31)")
    hour: float = Field(..., description="Hour of day (0-23)")
    O3_forecast: Optional[float] = Field(None, description="O3 forecast value")
    NO2_forecast: Optional[float] = Field(None, description="NO2 forecast value")
    T_forecast: Optional[float] = Field(None, description="Temperature forecast")
    q_forecast: Optional[float] = Field(None, description="Specific humidity forecast")
    u_forecast: Optional[float] = Field(None, description="U wind component forecast")
    v_forecast: Optional[float] = Field(None, description="V wind component forecast")
    w_forecast: Optional[float] = Field(None, description="W wind component forecast")
    NO2_satellite: Optional[float] = Field(None, description="NO2 satellite data")
    HCHO_satellite: Optional[float] = Field(None, description="HCHO satellite data")
    ratio_satellite: Optional[float] = Field(None, description="Satellite ratio")
    NO2_sat_flag: Optional[float] = Field(None, description="NO2 satellite flag")
    HCHO_sat_flag: Optional[float] = Field(None, description="HCHO satellite flag")
    ratio_sat_flag: Optional[float] = Field(None, description="Ratio satellite flag")
    O3_target_lag1: Optional[float] = Field(None, description="O3 target lag 1 hour")
    O3_target_lag24: Optional[float] = Field(None, description="O3 target lag 24 hours")
    O3_target_lag168: Optional[float] = Field(None, description="O3 target lag 168 hours")
    NO2_target_lag1: Optional[float] = Field(None, description="NO2 target lag 1 hour")
    NO2_target_lag24: Optional[float] = Field(None, description="NO2 target lag 24 hours")
    NO2_target_lag168: Optional[float] = Field(None, description="NO2 target lag 168 hours")
    site_id: Optional[int] = Field(None, description="Site ID (required for unified model)", ge=1, le=7)
    
    @validator('month')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('Month must be between 1 and 12')
        return v
    
    @validator('day')
    def validate_day(cls, v):
        if not 1 <= v <= 31:
            raise ValueError('Day must be between 1 and 31')
        return v
    
    @validator('hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('Hour must be between 0 and 23')
        return v


class PredictRequest(BaseModel):
    """Request model for prediction"""
    input_data: List[FeatureData] = Field(..., description="List of feature data points")
    site_id: Optional[int] = Field(None, description="Site ID (1-7) or None for unified model", ge=1, le=7)
    forecast_hours: int = Field(24, description="Number of hours to forecast (24 or 48)", ge=1, le=48)
    
    @validator('forecast_hours')
    def validate_forecast_hours(cls, v):
        if v not in [24, 48]:
            raise ValueError('forecast_hours must be 24 or 48')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "input_data": [
                    {
                        "year": 2024.0,
                        "month": 5.0,
                        "day": 5.0,
                        "hour": 0.0,
                        "O3_forecast": 7.93,
                        "NO2_forecast": 69.81,
                        "T_forecast": 20.71,
                        "q_forecast": 11.12,
                        "u_forecast": -0.17,
                        "v_forecast": -1.87,
                        "w_forecast": -1.56
                    }
                ],
                "site_id": 1,
                "forecast_hours": 24
            }
        }

