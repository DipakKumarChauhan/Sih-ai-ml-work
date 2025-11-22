"""
Prediction service for air pollution forecasting
Handles feature preparation and model predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from app.models.loader import get_model_loader


class PredictionService:
    """Service for making air pollution predictions"""
    
    def __init__(self, models_dir: str = "Data_SIH_2025/models"):
        """
        Initialize prediction service
        
        Args:
            models_dir: Path to directory containing model files (relative to backend root or absolute)
        """
        self.model_loader = get_model_loader(models_dir)
    
    def process_input_data(self, df: pd.DataFrame, site_id: Optional[int] = None) -> pd.DataFrame:
        """
        Process input data for prediction (matches training code's process_unseen_data)
        
        Args:
            df: Input dataframe with raw features
            site_id: Site ID (required for unified model)
            
        Returns:
            Processed DataFrame ready for feature preparation
        """
        df = df.copy()
        
        # Convert to datetime if year/month/day/hour are present
        if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
            # Convert to int (matching training code)
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['hour'] = df['hour'].astype(int)
            
            # Create datetime column and sort
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Handle satellite data (same as training code)
        satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
        if 'datetime' in df.columns:
            df['date'] = df['datetime'].dt.date
            
            for col in satellite_cols:
                if col in df.columns:
                    # Forward fill and backward fill grouped by date
                    df[col] = df.groupby('date')[col].transform(lambda x: x.ffill().bfill())
                    flag_col = col.replace('_satellite', '_sat_flag')
                    df[flag_col] = (~df[col].isnull()).astype(int)
            
            if 'date' in df.columns:
                df = df.drop('date', axis=1)
        
        # For unseen data, create lag features from forecast values if they don't exist
        # This matches the training code's process_unseen_data function
        if 'O3_forecast' in df.columns:
            if 'O3_target_lag1' not in df.columns:
                df['O3_target_lag1'] = df['O3_forecast'].shift(1).fillna(df['O3_forecast'].iloc[0] if len(df) > 0 else 0)
            if 'O3_target_lag24' not in df.columns:
                df['O3_target_lag24'] = df['O3_forecast'].shift(24).fillna(df['O3_forecast'].iloc[0] if len(df) > 0 else 0)
            if 'O3_target_lag168' not in df.columns:
                df['O3_target_lag168'] = df['O3_forecast'].shift(168).fillna(df['O3_forecast'].iloc[0] if len(df) > 0 else 0)
        
        if 'NO2_forecast' in df.columns:
            if 'NO2_target_lag1' not in df.columns:
                df['NO2_target_lag1'] = df['NO2_forecast'].shift(1).fillna(df['NO2_forecast'].iloc[0] if len(df) > 0 else 0)
            if 'NO2_target_lag24' not in df.columns:
                df['NO2_target_lag24'] = df['NO2_forecast'].shift(24).fillna(df['NO2_forecast'].iloc[0] if len(df) > 0 else 0)
            if 'NO2_target_lag168' not in df.columns:
                df['NO2_target_lag168'] = df['NO2_forecast'].shift(168).fillna(df['NO2_forecast'].iloc[0] if len(df) > 0 else 0)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, site_id: Optional[int] = None, is_training: bool = False) -> pd.DataFrame:
        """
        Prepare features for prediction (matches training code's prepare_features)
        
        Args:
            df: Input dataframe with raw features
            site_id: Site ID (required for unified model)
            is_training: Whether this is training data (affects which columns to exclude)
            
        Returns:
            DataFrame with prepared features
        """
        df = df.copy()
        
        # Columns to exclude from features (matching training code)
        exclude_cols = ['O3_target', 'NO2_target', 'datetime']
        
        if not is_training:
            # For unseen data, we don't have targets, but we still exclude datetime
            exclude_cols = ['datetime']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select features
        X = df[feature_cols].copy()
        
        # Fill NaN values (for lag features at the beginning of time series)
        X = X.fillna(0)
        
        return X
    
    def ensure_feature_columns(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """
        Ensure all required feature columns are present in the dataframe
        
        Args:
            df: Input dataframe
            required_features: List of required feature column names
            
        Returns:
            DataFrame with all required features (missing ones filled with 0)
        """
        df = df.copy()
        
        # Add missing features with default value 0
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match required order
        df = df[required_features]
        
        return df
    
    def predict(
        self,
        input_data: pd.DataFrame,
        site_id: Optional[int] = None,
        forecast_hours: int = 24
    ) -> pd.DataFrame:
        """
        Make predictions for O3 and NO2 (matches training code's predict_unseen_data)
        
        Args:
            input_data: DataFrame with input features
            site_id: Site number (1-7) or None for unified model
            forecast_hours: Number of hours to forecast (24 or 48)
            
        Returns:
            DataFrame with predictions including:
                - year, month, day, hour
                - O3_target: Predicted O3 values
                - NO2_target: Predicted NO2 values
        """
        # Load models
        model_data = self.model_loader.load_models(site_id)
        model_o3 = model_data['model_o3']
        model_no2 = model_data['model_no2']
        feature_cols = model_data['feature_cols']
        
        # Process input data (same as training code's process_unseen_data)
        processed_df = self.process_input_data(input_data, site_id)
        
        # Prepare features (matching training code's prepare_features)
        X = self.prepare_features(processed_df, site_id, is_training=False)
        
        # Ensure all required features are present (matching training code)
        # Training code adds missing features with 0 and reorders columns
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training (matching training code)
        X = X[feature_cols]
        
        # Make predictions
        predictions_o3 = model_o3.predict(X)
        predictions_no2 = model_no2.predict(X)
        
        # Create output dataframe (matching training code's output format)
        if 'datetime' in processed_df.columns:
            output_df = processed_df[['year', 'month', 'day', 'hour', 'datetime']].copy()
        else:
            output_df = processed_df[['year', 'month', 'day', 'hour']].copy()
        
        output_df['O3_target'] = predictions_o3
        output_df['NO2_target'] = predictions_no2
        
        # Limit to requested forecast hours
        if len(output_df) > forecast_hours:
            output_df = output_df.head(forecast_hours)
        
        return output_df
    
    def predict_from_dict(
        self,
        input_data: List[Dict],
        site_id: Optional[int] = None,
        forecast_hours: int = 24
    ) -> List[Dict]:
        """
        Make predictions from dictionary input
        
        Args:
            input_data: List of dictionaries with input features
            site_id: Site number (1-7) or None for unified model
            forecast_hours: Number of hours to forecast (24 or 48)
            
        Returns:
            List of dictionaries with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Add site_id if using unified model
        if site_id is None and 'site_id' not in df.columns:
            raise ValueError("site_id must be provided in input data for unified model")
        
        # Make predictions
        predictions_df = self.predict(df, site_id, forecast_hours)
        
        # Convert back to list of dictionaries
        return predictions_df.to_dict('records')
    
    def validate_input_data(self, input_data: List[Dict], site_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate input data format
        
        Args:
            input_data: List of dictionaries with input features
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not input_data:
            return False, "Input data is empty"
        
        # Check required columns
        required_cols = ['year', 'month', 'day', 'hour']
        for i, record in enumerate(input_data):
            for col in required_cols:
                if col not in record:
                    return False, f"Missing required column '{col}' in record {i}"
        
        # Check unified model requirement
        if site_id is None:
            if 'site_id' not in input_data[0]:
                return False, "site_id is required in input data for unified model"
        
        return True, None

