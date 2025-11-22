"""
Prediction API routes for air pollution forecasting
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional
import pandas as pd
import logging

from app.schemas.predict_request import PredictRequest
from app.schemas.predict_response import PredictResponse, ErrorResponse
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])

# Initialize prediction service
# Path will be resolved relative to backend root
prediction_service = PredictionService(models_dir="Data_SIH_2025/models")


@router.post(
    "/site/{site_id}",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution for a specific site",
    description="Make predictions for O3 and NO2 for a specific monitoring site (1-7)"
)
async def predict_site(
    site_id: int,
    request: PredictRequest
):
    """
    Predict air pollution for a specific site
    
    Args:
        site_id: Site number (1-7)
        request: Prediction request with input data and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Validate site_id
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )
        
        # Override site_id in request if provided
        if request.site_id is not None and request.site_id != site_id:
            logger.warning(f"Site ID in request ({request.site_id}) differs from path ({site_id}), using path value")
        
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Validate input data
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=site_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=request.forecast_hours
        )
        
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions for site {site_id}"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@router.post(
    "/unified",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution using unified model",
    description="Make predictions for O3 and NO2 using the unified model (requires site_id in input data)"
)
async def predict_unified(
    request: PredictRequest
):
    """
    Predict air pollution using unified model
    
    Args:
        request: Prediction request with input data (must include site_id) and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Validate input data (unified model requires site_id)
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=None)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=None,  # None means unified model
            forecast_hours=request.forecast_hours
        )
        
        return PredictResponse(
            success=True,
            site_id=None,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions using unified model"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Unified model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unified model not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@router.post(
    "/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution (auto-detect site or unified)",
    description="Make predictions using site_id from request or unified model if site_id is None"
)
async def predict(
    request: PredictRequest
):
    """
    Predict air pollution (auto-detect model based on site_id)
    
    Args:
        request: Prediction request with input data and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Determine which model to use
        site_id = request.site_id
        
        # If site_id is None, check if it's in input data (for unified model)
        if site_id is None:
            if input_data and 'site_id' in input_data[0]:
                # Use unified model
                site_id = None
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either site_id must be provided in request or in input_data for unified model"
                )
        
        # Validate input data
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=site_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=request.forecast_hours
        )
        
        model_type = f"site {site_id}" if site_id else "unified"
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions using {model_type} model"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )

