"""
Health check API routes
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, List
import os
from pathlib import Path

from app.models.loader import get_model_loader

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get(
    "/",
    summary="Health check",
    description="Check if the API is running"
)
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Air Pollution Forecasting API"
    }


@router.get(
    "/models",
    summary="Check model availability",
    description="Check which models are available and loaded"
)
async def check_models():
    """Check which models are available"""
    # Resolve path relative to backend root
    backend_root = Path(__file__).parent.parent.parent
    models_dir = backend_root / "Data_SIH_2025" / "models"
    
    available_models = []
    for site_id in range(1, 8):
        model_file = models_dir / f"site_{site_id}_models.joblib"
        features_file = models_dir / f"site_{site_id}_features.txt"
        if model_file.exists() and features_file.exists():
            available_models.append({
                "site_id": site_id,
                "model_file": str(model_file),
                "features_file": str(features_file),
                "status": "available"
            })
        else:
            available_models.append({
                "site_id": site_id,
                "status": "not_found"
            })
    
    # Check unified model
    unified_model_file = models_dir / "site_unified_models.joblib"
    unified_features_file = models_dir / "site_unified_features.txt"
    unified_status = {
        "site_id": "unified",
        "status": "available" if unified_model_file.exists() and unified_features_file.exists() else "not_found"
    }
    if unified_status["status"] == "available":
        unified_status["model_file"] = str(unified_model_file)
        unified_status["features_file"] = str(unified_features_file)
    
    return {
        "status": "ok",
        "models": available_models,
        "unified_model": unified_status
    }


@router.get(
    "/models/{site_id}",
    summary="Check specific model",
    description="Check if a specific model is available and get its details"
)
async def check_model(site_id: int):
    """Check if a specific model is available"""
    if not 1 <= site_id <= 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Site ID must be between 1 and 7, got {site_id}"
        )
    
    try:
        model_loader = get_model_loader()
        model_data = model_loader.load_models(site_id, use_cache=False)
        
        return {
            "status": "available",
            "site_id": site_id,
            "feature_count": len(model_data['feature_cols']),
            "features": model_data['feature_cols'],
            "saved_at": model_data.get('saved_at', 'Unknown')
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

