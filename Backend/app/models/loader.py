"""
Model loader utility for loading trained ML models
Supports loading models for 7 individual sites and 1 unified model
"""

import os
import joblib
from typing import Dict, Optional, Tuple
from pathlib import Path


class ModelLoader:
    """Loads and manages ML models for air pollution forecasting"""
    
    def __init__(self, models_dir: str = "Data_SIH_2025/models"):
        """
        Initialize model loader
        
        Args:
            models_dir: Path to directory containing model files (relative to backend root or absolute)
        """
        # Convert to Path and resolve relative to backend root if relative
        models_path = Path(models_dir)
        if not models_path.is_absolute():
            # Try relative to current working directory first, then relative to this file
            backend_root = Path(__file__).parent.parent.parent
            models_path = backend_root / models_dir
        self.models_dir = models_path
        self._models_cache: Dict[str, Dict] = {}
        self._feature_cols_cache: Dict[str, list] = {}
    
    def _get_model_path(self, site_id: Optional[int] = None) -> Path:
        """
        Get model file path for a site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Path to model file
        """
        if site_id is None:
            model_file = self.models_dir / "site_unified_models.joblib"
        else:
            if not 1 <= site_id <= 7:
                raise ValueError(f"Site ID must be between 1 and 7, got {site_id}")
            model_file = self.models_dir / f"site_{site_id}_models.joblib"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        return model_file
    
    def _get_features_path(self, site_id: Optional[int] = None) -> Path:
        """
        Get features file path for a site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Path to features file
        """
        if site_id is None:
            features_file = self.models_dir / "site_unified_features.txt"
        else:
            if not 1 <= site_id <= 7:
                raise ValueError(f"Site ID must be between 1 and 7, got {site_id}")
            features_file = self.models_dir / f"site_{site_id}_features.txt"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        return features_file
    
    def load_models(self, site_id: Optional[int] = None, use_cache: bool = True) -> Dict:
        """
        Load models for a specific site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            use_cache: Whether to use cached models if already loaded
            
        Returns:
            Dictionary containing:
                - 'model_o3': Trained model for O3 prediction
                - 'model_no2': Trained model for NO2 prediction
                - 'feature_cols': List of feature column names
                - 'site_id': Site ID or 'unified'
        """
        cache_key = f"site_{site_id}" if site_id else "unified"
        
        # Return cached model if available
        if use_cache and cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        # Load model file
        model_path = self._get_model_path(site_id)
        model_data = joblib.load(model_path)
        
        # Load feature columns
        features_path = self._get_features_path(site_id)
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
        # Extract models
        result = {
            'model_o3': model_data.get('model_o3'),
            'model_no2': model_data.get('model_no2'),
            'feature_cols': feature_cols,
            'site_id': site_id if site_id else 'unified',
            'saved_at': model_data.get('saved_at', 'Unknown')
        }
        
        # Cache the result
        if use_cache:
            self._models_cache[cache_key] = result
            self._feature_cols_cache[cache_key] = feature_cols
        
        return result
    
    def get_feature_columns(self, site_id: Optional[int] = None) -> list:
        """
        Get feature columns for a specific site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            List of feature column names
        """
        cache_key = f"site_{site_id}" if site_id else "unified"
        
        if cache_key in self._feature_cols_cache:
            return self._feature_cols_cache[cache_key]
        
        # Load features file
        features_path = self._get_features_path(site_id)
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
        self._feature_cols_cache[cache_key] = feature_cols
        return feature_cols
    
    def clear_cache(self):
        """Clear the model cache"""
        self._models_cache.clear()
        self._feature_cols_cache.clear()


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(models_dir: str = "Data_SIH_2025/models") -> ModelLoader:
    """
    Get or create global model loader instance
    
    Args:
        models_dir: Path to directory containing model files
        
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(models_dir)
    return _model_loader

