"""
Configuration settings for the application
"""

import os
from pathlib import Path

# Base directory (backend root)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directory
DATA_DIR = BASE_DIR / "Data_SIH_2025"
MODELS_DIR = DATA_DIR / "models"

# API settings
API_V1_PREFIX = "/api/v1"
API_TITLE = "Air Pollution Forecasting API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI/ML-based advanced model for short-term forecast (24-48 hours) of surface O3 and NO2 for Delhi"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# CORS settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

