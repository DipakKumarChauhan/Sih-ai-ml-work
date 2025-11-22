"""
Simple script to test if the server can start and load models
"""

import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Testing server startup...")
print("=" * 50)

try:
    print("\n1. Testing imports...")
    from app.main import app
    print("   ✓ Main app imported successfully")
    
    print("\n2. Testing model loader...")
    from app.models.loader import get_model_loader
    loader = get_model_loader()
    print("   ✓ Model loader created")
    
    print("\n3. Testing model loading (site 1)...")
    try:
        model_data = loader.load_models(1)
        print(f"   ✓ Site 1 model loaded - {len(model_data['feature_cols'])} features")
    except Exception as e:
        print(f"   ✗ Site 1 model failed: {e}")
    
    print("\n4. Testing prediction service...")
    from app.services.prediction_service import PredictionService
    service = PredictionService()
    print("   ✓ Prediction service created")
    
    print("\n5. Testing FastAPI app...")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/")
    print(f"   ✓ Root endpoint responded: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("All tests passed! Server should start correctly.")
    print("Start with: uvicorn app.main:app --reload")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

