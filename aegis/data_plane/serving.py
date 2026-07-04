"""
Model Serving - FastAPI endpoint with shadow/canary routing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
from enum import Enum
import os


class ModelVersion(Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    SHADOW = "shadow"


class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    request_id: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ModelServer:
    """Model server with champion/challenger/shadow routing"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.champion_model = None
        self.challenger_model = None
        self.shadow_models: Dict[str, Any] = {}
        self.canary_percentage = 0.0
        self.shadow_enabled = False
        self.app = FastAPI(title="AEGIS Model Server")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            """Main prediction endpoint with routing"""
            return self._route_prediction(request)
        
        @self.app.post("/predict/champion")
        async def predict_champion(request: PredictionRequest):
            """Predict using champion model only"""
            return self._predict_with_model(request, ModelVersion.CHAMPION)
        
        @self.app.post("/predict/challenger")
        async def predict_challenger(request: PredictionRequest):
            """Predict using challenger model only"""
            return self._predict_with_model(request, ModelVersion.CHALLENGER)
        
        @self.app.post("/admin/load_champion")
        async def load_champion(model_path: str):
            """Load champion model"""
            success = self.load_champion(model_path)
            return {"success": success, "model_path": model_path}
        
        @self.app.post("/admin/load_challenger")
        async def load_challenger(model_path: str):
            """Load challenger model"""
            success = self.load_challenger(model_path)
            return {"success": success, "model_path": model_path}
        
        @self.app.post("/admin/promote_challenger")
        async def promote_challenger():
            """Promote challenger to champion"""
            success = self.promote_challenger()
            return {"success": success}
        
        @self.app.post("/admin/set_canary")
        async def set_canary(percentage: float):
            """Set canary traffic percentage"""
            self.set_canary_percentage(percentage)
            return {"canary_percentage": self.canary_percentage}
        
        @self.app.post("/admin/enable_shadow")
        async def enable_shadow(enabled: bool):
            """Enable/disable shadow mode"""
            self.shadow_enabled = enabled
            return {"shadow_enabled": self.shadow_enabled}
        
        @self.app.get("/admin/status")
        async def get_status():
            """Get server status"""
            return {
                "champion_loaded": self.champion_model is not None,
                "challenger_loaded": self.challenger_model is not None,
                "canary_percentage": self.canary_percentage,
                "shadow_enabled": self.shadow_enabled,
                "shadow_models_count": len(self.shadow_models)
            }
    
    def load_champion(self, model_path: str) -> bool:
        """Load champion model"""
        try:
            full_path = os.path.join(self.model_dir, model_path)
            self.champion_model = joblib.load(full_path)
            return True
        except Exception as e:
            print(f"Error loading champion: {e}")
            return False
    
    def load_challenger(self, model_path: str) -> bool:
        """Load challenger model"""
        try:
            full_path = os.path.join(self.model_dir, model_path)
            self.challenger_model = joblib.load(full_path)
            return True
        except Exception as e:
            print(f"Error loading challenger: {e}")
            return False
    
    def promote_challenger(self) -> bool:
        """Promote challenger to champion"""
        if self.challenger_model is None:
            return False
        
        self.champion_model = self.challenger_model
        self.challenger_model = None
        self.canary_percentage = 0.0
        return True
    
    def set_canary_percentage(self, percentage: float):
        """Set canary traffic percentage (0-1)"""
        self.canary_percentage = max(0.0, min(1.0, percentage))
    
    def _route_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Route prediction based on canary configuration"""
        import random
        
        # Determine which model to use
        if self.challenger_model is not None and random.random() < self.canary_percentage:
            model_version = ModelVersion.CHALLENGER
        else:
            model_version = ModelVersion.CHAMPION
        
        response = self._predict_with_model(request, model_version)
        
        # If shadow enabled, also run shadow models
        if self.shadow_enabled and self.shadow_models:
            self._run_shadow_predictions(request)
        
        return response
    
    def _predict_with_model(
        self,
        request: PredictionRequest,
        model_version: ModelVersion
    ) -> PredictionResponse:
        """Make prediction with specific model"""
        model = self.champion_model if model_version == ModelVersion.CHAMPION else self.challenger_model
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get probability if available
        probability = 0.0
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_df)[0, 1]
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version.value,
            request_id=request.request_id,
            metadata={"model_version": model_version.value}
        )
    
    def _run_shadow_predictions(self, request: PredictionRequest):
        """Run shadow predictions (non-blocking)"""
        features_df = pd.DataFrame([request.features])
        
        for name, model in self.shadow_models.items():
            try:
                prediction = model.predict(features_df)[0]
                # Log shadow prediction (in production, would send to metrics store)
                pass
            except Exception:
                pass
    
    def add_shadow_model(self, name: str, model: Any):
        """Add a shadow model"""
        self.shadow_models[name] = model


def create_app() -> FastAPI:
    """Create FastAPI application"""
    server = ModelServer()
    return server.app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8001)
