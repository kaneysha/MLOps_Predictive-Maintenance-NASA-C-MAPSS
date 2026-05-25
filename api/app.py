from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="ML Model Inference API",
    description="API untuk inferensi model ML menggunakan MLflow",
    version="1.0.0"
)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variable untuk menyimpan model
loaded_model = None
model_info = {}

# Pydantic models untuk request/response
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"feature1": 5.1, "feature2": 3.5, "feature3": 1.4},
                    {"feature1": 6.2, "feature2": 2.8, "feature3": 4.8}
                ]
            }
        }

class PredictionResponse(BaseModel):
    predictions: List[Any]
    model_name: str
    model_version: str
    model_stage: str

class ModelLoadRequest(BaseModel):
    model_name: str
    version: str = None  # Jika None, akan load latest version
    stage: str = None    # Bisa: "Production", "Staging", "Archived", None
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "my_model",
                "version": "1",
                "stage": "Production"
            }
        }

# Health check endpoint
@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": loaded_model is not None,
        "model_info": model_info if loaded_model else None
    }

# Load model endpoint
@app.post("/load-model")
async def load_model(request: ModelLoadRequest):
    global loaded_model, model_info
    
    try:
        logger.info(f"Loading model: {request.model_name}")
        
        # Konstruksi model URI
        if request.stage:
            # Load berdasarkan stage (Production, Staging, etc)
            model_uri = f"models:/{request.model_name}/{request.stage}"
            logger.info(f"Loading model by stage: {model_uri}")
        elif request.version:
            # Load berdasarkan version spesifik
            model_uri = f"models:/{request.model_name}/{request.version}"
            logger.info(f"Loading model by version: {model_uri}")
        else:
            # Load latest version
            model_uri = f"models:/{request.model_name}/latest"
            logger.info(f"Loading latest model: {model_uri}")
        
        # Load model menggunakan MLflow
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Simpan informasi model
        model_info = {
            "model_name": request.model_name,
            "model_uri": model_uri,
            "version": request.version or "latest",
            "stage": request.stage or "N/A"
        }
        
        logger.info(f"Model loaded successfully: {model_info}")
        
        return {
            "message": "Model loaded successfully",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global loaded_model, model_info
    
    # Cek apakah model sudah di-load
    if loaded_model is None:
        raise HTTPException(
            status_code=400, 
            detail="No model loaded. Please load a model first using /load-model endpoint"
        )
    
    try:
        # Convert input data ke DataFrame
        input_df = pd.DataFrame(request.data)
        logger.info(f"Received prediction request with {len(input_df)} samples")
        
        # Lakukan prediksi
        predictions = loaded_model.predict(input_df)
        
        # Convert numpy array ke list untuk JSON serialization
        predictions_list = predictions.tolist()
        
        logger.info(f"Predictions completed: {len(predictions_list)} results")
        
        return PredictionResponse(
            predictions=predictions_list,
            model_name=model_info.get("model_name", "unknown"),
            model_version=model_info.get("version", "unknown"),
            model_stage=model_info.get("stage", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model info endpoint
@app.get("/model-info")
async def get_model_info():
    if loaded_model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    return {
        "model_info": model_info,
        "model_loaded": True
    }

# List models endpoint
@app.get("/list-models")
async def list_models():
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        models_list = []
        for rm in registered_models:
            models_list.append({
                "name": rm.name,
                "latest_versions": [
                    {
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id
                    }
                    for mv in rm.latest_versions
                ]
            })
        
        return {
            "models": models_list,
            "total": len(models_list)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")