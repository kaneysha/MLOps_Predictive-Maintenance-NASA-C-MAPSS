import os
import socket
import logging
from typing import List, Dict

import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

# SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", version="1.0")

# Custom metrics untuk data drift
prediction_score = Histogram(
    "model_prediction_score",
    "Distribution of model prediction scores",
    buckets=[0, 10, 20, 30, 50, 75, 100, 150, float("inf")]
)

request_count = Counter(
    "model_prediction_requests_total",
    "Total prediction requests"
)

# Expose /metrics endpoint
Instrumentator().instrument(app).expose(app)

MODEL = None
HOSTNAME = socket.gethostname()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-server:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# STARTUP LOAD MODEL
@app.on_event("startup")
def load_model():
    global MODEL
    try:
        model_uri = "models:/nasa_cmapss_model/Production"
        MODEL = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded from {model_uri}")
    except Exception as e:
        logger.error(f"MODEL LOAD ERROR: {e}")
        MODEL = None

# SCHEMA
class PredictionRequest(BaseModel):
    data: List[Dict[str, float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    served_by: str

# HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "hostname": HOSTNAME
    }

# PREDICT ENDPOINT
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    df = pd.DataFrame(req.data)
    preds = MODEL.predict(df)

    # Track metrics untuk data drift
    for score in preds:
        prediction_score.observe(float(score))
    request_count.inc()

    return PredictionResponse(
        predictions=preds.tolist(),
        served_by=HOSTNAME
    )

# LIST MODELS
@app.get("/models")
def list_models():
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    return [
        {
            "name": m.name,
            "versions": [
                {"version": v.version, "stage": v.current_stage}
                for v in m.latest_versions
            ]
        }
        for m in models
    ]