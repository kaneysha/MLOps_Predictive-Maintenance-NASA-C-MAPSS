import os
import socket
import logging
from typing import List, Dict, Any

import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter, Gauge

# ======================
# SETUP
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", version="1.0")

HOSTNAME = socket.gethostname()

# ======================
# METRICS
# ======================
prediction_score = Histogram(
    "model_prediction_score",
    "Distribution of model prediction scores",
    buckets=[0, 10, 20, 30, 50, 75, 100, 150, float("inf")]
)

request_count = Counter(
    "model_prediction_requests_total",
    "Total prediction requests"
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

data_drift_score = Gauge(
    "data_drift_score",
    "Data drift score"
)

Instrumentator().instrument(app).expose(app)

# ======================
# MODEL
# ======================
MODEL = None

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-server:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# baseline (INI PENTING)
BASELINE_MEAN = 5000  # dari sensor normal NASA CMAPSS (approx anchor)

# ======================
# STARTUP
# ======================
@app.on_event("startup")
def load_model():
    global MODEL
    try:
        model_uri = "models:/nasa_cmapss_model/Production"
        MODEL = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Model loaded from {model_uri}")

        model_accuracy.set(1.0)
        data_drift_score.set(0.0)

    except Exception as e:
        MODEL = None
        logger.error(f"MODEL LOAD ERROR: {e}")


# ======================
# SCHEMA
# ======================
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[float]
    served_by: str


# ======================
# HEALTH
# ======================
@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL else "degraded",
        "model_loaded": MODEL is not None,
        "hostname": HOSTNAME
    }


# ======================
# PREDICT
# ======================
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame(req.data)
        preds = MODEL.predict(df)
        preds_list = list(preds)

        # ======================
        # METRICS BASIC
        # ======================
        request_count.inc()

        for score in preds_list:
            prediction_score.observe(float(score))

        # ======================
        # DRIFT DETECTION (REALISTIC)
        # ======================

        # ambil mean input data
        input_mean = float(df.mean(numeric_only=True).mean())

        # hitung deviation dari baseline
        drift = abs(input_mean - BASELINE_MEAN) / BASELINE_MEAN

        # clamp 0 - 1
        drift_score = min(drift, 1.0)

        data_drift_score.set(drift_score)

        # ======================
        # ACCURACY SIMULATION
        # ======================

        # semakin besar drift → accuracy turun
        accuracy = max(0.2, 1.0 - drift_score)

        model_accuracy.set(accuracy)

        return PredictionResponse(
            predictions=preds_list,
            served_by=HOSTNAME
        )

    except Exception as e:
        logger.error(f"PREDICT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# MODELS LIST
# ======================
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