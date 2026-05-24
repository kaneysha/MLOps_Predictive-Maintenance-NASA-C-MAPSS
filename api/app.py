from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "ML API Running"}

@app.post("/predict")
def predict(
    feature1: float,
    feature2: float,
    feature3: float
):
    data = np.array([[feature1, feature2, feature3]])

    prediction = model.predict(data)

    return {
        "prediction": prediction.tolist()
    }