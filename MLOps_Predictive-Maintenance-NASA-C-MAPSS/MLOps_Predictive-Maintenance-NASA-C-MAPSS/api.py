from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# load model sekali saat API start
model = mlflow.pyfunc.load_model(
    "runs:/d489fdf1c716fa0e789b155f5e757a3e/model"
)

@app.get("/")
def home():
    return {"status": "API is running"}

@app.get("/predict")
def predict():
    # contoh input dummy
    data = [[1, 2, 3, 4]]

    pred = model.predict(data)

    return {"prediction": pred.tolist()}