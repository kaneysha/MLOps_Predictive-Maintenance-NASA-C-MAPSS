import joblib
import numpy as np

def test_model_load():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_model_prediction():
    model = joblib.load("models/model.pkl")

    sample = np.random.rand(1, 10)  # sesuaikan feature count
    pred = model.predict(sample)

    assert pred is not None
    assert len(pred) == 1