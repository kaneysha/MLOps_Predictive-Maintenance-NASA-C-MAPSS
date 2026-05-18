import pandas as pd
import joblib
import os


# =========================
# 1. TEST DATA EXISTENCE
# =========================
def test_data_exists():
    path = "data/processed/cleaned_data.csv"
    assert os.path.exists(path), "Dataset tidak ditemukan!"


# =========================
# 2. TEST DATA STRUCTURE CMAPSS
# =========================
def test_data_structure():
    df = pd.read_csv("data/processed/cleaned_data.csv")

    # CMAPSS biasanya punya RUL
    assert "RUL" in df.columns, "Kolom RUL tidak ada!"

    # minimal tidak kosong
    assert len(df) > 0, "Dataset kosong!"


# =========================
# 3. TEST NO NULL VALUES
# =========================
def test_no_missing_values():
    df = pd.read_csv("data/processed/cleaned_data.csv")

    assert df.isnull().sum().sum() == 0, "Masih ada missing value!"


# =========================
# 4. TEST MODEL LOAD
# =========================
def test_model_load():
    model_path = "model.pkl"
    assert os.path.exists(model_path), "Model belum di-training!"

    model = joblib.load(model_path)
    assert model is not None


# =========================
# 5. TEST MODEL PREDICTION (CMAPSS REAL TEST)
# =========================
def test_model_predict_shape():
    df = pd.read_csv("data/processed/cleaned_data.csv")

    X = df.drop("RUL", axis=1)

    model = joblib.load("model.pkl")

    preds = model.predict(X)

    # jumlah prediksi harus sama dengan data
    assert len(preds) == len(X), "Shape prediction tidak sesuai!"


# =========================
# 6. SANITY CHECK OUTPUT RANGE (CMAPSS RUL)
# =========================
def test_prediction_range():
    df = pd.read_csv("data/processed/cleaned_data.csv")

    X = df.drop("RUL", axis=1)

    model = joblib.load("model.pkl")

    preds = model.predict(X)

    # RUL tidak boleh negatif (CMAPSS logic)
    assert (preds >= 0).all(), "Prediksi RUL negatif!"