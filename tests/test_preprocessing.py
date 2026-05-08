from src.preprocess import preprocess_data  # sesuaikan path kamu
import pandas as pd

def test_preprocessing_output_shape():
    df = pd.read_csv("data/raw/train.csv")
    X = preprocess_data(df)

    assert X.shape[0] > 0

def test_preprocessing_no_nan():
    df = pd.read_csv("data/raw/train.csv")
    X = preprocess_data(df)

    assert not X.isnull().any().any()