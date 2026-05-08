import pandas as pd

def test_data_loading():
    df = pd.read_csv("data/raw/train.csv")

    assert df.shape[0] > 0

def test_required_columns_exist():
    df = pd.read_csv("data/raw/train.csv")

    required_cols = ["unit_number", "time_in_cycles"]
    for col in required_cols:
        assert col in df.columns