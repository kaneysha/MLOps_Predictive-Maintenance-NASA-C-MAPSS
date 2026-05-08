import pandas as pd
import os

def test_data_loading():
    path = "data/raw"

    assert os.path.exists(path), "Jalankan dvc pull dulu"

def test_structure():
    # pakai sample kecil / dummy check
    assert True