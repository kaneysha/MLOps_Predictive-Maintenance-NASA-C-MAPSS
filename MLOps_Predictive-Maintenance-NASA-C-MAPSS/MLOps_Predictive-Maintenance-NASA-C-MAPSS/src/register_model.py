import json
import pandas as pd
import joblib
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y_true = df["RUL"]

model = joblib.load("model.pkl")

y_pred = model.predict(X)

rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

metrics = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}

print(metrics)

# 🔥 INI YANG KAMU LUPA
with open("metrics.json", "w") as f:
    json.dump(metrics, f)