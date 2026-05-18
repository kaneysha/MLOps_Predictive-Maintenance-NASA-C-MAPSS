import pandas as pd
import joblib
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y_true = df["RUL"]

# Load model
model = joblib.load("model.pkl")

# Predict
y_pred = model.predict(X)

# Metrics REGRESSION (INI WAJIB)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")