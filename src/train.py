import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import math
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("NASA_CMAPSS_RUL")

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--register", action="store_true")
args = parser.parse_args()

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop(columns=["RUL"])
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    if r2 < 0.05:
        raise ValueError(f"Model too weak: R2={r2}")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # SAVE MODEL LOCAL (INI PENTING)
    joblib.dump(model, "model.pkl")

    # SAVE TO MLFLOW
    mlflow.sklearn.log_model(model, "model")

    if args.register:
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "NASA_RUL_Model"
        )

    print(f"RMSE={rmse:.4f} R2={r2:.4f}")