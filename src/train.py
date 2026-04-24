import pandas as pd
import numpy as np
import argparse
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

GDRIVE     = "/content/drive/MyDrive/SEMESTER 6/MLOPS/Experiment"
MLFLOW_URI = f"file://{GDRIVE}/mlruns"
DATA_PATH  = "data/processed/cleaned_data.csv"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("cmapss-continual-learning")

parser = argparse.ArgumentParser()
parser.add_argument("--batch",         type=int,   default=0)
parser.add_argument("--model",         type=str,   default="random_forest",
                    choices=["random_forest","gradient_boosting","ridge"])
parser.add_argument("--n_estimators",  type=int,   default=100)
parser.add_argument("--max_depth",     type=int,   default=10)
parser.add_argument("--learning_rate", type=float, default=0.1)
args = parser.parse_args()

if not os.path.exists(DATA_PATH):
    print("[TRAIN] Data belum ada, skip."); exit(0)

df = pd.read_csv(DATA_PATH)
if len(df) < 100:
    print(f"[TRAIN] Data cuma {len(df)} rows, minimal 100. Skip."); exit(0)

feature_cols = [c for c in df.columns if c != 'RUL']
X = df[feature_cols].values
y = df['RUL'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def build_model():
    if args.model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42, n_jobs=-1
        )
    elif args.model == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )
    elif args.model == "ridge":
        return Ridge(alpha=args.learning_rate * 100)

run_name = f"{args.model}_batch{args.batch:04d}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "model_type":    args.model,
        "batch_number":  args.batch,
        "n_estimators":  args.n_estimators,
        "max_depth":     args.max_depth,
        "learning_rate": args.learning_rate,
        "train_size":    len(X_train),
        "total_rows":    len(df),
    })

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = model.score(X_test, y_test)

    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2_score": r2})
    mlflow.sklearn.log_model(model, "model")

    if hasattr(model, "feature_importances_"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fi  = pd.Series(model.feature_importances_, index=feature_cols)
        fig, ax = plt.subplots(figsize=(7, 4))
        fi.nlargest(10).plot(kind="barh", ax=ax)
        ax.set_title(f"Feature importance — {run_name}")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

    print(f"[TRAIN] {run_name} | RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f}")
    print(f"[TRAIN] Run logged → {MLFLOW_URI}")