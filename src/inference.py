import mlflow.pyfunc
import pandas as pd

# load model dari PRODUCTION
model = mlflow.pyfunc.load_model(
    "models:/NASA_RUL_Model/Production"
)

# contoh input (ambil 1 row dari dataset)
df = pd.read_csv("data/processed/cleaned_data.csv")
sample = df.drop(columns=["RUL"]).iloc[:1]

pred = model.predict(sample)

print("Prediction:", pred)