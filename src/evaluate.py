import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import json

df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load("model.pkl")
pred = model.predict(X)

acc = accuracy_score(y, pred)

print(f"Accuracy: {acc}")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)