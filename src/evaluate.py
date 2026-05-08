import json
import sys

THRESHOLD = 0.8

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print("Model tidak memenuhi threshold")
    sys.exit(1)
else:
    print("Model lolos validasi")