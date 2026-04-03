import pandas as pd
import time
import os
from datetime import datetime

df = pd.read_csv('data/external/train_FD001.txt', sep=' ', header=None)

df = df.dropna(axis=1)

columns = ['unit', 'cycle'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
df.columns = columns

os.makedirs('data/raw', exist_ok=True)

batch_size = 50

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/data_{timestamp}.csv"

    batch.to_csv(filename, index=False)

    print(f"Saved: {filename}")

    time.sleep(5)