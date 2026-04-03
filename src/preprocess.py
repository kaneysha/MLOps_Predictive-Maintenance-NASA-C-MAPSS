import pandas as pd
import glob
import os

RAW_PATH = 'data/raw/*.csv'
LOG_FILE = 'data/processed_log.txt'
OUTPUT_FILE = 'data/processed/cleaned_data.csv'

all_files = glob.glob(RAW_PATH)

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

new_data = []
new_processed_files = []

for file in all_files:
    if file in processed_files:
        print(f"[SKIP] {file} sudah diproses")
        continue

    print(f"[INFO] Processing {file}")

    df = pd.read_csv(file)

    df = df.fillna(method='ffill')
    if 'unit' in df.columns:
        df = df.drop(columns=['unit'])

    new_data.append(df)
    new_processed_files.append(file)

if new_data:
    final_df = pd.concat(new_data, ignore_index=True)

    os.makedirs('data/processed', exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        final_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        final_df.to_csv(OUTPUT_FILE, index=False)

    with open(LOG_FILE, 'a') as f:
        for file in new_processed_files:
            f.write(file + '\n')

    print("[SUCCESS] Data baru diproses & disimpan")

else:
    print("[INFO] Tidak ada data baru")