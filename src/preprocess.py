import pandas as pd
import glob
import os

RAW_PATH    = 'data/raw/*.csv'
LOG_FILE    = 'data/processed_log.txt'
OUTPUT_FILE = 'data/processed/cleaned_data.csv'

all_files = glob.glob(RAW_PATH)

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

new_data           = []
new_processed_files = []

for file in sorted(all_files):          # sorted biar urutan konsisten
    if file in processed_files:
        print(f"[SKIP] {file}")
        continue
    print(f"[INFO] Processing {file}")
    df = pd.read_csv(file)
    df = df.ffill()                      # fix: ganti fillna deprecated
    # JANGAN drop 'unit' di sini — masih butuh buat RUL
    new_data.append(df)
    new_processed_files.append(file)

if new_data:
    os.makedirs('data/processed', exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat([existing, pd.concat(new_data, ignore_index=True)],
                              ignore_index=True)
    else:
        combined = pd.concat(new_data, ignore_index=True)

    max_cycle        = combined.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    combined         = combined.merge(max_cycle, on='unit')
    combined['RUL']  = combined['max_cycle'] - combined['cycle']
    combined         = combined.drop(columns=['max_cycle', 'unit'])

    combined.to_csv(OUTPUT_FILE, index=False)

    with open(LOG_FILE, 'a') as f:
        for file in new_processed_files:
            f.write(file + '\n')

    print(f"[SUCCESS] Total rows: {len(combined)}, kolom: {list(combined.columns)}")
else:
    print("[INFO] Tidak ada data baru")