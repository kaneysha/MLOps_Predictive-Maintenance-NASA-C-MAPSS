import pandas as pd
import glob
import os

# Path
RAW_PATH = 'data/raw/*.csv'
LOG_FILE = 'data/processed_log.txt'
OUTPUT_FILE = 'data/processed/cleaned_data.csv'

# Ambil semua file raw
all_files = glob.glob(RAW_PATH)

# Load file yang sudah pernah diproses
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

    try:
        df = pd.read_csv(file)

        # HANDLE MISSING VALUE
        df = df.ffill()

        # HITUNG RUL (WAJIB NASA)
        if 'unit' in df.columns and 'cycle' in df.columns:
            rul_df = df.groupby('unit')['cycle'].max().reset_index()
            rul_df.columns = ['unit', 'max_cycle']

            df = df.merge(rul_df, on='unit')
            df['RUL'] = df['max_cycle'] - df['cycle']

            df = df.drop(columns=['max_cycle'])

        else:
            print(f"[WARNING] Kolom 'unit' atau 'cycle' tidak ditemukan di {file}")

        # DROP KOLOM TIDAK PERLU
        if 'unit' in df.columns:
            df = df.drop(columns=['unit'])

        # SIMPAN KE LIST
        new_data.append(df)
        new_processed_files.append(file)

    except Exception as e:
        print(f"[ERROR] Gagal proses {file}: {e}")

# GABUNG & SIMPAN
if new_data:
    final_df = pd.concat(new_data, ignore_index=True)

    os.makedirs('data/processed', exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        final_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print("[INFO] Data ditambahkan ke file existing")
    else:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("[INFO] File baru dibuat")

    # Update log file
    with open(LOG_FILE, 'a') as f:
        for file in new_processed_files:
            f.write(file + '\n')

    print("[SUCCESS] Data baru diproses & disimpan")

else:
    print("[INFO] Tidak ada data baru")