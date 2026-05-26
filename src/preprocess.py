import pandas as pd
import glob
import os

# PATH CONFIG
RAW_PATH = "data/raw/*.csv"
OUTPUT_FILE = "data/processed/cleaned_data.csv"
LOG_FILE = "data/processed_log.txt"

# GET RAW FILES
all_files = glob.glob(RAW_PATH)

# LOAD PROCESSED LOG
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

new_data = []
new_processed_files = []

# PROCESS FILES
for file in all_files:

    if file in processed_files:
        print(f"[SKIP] {file} already processed")
        continue

    print(f"[INFO] Processing {file}")

    try:

        df = pd.read_csv(file)

        # HANDLE MISSING VALUE
        df = df.ffill()

        # COMPUTE RUL
        if "unit" in df.columns and "cycle" in df.columns:

            rul_df = (
                df.groupby("unit")["cycle"]
                .max()
                .reset_index()
            )

            rul_df.columns = ["unit", "max_cycle"]

            df = df.merge(rul_df, on="unit")

            df["RUL"] = (
                df["max_cycle"] - df["cycle"]
            )

            df = df.drop(columns=["max_cycle"])

        else:
            print(
                f"[WARNING] unit/cycle not found in {file}"
            )

        # DROP UNIT ID
        if "unit" in df.columns:
            df = df.drop(columns=["unit"])

        # SAVE TO LIST
        new_data.append(df)
        new_processed_files.append(file)

        print(f"[SUCCESS] {file} processed")

    except Exception as e:
        print(f"[ERROR] Failed processing {file}: {e}")

# SAVE FINAL DATA
if new_data:

    final_df = pd.concat(
        new_data,
        ignore_index=True
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # OVERWRITE
    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n[INFO] Dataset saved")
    print(f"[INFO] Shape: {final_df.shape}")

    # Update processed log
    with open(LOG_FILE, "a") as f:
        for file in new_processed_files:
            f.write(file + "\n")

    print("[SUCCESS] Processing completed")

else:
    print("[INFO] No new files found")