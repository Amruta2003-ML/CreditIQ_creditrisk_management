import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_RAW_FILE, SAMPLE_RAW_FILE, TARGET_MAP, RAW_FEATURE_COLUMNS

def inspect_dataset(file_path=None, nrows=10000):
    if file_path is None or not os.path.exists(file_path):
        if os.path.exists(DEFAULT_RAW_FILE):
            file_path = DEFAULT_RAW_FILE
        elif os.path.exists(SAMPLE_RAW_FILE):
            file_path = SAMPLE_RAW_FILE
            print(f"[INFO] Using sample dataset: {file_path}")
        else:
            raise FileNotFoundError(f"Neither {DEFAULT_RAW_FILE} nor {SAMPLE_RAW_FILE} was found.")

    print("=" * 60)
    print(f"CreditIQ EDA - Inspecting Dataset: {os.path.basename(file_path)}")
    print("=" * 60)

    # Detect compression
    compression = "gzip" if file_path.endswith(".gz") else None
    
    # Load sample / chunk
    df = pd.read_csv(file_path, compression=compression, nrows=nrows, low_memory=False)
    print(f"Sample shape loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target variable distribution
    if "loan_status" in df.columns:
        print("\n--- Target Variable ('loan_status') Distribution ---")
        status_counts = df["loan_status"].value_counts(dropna=False)
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            mapped = TARGET_MAP.get(status, "EXCLUDE")
            print(f"  {status:<30} Count: {count:<8} ({pct:5.2f}%) -> Target: {mapped}")
        
        resolved_mask = df["loan_status"].isin(TARGET_MAP.keys())
        resolved_df = df[resolved_mask]
        binary_targets = resolved_df["loan_status"].map(TARGET_MAP)
        default_rate = binary_targets.mean() * 100
        print(f"\nResolved Loans count: {len(resolved_df)} / {len(df)} ({len(resolved_df)/len(df)*100:.1f}%)")
        print(f"Baseline Default Rate (1s): {default_rate:.2f}% (Fully Paid: {100-default_rate:.2f}%)")
    else:
        print("\n[WARNING] 'loan_status' column not found!")

    # Check candidate features availability
    print("\n--- Candidate Features Verification (20 Features) ---")
    available_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    print(f"Available: {len(available_cols)} / {len(RAW_FEATURE_COLUMNS)}")
    if missing_cols:
        print(f"[!] Missing candidate columns: {missing_cols}")
    else:
        print("[OK] All candidate features present!")

    # Missing value inspection
    print("\n--- Missing Values in Candidate Features ---")
    sub_df = df[[c for c in available_cols if c in df.columns]]
    missing = sub_df.isnull().sum()
    missing_pct = (missing / len(sub_df)) * 100
    missing_table = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
    print(missing_table[missing_table["Missing Count"] > 0].sort_values("Percentage (%)", ascending=False).to_string())

    # Data types
    print("\n--- Data Types Summary ---")
    print(sub_df.dtypes.value_counts())

    print("=" * 60)
    print("EDA inspection complete.")
    print("=" * 60)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CreditIQ LendingClub Data Inspection Script")
    parser.add_argument("--data-path", type=str, default=None, help="Path to accepted_2007_to_2018Q4.csv.gz or CSV")
    parser.add_argument("--nrows", type=int, default=10000, help="Number of rows to inspect")
    args = parser.parse_args()
    inspect_dataset(file_path=args.data_path, nrows=args.nrows)
