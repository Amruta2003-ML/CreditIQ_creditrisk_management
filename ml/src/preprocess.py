import os
import sys
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_RAW_FILE,
    SAMPLE_RAW_FILE,
    PREPROCESSOR_FILE,
    PROCESSED_DATA_DIR,
    TARGET_MAP,
    RAW_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

def parse_emp_length(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if "<" in val_str:
        return 0.0
    elif "10+" in val_str:
        return 10.0
    match = re.search(r"\d+", val_str)
    return float(match.group()) if match else np.nan

def parse_term(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"\d+", str(val))
    return float(match.group()) if match else np.nan

def parse_revol_util(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("%", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def engineer_features(df):
    """Applies domain-specific transformations to raw LendingClub columns."""
    processed = pd.DataFrame(index=df.index)

    # 1. Term in months
    processed["term_months"] = df["term"].apply(parse_term)

    # 2. Loan amount and interest
    processed["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    processed["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    processed["installment"] = pd.to_numeric(df["installment"], errors="coerce")

    # 3. Income and monthly income
    annual_inc = pd.to_numeric(df["annual_inc"], errors="coerce")
    processed["annual_inc"] = annual_inc
    processed["monthly_income"] = annual_inc / 12.0

    # 4. Employment duration in years
    processed["emp_length_years"] = df["emp_length"].apply(parse_emp_length)

    # 5. DTI and previous delinquencies
    processed["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    processed["delinq_2yrs"] = pd.to_numeric(df["delinq_2yrs"], errors="coerce")

    # 6. Credit history length from issue_d and earliest_cr_line
    issue_date = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    earliest_cr = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    processed["credit_hist_months"] = ((issue_date - earliest_cr).dt.days / 30.4375).round()
    # Also save issue_year for time-based train/test splitting
    processed["issue_year"] = issue_date.dt.year

    # 7. Inquiries and accounts
    processed["inq_last_6mths"] = pd.to_numeric(df["inq_last_6mths"], errors="coerce")
    processed["open_acc"] = pd.to_numeric(df["open_acc"], errors="coerce")
    processed["pub_rec"] = pd.to_numeric(df["pub_rec"], errors="coerce")
    processed["revol_bal"] = pd.to_numeric(df["revol_bal"], errors="coerce")
    processed["revol_util"] = df["revol_util"].apply(parse_revol_util)
    processed["total_acc"] = pd.to_numeric(df["total_acc"], errors="coerce")

    # 8. FICO score (average of low and high)
    fico_low = pd.to_numeric(df["fico_range_low"], errors="coerce")
    fico_high = pd.to_numeric(df["fico_range_high"], errors="coerce")
    processed["fico_score"] = (fico_low + fico_high) / 2.0

    # 9. Recent collections
    processed["collections_12_mths_ex_med"] = pd.to_numeric(df["collections_12_mths_ex_med"], errors="coerce")

    # 10. Categoricals
    processed["home_ownership"] = df["home_ownership"].astype(str).str.strip().str.upper()
    processed["verification_status"] = df["verification_status"].astype(str).str.strip()
    processed["purpose"] = df["purpose"].astype(str).str.strip().str.lower()

    # Target variable
    if "loan_status" in df.columns:
        processed["loan_status"] = df["loan_status"]

    return processed

class CreditIQPreprocessor:
    def __init__(self):
        self.num_cols = NUMERICAL_FEATURES
        self.cat_cols = CATEGORICAL_FEATURES
        self.pipeline = None
        self.feature_names_out = None

    def build_pipeline(self):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.num_cols),
                ("cat", cat_pipeline, self.cat_cols)
            ]
        )
        return self.pipeline

    def fit(self, X_train):
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(X_train[self.num_cols + self.cat_cols])
        # Retrieve feature names
        num_names = self.num_cols
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(self.cat_cols))
        self.feature_names_out = num_names + cat_names
        return self

    def transform(self, X):
        transformed = self.pipeline.transform(X[self.num_cols + self.cat_cols])
        return transformed

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, filepath=PREPROCESSOR_FILE):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_names": self.feature_names_out,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols
        }, filepath)
        print(f"[OK] Preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath=PREPROCESSOR_FILE):
        obj = cls()
        data = joblib.load(filepath)
        obj.pipeline = data["pipeline"]
        obj.feature_names_out = data["feature_names"]
        obj.num_cols = data["num_cols"]
        obj.cat_cols = data["cat_cols"]
        return obj

def load_and_preprocess_data(file_path=None, nrows=None):
    if file_path is None or not os.path.exists(file_path):
        if os.path.exists(DEFAULT_RAW_FILE):
            file_path = DEFAULT_RAW_FILE
        elif os.path.exists(SAMPLE_RAW_FILE):
            file_path = SAMPLE_RAW_FILE
        else:
            raise FileNotFoundError("No raw or sample dataset found.")

    print(f"\n[INFO] Loading dataset from: {file_path}")
    compression = "gzip" if file_path.endswith(".gz") else None
    
    # Read required columns
    usecols = [c for c in RAW_FEATURE_COLUMNS]
    try:
        raw_df = pd.read_csv(file_path, compression=compression, nrows=nrows, usecols=usecols, low_memory=False)
    except ValueError:
        raw_df = pd.read_csv(file_path, compression=compression, nrows=nrows, low_memory=False)

    print(f"[INFO] Raw records loaded: {len(raw_df)}")

    # 1. Filter resolved loans only (Lifetime default classification)
    resolved_mask = raw_df["loan_status"].isin(TARGET_MAP.keys())
    filtered_df = raw_df[resolved_mask].copy()
    print(f"[INFO] Retained resolved loans: {len(filtered_df)} / {len(raw_df)} ({len(filtered_df)/len(raw_df)*100:.1f}%)")

    # 2. Map target
    y_all = filtered_df["loan_status"].map(TARGET_MAP).values

    # 3. Engineer features
    print("[INFO] Engineering domain features...")
    features_df = engineer_features(filtered_df)

    # 4. Temporal Out-of-Time Splitting (prevents data leakage across macro cycles)
    years = features_df["issue_year"].dropna().unique()
    years = sorted(years)
    print(f"[INFO] Available origination years: {years}")

    if len(years) >= 3:
        # e.g., Train <= 2016, Val = 2017, Test = 2018
        train_years = years[:-2]
        val_years = [years[-2]]
        test_years = [years[-1]]
        train_mask = features_df["issue_year"].isin(train_years)
        val_mask = features_df["issue_year"].isin(val_years)
        test_mask = features_df["issue_year"].isin(test_years)
    else:
        # Fallback to 70/15/15 chronological split if years are missing/insufficient
        n = len(features_df)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        n_train = int(n * 0.70)
        n_val = int(n * 0.85)
        train_mask[:n_train] = True
        val_mask[n_train:n_val] = True
        test_mask[n_val:] = True

    X_train_raw = features_df[train_mask]
    y_train = y_all[train_mask]
    X_val_raw = features_df[val_mask]
    y_val = y_all[val_mask]
    X_test_raw = features_df[test_mask]
    y_test = y_all[test_mask]

    print(f"[INFO] Splits -> Train: {len(X_train_raw)} (Default rate: {y_train.mean():.2%}), "
          f"Val: {len(X_val_raw)} (Default rate: {y_val.mean():.2%}), "
          f"Test: {len(X_test_raw)} (Default rate: {y_test.mean():.2%})")

    # 5. Fit preprocessor on Train ONLY
    preprocessor = CreditIQPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"[INFO] Transformed features dimension: {X_train.shape[1]}")
    preprocessor.save(PREPROCESSOR_FILE)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names_out
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    data = load_and_preprocess_data(file_path=args.data_path, nrows=args.nrows)
    print("[SUCCESS] Data preprocessing pipeline verified successfully!")
