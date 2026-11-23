import os

# Set environment variables for sandbox compatibility
os.environ.setdefault("KERAS_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".keras")))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".matplotlib")))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DEFAULT_RAW_FILE = os.path.join(RAW_DATA_DIR, "accepted_2007_to_2018Q4.csv.gz")
SAMPLE_RAW_FILE = os.path.join(RAW_DATA_DIR, "sample_accepted_loans.csv")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "creditiq_preprocessor.joblib")
MODEL_FILE = os.path.join(MODELS_DIR, "creditiq_ann_model.keras")
METRICS_FILE = os.path.join(MODELS_DIR, "evaluation_metrics.json")

# Target mapping: Resolved loans only for credit default prediction
TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
}

# The 20 candidate LendingClub raw features + issue date for temporal split
RAW_FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "annual_inc",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "issue_d",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "fico_range_low",
    "fico_range_high",
    "collections_12_mths_ex_med",
    "loan_status"
]

NUMERICAL_FEATURES = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "monthly_income",
    "emp_length_years",
    "dti",
    "delinq_2yrs",
    "credit_hist_months",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "fico_score",
    "collections_12_mths_ex_med"
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose"
]

# Risk tier thresholds for CreditIQ application
RISK_TIERS = {
    "Prime (Low Risk)": {"max_pd": 0.10, "action": "Approve", "color": "green"},
    "Near Prime (Moderate Risk)": {"max_pd": 0.25, "action": "Manual Review / Stricter Terms", "color": "yellow"},
    "Subprime (High Risk)": {"max_pd": 1.00, "action": "Decline", "color": "red"}
}
