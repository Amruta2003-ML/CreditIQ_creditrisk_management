import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_lendingclub_sample(n_samples=10000, output_path=None):
    """
    Generates a realistic synthetic sample following LendingClub's schema and distributions.
    This allows end-to-end testing of EDA, preprocessing, ANN training, and inference.
    """
    np.random.seed(42)
    random.seed(42)

    # 1. Loan details
    loan_amnt = np.random.choice([5000, 10000, 15000, 20000, 25000, 30000, 35000], size=n_samples, p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.07, 0.03])
    term = np.random.choice([" 36 months", " 60 months"], size=n_samples, p=[0.72, 0.28])
    int_rate = np.round(np.random.uniform(5.32, 28.99, size=n_samples), 2)
    
    # Calculate approximate monthly installment
    months = np.where(term == " 36 months", 36, 60)
    r = (int_rate / 100) / 12
    installment = np.round(loan_amnt * (r * (1 + r)**months) / ((1 + r)**months - 1), 2)

    # 2. Borrower details
    annual_inc = np.round(np.random.lognormal(mean=11.1, sigma=0.5, size=n_samples), 2)
    annual_inc = np.clip(annual_inc, 15000, 300000)

    emp_lengths = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", 
                   "6 years", "7 years", "8 years", "9 years", "10+ years", np.nan]
    emp_length = np.random.choice(emp_lengths, size=n_samples, p=[0.08, 0.07, 0.09, 0.08, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.35, 0.05])

    home_ownership = np.random.choice(["MORTGAGE", "RENT", "OWN", "ANY", "OTHER"], size=n_samples, p=[0.49, 0.40, 0.10, 0.005, 0.005])
    verification_status = np.random.choice(["Source Verified", "Verified", "Not Verified"], size=n_samples, p=[0.38, 0.32, 0.30])
    
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", 
                "small_business", "car", "medical", "moving", "vacation", "house", "wedding"]
    purpose = np.random.choice(purposes, size=n_samples, p=[0.58, 0.23, 0.06, 0.05, 0.02, 0.015, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002])

    dti = np.round(np.random.normal(loc=18.5, scale=8.2, size=n_samples), 2)
    dti = np.clip(dti, 0.0, 45.0)

    # 3. Credit profile
    delinq_2yrs = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.80, 0.13, 0.04, 0.02, 0.01])
    inq_last_6mths = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.55, 0.25, 0.12, 0.05, 0.02, 0.01])
    open_acc = np.random.poisson(lam=11, size=n_samples)
    open_acc = np.clip(open_acc, 2, 40)
    total_acc = open_acc + np.random.poisson(lam=13, size=n_samples)
    pub_rec = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.12, 0.02, 0.01])
    revol_bal = np.round(np.random.lognormal(mean=9.3, sigma=0.8, size=n_samples), 2)
    revol_util = np.round(np.random.uniform(0.0, 99.8, size=n_samples), 1)
    collections_12_mths_ex_med = np.random.choice([0, 1, 2], size=n_samples, p=[0.97, 0.025, 0.005])

    # FICO score ranges
    fico_low = np.random.choice(range(660, 845, 5), size=n_samples)
    fico_high = fico_low + 4

    # Temporal dates
    issue_years = np.random.choice([2014, 2015, 2016, 2017, 2018], size=n_samples, p=[0.15, 0.25, 0.25, 0.20, 0.15])
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    issue_d = [f"{random.choice(months_abbr)}-{y}" for y in issue_years]
    
    # Earliest credit line 3 to 25 years before issue date
    earliest_cr_line = []
    for y in issue_years:
        credit_years_back = random.randint(3, 25)
        earliest_cr_line.append(f"{random.choice(months_abbr)}-{y - credit_years_back}")

    # Probability of default logic (correlated with FICO, DTI, int_rate, inquiries)
    # Higher int_rate, higher DTI, lower FICO -> higher default risk
    risk_score = (
        (int_rate - 13.0) * 0.10 +
        (dti - 18.0) * 0.05 +
        (700 - fico_low) * 0.015 +
        delinq_2yrs * 0.3 +
        inq_last_6mths * 0.2 +
        (loan_amnt / (annual_inc + 1e-5)) * 1.5
    )
    prob_default = 1 / (1 + np.exp(-risk_score))
    # Calibrate default rate around ~20% (historical LendingClub rate)
    defaults = (prob_default > np.percentile(prob_default, 80)).astype(int)

    loan_status = []
    for d in defaults:
        if d == 1:
            loan_status.append(np.random.choice(["Charged Off", "Default"], p=[0.95, 0.05]))
        else:
            loan_status.append("Fully Paid")

    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "earliest_cr_line": earliest_cr_line,
        "issue_d": issue_d,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": [f"{u}%" if random.random() > 0.05 else f"{u}" for u in revol_util],
        "total_acc": total_acc,
        "fico_range_low": fico_low,
        "fico_range_high": fico_high,
        "collections_12_mths_ex_med": collections_12_mths_ex_med,
        "loan_status": loan_status
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic sample saved to {output_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    return df

if __name__ == "__main__":
    from config import SAMPLE_RAW_FILE
    generate_synthetic_lendingclub_sample(n_samples=10000, output_path=SAMPLE_RAW_FILE)
