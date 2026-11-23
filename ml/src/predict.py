import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_FILE, PREPROCESSOR_FILE, RISK_TIERS

import tensorflow as tf
from tensorflow import keras
from preprocess import CreditIQPreprocessor, engineer_features

class CreditRiskPredictor:
    def __init__(self, model_path=MODEL_FILE, preprocessor_path=PREPROCESSOR_FILE):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run train.py first.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}. Please run train.py first.")

        print(f"[INFO] Loading CreditIQ Model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print(f"[INFO] Loading Preprocessor from: {preprocessor_path}")
        self.preprocessor = CreditIQPreprocessor.load(preprocessor_path)

    def predict_single_applicant(self, applicant_data: dict) -> dict:
        """
        Scores a single loan applicant against the trained ANN model.
        applicant_data: dict containing candidate fields (e.g., loan_amnt, annual_inc, fico_range_low, etc.)
        """
        df = pd.DataFrame([applicant_data])
        features_df = engineer_features(df)
        X_trans = self.preprocessor.transform(features_df)
        
        prob_default = float(self.model.predict(X_trans, verbose=0)[0][0])

        # Assign CreditIQ Risk Tier
        tier_name = "Subprime (High Risk)"
        action = "Decline"
        badge_color = "red"
        
        for name, tier in RISK_TIERS.items():
            if prob_default <= tier["max_pd"]:
                tier_name = name
                action = tier["action"]
                badge_color = tier["color"]
                break

        # Calculate credit score equivalent (e.g. 300 - 850 scale)
        # Low default probability corresponds to high credit score
        simulated_credit_score = int(850 - (prob_default * 550))
        simulated_credit_score = max(300, min(850, simulated_credit_score))

        # Risk drivers check
        risk_flags = []
        dti = float(applicant_data.get("dti", 0))
        fico = (float(applicant_data.get("fico_range_low", 700)) + float(applicant_data.get("fico_range_high", 700))) / 2.0
        inquiries = int(applicant_data.get("inq_last_6mths", 0))
        revol_util = str(applicant_data.get("revol_util", "0%")).replace("%", "")
        try:
            revol_util_val = float(revol_util)
        except:
            revol_util_val = 0.0

        if fico < 660:
            risk_flags.append(f"Low FICO Score ({fico:.0f})")
        if dti > 35.0:
            risk_flags.append(f"Elevated Debt-to-Income Ratio ({dti:.1f}%)")
        if inquiries >= 3:
            risk_flags.append(f"High Recent Credit Inquiries ({inquiries} in last 6 mos)")
        if revol_util_val > 75.0:
            risk_flags.append(f"High Revolving Credit Utilization ({revol_util_val:.1f}%)")
        if not risk_flags:
            risk_flags.append("Strong credit fundamentals across primary indicators")

        return {
            "probability_of_default": round(prob_default, 4),
            "probability_of_default_pct": f"{prob_default * 100:.2f}%",
            "risk_tier": tier_name,
            "decision": action,
            "color_badge": badge_color,
            "internal_credit_score": simulated_credit_score,
            "key_risk_drivers": risk_flags
        }

    def predict_batch(self, applicants_df: pd.DataFrame) -> pd.DataFrame:
        """Scores a batch of applicants and appends PD, Risk Tier, and Decision."""
        features_df = engineer_features(applicants_df.copy())
        X_trans = self.preprocessor.transform(features_df)
        preds = self.model.predict(X_trans, batch_size=256, verbose=0).flatten()

        results_df = applicants_df.copy()
        results_df["predicted_pd"] = np.round(preds, 4)
        results_df["predicted_pd_pct"] = (results_df["predicted_pd"] * 100).round(2).astype(str) + "%"

        def map_tier(pd_val):
            for name, tier in RISK_TIERS.items():
                if pd_val <= tier["max_pd"]:
                    return name, tier["action"]
            return "Subprime (High Risk)", "Decline"

        tier_actions = [map_tier(p) for p in preds]
        results_df["risk_tier"] = [t[0] for t in tier_actions]
        results_df["recommended_decision"] = [t[1] for t in tier_actions]

        return results_df

if __name__ == "__main__":
    # Example applicant testing
    sample_borrower = {
        "loan_amnt": 15000,
        "term": " 36 months",
        "int_rate": 8.99,
        "installment": 476.92,
        "annual_inc": 85000,
        "emp_length": "6 years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 14.5,
        "delinq_2yrs": 0,
        "earliest_cr_line": "Jan-2008",
        "issue_d": "Mar-2018",
        "inq_last_6mths": 0,
        "open_acc": 12,
        "pub_rec": 0,
        "revol_bal": 12400,
        "revol_util": "28.5%",
        "total_acc": 22,
        "fico_range_low": 740,
        "fico_range_high": 744,
        "collections_12_mths_ex_med": 0
    }
    
    print("Sample Borrower Payload:")
    for k, v in sample_borrower.items():
        print(f"  {k}: {v}")

    try:
        predictor = CreditRiskPredictor()
        result = predictor.predict_single_applicant(sample_borrower)
        print("\nUnderwriting Decision Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Note] Model will run once trained: {e}")
