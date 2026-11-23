import os
import sys
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import CreditRiskPredictor

app = FastAPI(
    title="CreditIQ Risk Decision Engine API",
    description="Production REST API & Web Dashboard for Credit Default Probability (PD) & Automated Underwriting powered by TensorFlow ANN",
    version="1.0.0"
)

# Global predictor instance (loaded eagerly if available)
try:
    predictor = CreditRiskPredictor()
    print("[INFO] CreditIQ Risk Engine initialized eagerly.")
except Exception as e:
    predictor = None
    print(f"[INFO] Predictor deferred: {e}")

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

@app.on_event("startup")
def load_engine():
    global predictor
    if predictor is None:
        try:
            predictor = CreditRiskPredictor()
            print("[INFO] CreditIQ Risk Engine loaded on startup.")
        except Exception as e:
            print(f"[WARNING] Model not loaded yet: {e}")

class ApplicantRequest(BaseModel):
    loan_amnt: float = Field(..., example=15000.0, description="Requested loan amount ($)")
    term: str = Field(..., example=" 36 months", description="Loan duration (' 36 months' or ' 60 months')")
    int_rate: float = Field(..., example=10.5, description="Offered/market interest rate (%)")
    installment: float = Field(..., example=487.50, description="Calculated monthly installment ($)")
    annual_inc: float = Field(..., example=75000.0, description="Verified or self-reported annual income ($)")
    emp_length: str = Field(..., example="5 years", description="Employment duration (e.g. '10+ years', '< 1 year')")
    home_ownership: str = Field(..., example="MORTGAGE", description="RENT, MORTGAGE, OWN, OTHER")
    verification_status: str = Field(..., example="Verified", description="Verified, Source Verified, Not Verified")
    purpose: str = Field(..., example="debt_consolidation", description="Loan purpose category")
    dti: float = Field(..., example=16.8, description="Debt-to-Income ratio (%)")
    delinq_2yrs: int = Field(0, example=0, description="Delinquencies in the past 2 years")
    earliest_cr_line: str = Field(..., example="May-2010", description="Month-Year of first credit line")
    issue_d: str = Field(..., example="Nov-2018", description="Origination Month-Year")
    inq_last_6mths: int = Field(0, example=1, description="Hard inquiries in past 6 months")
    open_acc: int = Field(..., example=10, description="Number of currently open credit lines")
    pub_rec: int = Field(0, example=0, description="Derogatory public records count")
    revol_bal: float = Field(..., example=9500.0, description="Total credit revolving balance ($)")
    revol_util: str = Field(..., example="32.5%", description="Revolving credit utilization (%)")
    total_acc: int = Field(..., example=22, description="Total credit lines in history")
    fico_range_low: int = Field(..., example=720, description="Lower FICO boundary")
    fico_range_high: int = Field(..., example=724, description="Upper FICO boundary")
    collections_12_mths_ex_med: int = Field(0, example=0, description="Collections in past 12 mos")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine_loaded": predictor is not None
    }

@app.post("/api/v1/score_applicant")
def score_applicant(applicant: ApplicantRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="CreditIQ model is not loaded. Train the model first.")
    
    try:
        data_dict = applicant.dict()
        decision = predictor.predict_single_applicant(data_dict)
        return {
            "success": True,
            "data": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Frontend web dashboard routes
if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    def serve_frontend_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{file_name}")
    def serve_frontend_asset(file_name: str):
        target = os.path.join(FRONTEND_DIR, file_name)
        if os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
