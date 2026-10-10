"""FastAPI service.   uvicorn src.api:app --reload"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.explain import RiskModel, explain_in_words

def _load_threshold() -> float:
    path = Path("models/threshold.json")
    if path.exists():
        return float(json.loads(path.read_text())["threshold"])
    return float(os.getenv("DEFAULT_THRESHOLD", "0.25"))


THRESHOLD = _load_threshold()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["risk"] = RiskModel()  # load once at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Credit Default Risk API", lifespan=lifespan)


class LoanApplication(BaseModel):
    loan_amnt: float = Field(gt=0, le=100_000)
    term: Literal[36, 60]
    int_rate: float = Field(gt=0, le=40)
    installment: float = Field(gt=0)
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    emp_length: Optional[float] = Field(default=None, ge=0, le=10)
    home_ownership: str
    annual_inc: float = Field(gt=0)
    verification_status: str
    purpose: str
    dti: float = Field(ge=0, le=100)
    delinq_2yrs: float = Field(default=0, ge=0)
    inq_last_6mths: float = Field(default=0, ge=0)
    open_acc: float = Field(ge=0)
    pub_rec: float = Field(default=0, ge=0)
    revol_bal: float = Field(ge=0)
    revol_util: Optional[float] = Field(default=None, ge=0)
    total_acc: float = Field(ge=0)
    mort_acc: Optional[float] = Field(default=None, ge=0)
    pub_rec_bankruptcies: Optional[float] = Field(default=None, ge=0)
    application_type: str = "Individual"
    credit_history_years: float = Field(ge=0)
    fico: float = Field(ge=300, le=850)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/predict")
def predict(app_in: LoanApplication):
    result = state["risk"].predict(app_in.model_dump())
    decision = "decline" if result["probability"] >= THRESHOLD else "approve"
    return {
        "default_probability": round(result["probability"], 4),
        "decision": decision,
        "top_factors": result["factors"],
        "explanation": explain_in_words(result["probability"], decision,
                                        result["factors"]),
    }
