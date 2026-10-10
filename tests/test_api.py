"""API and explanation tests.  Run from the project root:  pytest -v"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import explain
from src.api import app

needs_model = pytest.mark.skipif(
    not Path("models/lgbm.joblib").exists(), reason="train the model first")

GOOD = {
    "loan_amnt": 15000, "term": 36, "int_rate": 13.0, "installment": 505.0,
    "grade": "C", "emp_length": 5, "home_ownership": "MORTGAGE",
    "annual_inc": 70000, "verification_status": "Verified",
    "purpose": "debt_consolidation", "dti": 18.0, "delinq_2yrs": 0,
    "inq_last_6mths": 1, "open_acc": 10, "pub_rec": 0, "revol_bal": 12000,
    "revol_util": 50.0, "total_acc": 22, "mort_acc": 1,
    "pub_rec_bankruptcies": 0, "application_type": "Individual",
    "credit_history_years": 15.0, "fico": 690,
}
SAFE = {**GOOD, "fico": 780, "grade": "A", "int_rate": 7.0, "dti": 8.0,
        "annual_inc": 120000, "installment": 463.0}
RISKY = {**GOOD, "fico": 600, "grade": "E", "int_rate": 25.0, "dti": 35.0,
         "annual_inc": 30000, "term": 60, "installment": 590.0}

FACTORS = [
    {"feature": "int_rate", "value": "20.0", "shap": 0.3, "direction": "raises risk"},
    {"feature": "fico", "value": "780", "shap": -0.2, "direction": "lowers risk"},
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup, so the model is loaded
        yield c


# ---------- API behaviour ----------
@needs_model
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@needs_model
def test_valid_prediction(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"approve", "decline"}
    assert len(body["top_factors"]) == 5
    assert isinstance(body["explanation"], str) and body["explanation"]


@needs_model
@pytest.mark.parametrize("bad", [
    {"fico": 1000}, {"annual_inc": -5}, {"grade": "Z"}, {"term": 48},
    {"dti": 500}, {"loan_amnt": 0},
])
def test_bad_input_rejected(client, bad):
    r = client.post("/predict", json={**GOOD, **bad})
    assert r.status_code == 422


@needs_model
def test_missing_field_rejected(client):
    incomplete = {k: v for k, v in GOOD.items() if k != "fico"}
    assert client.post("/predict", json=incomplete).status_code == 422


@needs_model
def test_risky_applicant_scores_higher_than_safe(client):
    safe = client.post("/predict", json=SAFE).json()["default_probability"]
    risky = client.post("/predict", json=RISKY).json()["default_probability"]
    assert risky > safe


@needs_model
def test_works_without_groq_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert "Estimated default risk" in r.json()["explanation"]


# ---------- LLM explanation layer ----------
def _fake_groq(captured, fail=False):
    class FakeCompletions:
        def create(self, **kwargs):
            if fail:
                raise RuntimeError("simulated Groq outage")
            captured.update(kwargs)

            class Msg:
                content = "ok"

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]
            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = FakeChat()
    return FakeGroq


def test_llm_prompt_only_contains_given_factors(monkeypatch):
    captured = {}
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr("groq.Groq", _fake_groq(captured))
    out = explain.explain_in_words(0.31, "decline", FACTORS)
    system_msg, user_msg = (m["content"] for m in captured["messages"])
    assert out == "ok"
    assert "int_rate = 20.0 (raises risk)" in user_msg
    assert "fico = 780 (lowers risk)" in user_msg
    assert user_msg.count("\n- ") == len(FACTORS)  # nothing extra slipped in
    assert "ONLY" in system_msg
    assert captured["temperature"] == 0


def test_llm_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setattr("groq.Groq", _fake_groq({}, fail=True))
    out = explain.explain_in_words(0.31, "decline", FACTORS)
    assert "Estimated default risk" in out


@pytest.mark.skipif(not __import__("os").getenv("GROQ_API_KEY"),
                    reason="needs a real GROQ_API_KEY")
def test_live_llm_does_not_name_unlisted_features():
    out = explain.explain_in_words(0.31, "decline", FACTORS).lower()
    for unlisted in ["revol_bal", "open_acc", "total_acc", "mort_acc",
                     "delinq_2yrs", "shap"]:
        assert unlisted not in out
