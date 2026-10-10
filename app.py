"""Streamlit UI that calls the FastAPI service.

Terminal 1:  uvicorn src.api:app --reload
Terminal 2:  streamlit run app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Default Risk", page_icon="💳", layout="wide")
st.title("💳 Explainable Credit Default Risk")
st.caption("Enter a loan application. The model estimates default risk, shows which "
           "factors drove it, and explains the decision in plain English.")


def monthly_payment(amount: float, rate_pct: float, months: int) -> float:
    r = rate_pct / 100 / 12
    return amount * r / (1 - (1 + r) ** -months) if r else amount / months


left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Loan")
    c1, c2 = st.columns(2)
    loan_amnt = c1.number_input("Loan amount ($)", 500, 100_000, 15_000, step=500)
    term = c2.selectbox("Term (months)", [36, 60])
    int_rate = c1.number_input("Interest rate (%)", 5.0, 40.0, 13.0, step=0.1)
    grade = c2.selectbox("Grade", list("ABCDEFG"), index=2)
    purpose = st.selectbox("Purpose", [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "car", "medical", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational"])

    st.subheader("Borrower")
    c1, c2 = st.columns(2)
    annual_inc = c1.number_input("Annual income ($)", 1_000, 5_000_000, 70_000, step=1_000)
    emp_length = c2.number_input("Employment (years)", 0, 10, 5)
    home_ownership = c1.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    verification_status = c2.selectbox(
        "Income verification", ["Verified", "Source Verified", "Not Verified"])
    application_type = c1.selectbox("Application type", ["Individual", "Joint App"])
    dti = c2.number_input("Debt-to-income (%)", 0.0, 100.0, 18.0, step=0.5)

    st.subheader("Credit history")
    c1, c2 = st.columns(2)
    fico = c1.number_input("FICO score", 300, 850, 690)
    credit_history_years = c2.number_input("Credit history (years)", 0.0, 80.0, 15.0)
    open_acc = c1.number_input("Open accounts", 0, 80, 10)
    total_acc = c2.number_input("Total accounts", 0, 150, 22)
    revol_bal = c1.number_input("Revolving balance ($)", 0, 2_000_000, 12_000, step=500)
    revol_util = c2.number_input("Revolving utilisation (%)", 0.0, 200.0, 50.0)
    delinq_2yrs = c1.number_input("Delinquencies (2 yrs)", 0, 30, 0)
    inq_last_6mths = c2.number_input("Credit inquiries (6 mths)", 0, 30, 1)
    pub_rec = c1.number_input("Public records", 0, 20, 0)
    pub_rec_bankruptcies = c2.number_input("Bankruptcies", 0, 10, 0)
    mort_acc = c1.number_input("Mortgage accounts", 0, 30, 1)

    submit = st.button("Assess risk", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    if not submit:
        st.info("Fill in the application and press **Assess risk**.")
    else:
        payload = {
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
            "installment": round(monthly_payment(loan_amnt, int_rate, term), 2),
            "grade": grade, "emp_length": emp_length,
            "home_ownership": home_ownership, "annual_inc": annual_inc,
            "verification_status": verification_status, "purpose": purpose,
            "dti": dti, "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
            "revol_util": revol_util, "total_acc": total_acc, "mort_acc": mort_acc,
            "pub_rec_bankruptcies": pub_rec_bankruptcies,
            "application_type": application_type,
            "credit_history_years": credit_history_years, "fico": fico,
        }
        try:
            with st.spinner("Scoring..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}. Is `uvicorn src.api:app` running?")
            st.stop()
        except requests.exceptions.HTTPError:
            st.error(f"The API rejected the input: {resp.text}")
            st.stop()

        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{out['default_probability']:.1%}")
        m2.metric("Decision", out["decision"].upper())
        if out["decision"] == "decline":
            st.error("Declined: risk is above the lender's threshold.")
        else:
            st.success("Approved: risk is below the lender's threshold.")

        st.markdown("#### Explanation")
        st.write(out["explanation"])

        st.markdown("#### What drove this prediction")
        df = pd.DataFrame(out["top_factors"])
        st.bar_chart(df.set_index("feature")["shap"])
        st.caption("Bars to the right raise default risk; bars to the left lower it "
                   "(SHAP values, in log-odds).")
        st.dataframe(df[["feature", "value", "direction", "shap"]],
                     hide_index=True, use_container_width=True)
