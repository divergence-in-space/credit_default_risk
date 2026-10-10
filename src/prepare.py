"""Clean the raw Lending Club file into a modelling-ready dataset.

Put the Kaggle file (accepted_2007_to_2018Q4.csv) in data/ then run:
    python src/prepare.py
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/accepted_2007_to_2018Q4.csv")
OUT = Path("data/loans_clean.csv")

# Only columns known AT APPLICATION TIME. Anything that is only known after the
# loan is issued (recoveries, total_pymnt, last_pymnt_d, out_prncp ...) leaks the
# answer and must NOT be used.
FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "application_type",
]
KEEP = FEATURES + ["loan_status", "issue_d"]


def main() -> None:
    df = pd.read_csv(RAW, usecols=KEEP, low_memory=False)
    print("Raw rows:", len(df))

    # Keep only loans with a final outcome
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    # Parse fields
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    df["emp_length"] = (df["emp_length"].str.extract(r"(\d+)")[0].astype(float))
    df.loc[df["emp_length"].isna(), "emp_length"] = pd.NA
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["earliest_cr_line", "fico_range_low", "fico_range_high",
                          "loan_status"])

    print("Modelling rows:", len(df))
    print("Default rate:", round(df["default"].mean(), 4))
    df.to_csv(OUT, index=False)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
