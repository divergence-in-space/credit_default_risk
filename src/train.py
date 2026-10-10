"""Train LightGBM default-risk model with a time-based split.

    python src/train.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/loans_clean.csv")
MODELS = Path("models")
CATEGORICAL = ["grade", "home_ownership", "verification_status",
               "purpose", "application_type"]


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["issue_d"])
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    # Time-based split: train on older loans, test on the newest (mimics real use)
    # 2015 is held out as a validation period (used only by evaluate.py to pick the
    # approve/decline threshold). Test = 2016 onwards.
    val_start = pd.Timestamp("2015-01-01")
    test_start = pd.Timestamp("2016-01-01")
    train = df[df["issue_d"] < val_start]
    test = df[df["issue_d"] >= test_start]
    drop = ["default", "issue_d"]
    X_train, y_train = train.drop(columns=drop), train["default"]
    X_test, y_test = test.drop(columns=drop), test["default"]
    print(f"Train: {len(train)}  Test: {len(test)}")

    # Baseline: logistic regression (numeric features only, median-filled)
    num = X_train.select_dtypes("number").columns
    med = X_train[num].median()
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    base.fit(X_train[num].fillna(med), y_train)
    base_p = base.predict_proba(X_test[num].fillna(med))[:, 1]

    # LightGBM
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]

    metrics = {
        "baseline_logreg": {"roc_auc": roc_auc_score(y_test, base_p),
                            "pr_auc": average_precision_score(y_test, base_p)},
        "lightgbm": {"roc_auc": roc_auc_score(y_test, p),
                     "pr_auc": average_precision_score(y_test, p)},
    }
    print(json.dumps(metrics, indent=2))
    joblib.dump(model, MODELS / "lgbm.joblib")
    meta = {"features": list(X_train.columns),
            "categories": {c: [str(v) for v in X_train[c].cat.categories]
                           for c in CATEGORICAL}}
    with open(MODELS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(MODELS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
