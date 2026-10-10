"""Calibration, business-cost threshold and SHAP summary plots.

    python src/evaluate.py                  # default: missed default costs 4x a rejected good loan
    python src/evaluate.py --cost-ratio 6
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

DATA = Path("data/loans_clean.csv")
MODELS = Path("models")
OUT = Path("outputs")
VAL_START = pd.Timestamp("2015-01-01")   # must match train.py
TEST_START = pd.Timestamp("2016-01-01")


def load_splits():
    meta = json.load(open(MODELS / "meta.json"))
    df = pd.read_csv(DATA, parse_dates=["issue_d"])

    def prep(part):
        y = part["default"].to_numpy()
        X = part.drop(columns=["default", "issue_d"]).reindex(columns=meta["features"])
        for col, levels in meta["categories"].items():
            X[col] = pd.Categorical(X[col].astype(str), categories=levels)
        return X, y

    val = df[(df["issue_d"] >= VAL_START) & (df["issue_d"] < TEST_START)]
    test = df[df["issue_d"] >= TEST_START]
    return prep(val), prep(test), meta


def cost_at(y, p, t, cost_ratio):
    decline = p >= t
    missed = int(((~decline) & (y == 1)).sum())
    lost_good = int((decline & (y == 0)).sum())
    return {"threshold": float(t), "cost": cost_ratio * missed + lost_good,
            "missed_defaults": missed, "declined_good": lost_good,
            "decline_rate": float(decline.mean()),
            "cost_if_approve_everyone": float(cost_ratio * (y == 1).sum())}


def calibration(y, p):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label="LightGBM")
    plt.xlabel("Predicted default probability")
    plt.ylabel("Actual default rate")
    plt.title("Calibration (test set, 2016+ loans)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=150)
    plt.close()
    return {"brier": float(brier_score_loss(y, p)),
            "mean_predicted": float(p.mean()),
            "actual_default_rate": float(y.mean())}


def threshold_search(y, p, cost_ratio):
    """Cost = cost_ratio * (approved loans that defaulted) + 1 * (declined good loans)."""
    grid = np.round(np.arange(0.05, 0.65, 0.01), 2)
    rows = []
    for t in grid:
        decline = p >= t
        missed_default = int(((~decline) & (y == 1)).sum())
        lost_good = int((decline & (y == 0)).sum())
        rows.append({"threshold": float(t),
                     "cost": cost_ratio * missed_default + lost_good,
                     "missed_defaults": missed_default,
                     "declined_good": lost_good,
                     "decline_rate": float(decline.mean())})
    res = pd.DataFrame(rows)
    best = res.loc[res["cost"].idxmin()]
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(res["threshold"], res["cost"])
    plt.axvline(best["threshold"], color="red", ls="--",
                label=f"Best threshold = {best['threshold']:.2f}")
    plt.xlabel("Decline if predicted default probability >=")
    plt.ylabel("Total cost (units)")
    plt.title(f"Threshold vs cost on validation set (missed default = {cost_ratio}x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_cost.png", dpi=150)
    plt.close()
    res.to_csv(OUT / "threshold_sweep.csv", index=False)
    return best.to_dict()


def shap_plots(model, X, categorical):
    sample = X.sample(5000, random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    plot_X = sample.copy()
    for c in categorical:  # numeric codes so the colour scale works
        plot_X[c] = plot_X[c].cat.codes
    for kind, name in [("bar", "shap_importance.png"), (None, "shap_beeswarm.png")]:
        plt.figure()
        shap.summary_plot(sv, plot_X, plot_type=kind, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
        plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return imp.head(10).round(4).to_dict()


def main(cost_ratio: float) -> None:
    OUT.mkdir(exist_ok=True)
    model = joblib.load(MODELS / "lgbm.joblib")
    (X_val, y_val), (X, y), meta = load_splits()
    p_val = model.predict_proba(X_val)[:, 1]
    p = model.predict_proba(X)[:, 1]
    print(f"Validation rows (2015): {len(y_val)}   Test rows (2016+): {len(y)}")
    print(f"Test ROC-AUC: {roc_auc_score(y, p):.4f}")

    cal = calibration(y, p)
    print("Calibration (test):", json.dumps(cal, indent=2))

    # Threshold is chosen on the VALIDATION period, then reported on the untouched TEST set
    best_val = threshold_search(y_val, p_val, cost_ratio)
    t = best_val["threshold"]
    print("Best threshold on validation:", json.dumps(best_val, indent=2))
    test_result = cost_at(y, p, t, cost_ratio)
    print("Same threshold applied to TEST:", json.dumps(test_result, indent=2))
    with open(MODELS / "threshold.json", "w") as f:
        json.dump({"threshold": t, "cost_ratio": cost_ratio}, f)

    top = shap_plots(model, X, list(meta["categories"].keys()))
    print("Top features by mean |SHAP|:", json.dumps(top, indent=2))

    with open(OUT / "evaluation.json", "w") as f:
        json.dump({"calibration": cal, "threshold_chosen_on_validation": best_val,
                   "test_result_at_that_threshold": test_result,
                   "top_shap_features": top}, f, indent=2)
    print("Saved charts and evaluation.json to outputs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-ratio", type=float, default=4.0)
    main(ap.parse_args().cost_ratio)
