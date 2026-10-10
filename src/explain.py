"""Model loading, prediction, SHAP explanation and LLM (Groq) wording."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

MODELS = Path("models")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class RiskModel:
    def __init__(self) -> None:
        self.model = joblib.load(MODELS / "lgbm.joblib")
        with open(MODELS / "meta.json") as f:
            self.meta = json.load(f)
        self.explainer = shap.TreeExplainer(self.model)

    def _frame(self, payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload]).reindex(columns=self.meta["features"])
        for col, levels in self.meta["categories"].items():
            df[col] = pd.Categorical(df[col].astype(str), categories=levels)
        return df

    def predict(self, payload: dict, top_k: int = 5) -> dict:
        df = self._frame(payload)
        prob = float(self.model.predict_proba(df)[:, 1][0])
        sv = self.explainer.shap_values(df)
        if isinstance(sv, list):  # older shap versions return one array per class
            sv = sv[1]
        sv = np.asarray(sv)[0]
        order = np.argsort(-np.abs(sv))[:top_k]
        factors = [{
            "feature": self.meta["features"][i],
            "value": None if pd.isna(df.iloc[0, i]) else str(df.iloc[0, i]),
            "shap": round(float(sv[i]), 4),
            "direction": "raises risk" if sv[i] > 0 else "lowers risk",
        } for i in order]
        return {"probability": prob, "factors": factors}


def explain_in_words(prob: float, decision: str, factors: list[dict]) -> str:
    """Plain-English explanation. The LLM may only use the factors it is given."""
    facts = "\n".join(
        f"- {f['feature']} = {f['value']} ({f['direction']})" for f in factors)
    fallback = (f"Estimated default risk is {prob:.0%} ({decision}). Main drivers:\n"
                + facts)
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return fallback
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content":
                    "You explain a credit risk model's decision to a loan officer in "
                    "3-4 plain sentences. Use ONLY the factors listed. Do not invent "
                    "reasons, numbers or advice. Do not mention SHAP."},
                {"role": "user", "content":
                    f"Predicted default probability: {prob:.1%}\nDecision: {decision}\n"
                    f"Top factors:\n{facts}"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback
