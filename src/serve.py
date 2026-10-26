"""Serving API — mô phỏng cách model này được tích hợp vào hệ thống CRM/
marketing automation thật (không chỉ dừng ở batch notebook/script).

Chạy: uvicorn src.serve:app --reload --port 8000   (từ project root)
hoặc: python src/serve.py

Test: curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \
  -d '{"recency_days": 30, "frequency": 5, "monetary": 500, "avg_order_value": 100, "n_distinct_products": 10, "tenure_days": 200, "return_rate": 0.1, "rfm_segment": "Loyal"}'
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config  # noqa: E402

FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_order_value", "n_distinct_products", "tenure_days", "return_rate",
]

app = FastAPI(title="Customer Value & Promotion Targeting — Scoring API")

_state: dict = {}


class CustomerFeatures(BaseModel):
    recency_days: float = Field(..., ge=0)
    frequency: int = Field(..., ge=0)
    monetary: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    n_distinct_products: int = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0)
    return_rate: float = Field(..., ge=0, le=1)
    rfm_segment: str = Field(..., description="Champions | Loyal | At-risk | Lost")


class ScoreResponse(BaseModel):
    predicted_repeat_purchase_proba: float
    predicted_future_value: float
    incremental_margin_reference_scenario: float
    final_recommendation: str


@app.on_event("startup")
def load_models() -> None:
    _state["config"] = load_config(PROJECT_ROOT / "config" / "params.yaml")
    _state["clf"] = joblib.load(PROJECT_ROOT / "models" / "gbm_classifier_v1.joblib")
    _state["reg"] = joblib.load(PROJECT_ROOT / "models" / "gbm_regressor_v1.joblib")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "clf" in _state}


@app.post("/score", response_model=ScoreResponse)
def score(customer: CustomerFeatures):
    import pandas as pd

    from importlib import import_module
    decision = import_module("05_decision_scenario")

    config = _state["config"]
    row = pd.DataFrame([customer.model_dump()])
    X = row[FEATURE_COLS]

    proba = float(_state["clf"].predict_proba(X)[:, 1][0])
    value = float(_state["reg"].predict(X)[0])

    dec = config["decision"]
    ref = next(s for s in dec["discount_response_scenarios"] if s["name"] == dec["reference_scenario"])
    eff_rr = decision.effective_response_rate(row, ref["response_rate"], dec["segment_response_multiplier"])
    margin = decision.incremental_margin(
        pd.Series([value]), eff_rr, ref["discount_pct"], dec["gross_margin_pct"]
    ).iloc[0]
    recommendation = decision.assign_recommendation(row.assign(predicted_future_value_gbm=value), pd.Series([margin])).iloc[0]

    return ScoreResponse(
        predicted_repeat_purchase_proba=proba,
        predicted_future_value=value,
        incremental_margin_reference_scenario=float(margin),
        final_recommendation=recommendation,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
