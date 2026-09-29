"""EnviroSense inference API."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DECIMALS = {"N": 0, "P": 0, "K": 0, "temperature": 1, "humidity": 0, "ph": 1, "rainfall": 0}
CLIMATE = {"temperature", "humidity", "rainfall"}  # can't be "adjusted"; the farmer can only choose around them
CLIMATE_TIPS = {
    ("temperature", "low"): "Shift the planting window later or choose a cooler-season crop.",
    ("temperature", "high"): "Shift the planting window, add shade, or choose a heat-tolerant crop.",
    ("humidity", "low"): "Irrigate or mulch to hold moisture, or choose a drier-climate crop.",
    ("humidity", "high"): "Improve airflow and watch for fungal disease, or choose a humid-climate crop.",
    ("rainfall", "low"): "Plan for irrigation, or choose a drought-tolerant crop.",
    ("rainfall", "high"): "Make sure the field drains well, or choose a wetter-climate crop.",
}
UNITS = {"N": "kg/ha", "P": "kg/ha", "K": "kg/ha", "temperature": "°C", "humidity": "%", "ph": "", "rainfall": "mm"}
LABELS = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium", "temperature": "Temperature",
          "humidity": "Humidity", "ph": "Soil pH", "rainfall": "Rainfall"}

rf = joblib.load(MODELS / "random_forest.joblib")
lr = joblib.load(MODELS / "logistic_regression.joblib")
METRICS = json.loads((MODELS / "metrics.json").read_text())
PROFILES: dict = json.loads((MODELS / "crop_profiles.json").read_text())
CROPS = sorted(PROFILES)

app = FastAPI(title="EnviroSense API", version="2.0.0",
              description="Crop recommendation and field assessment from soil and climate readings.")
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


class Reading(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen, kg/ha")
    P: float = Field(..., ge=0, le=200, description="Phosphorus, kg/ha")
    K: float = Field(..., ge=0, le=250, description="Potassium, kg/ha")
    temperature: float = Field(..., ge=-10, le=55, description="°C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity, %")
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0, le=400, description="mm")
    target_crop: str | None = Field(None, description="Assess the field for this crop instead of the top recommendation")

    model_config = {"json_schema_extra": {"example": {
        "N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82, "ph": 6.5, "rainfall": 203}}}


class FeatureCheck(BaseModel):
    feature: str
    label: str
    value: float
    low: float
    high: float
    unit: str
    status: Literal["low", "ok", "high"]


class Prediction(BaseModel):
    recommended_crop: str
    confidence: float
    top_crops: list[dict]
    baseline: dict
    assessment: dict
    model_version: str


def assess(values: dict, crop: str) -> dict:
    prof = PROFILES[crop]
    checks, actions = [], []
    for f in FEATURES:
        lo, hi, v = prof[f]["low"], prof[f]["high"], values[f]
        status = "low" if v < lo else "high" if v > hi else "ok"
        checks.append(FeatureCheck(feature=f, label=LABELS[f], value=v, low=lo, high=hi,
                                   unit=UNITS[f], status=status).model_dump())
        if status != "ok":
            d = DECIMALS[f]
            u = "" if not UNITS[f] else UNITS[f] if UNITS[f] in ("%", "°C") else f" {UNITS[f]}"
            rng = f"{lo:.{d}f}–{hi:.{d}f}{u}"
            if f in CLIMATE:
                word = "below" if status == "low" else "above"
                tip = CLIMATE_TIPS[(f, status)]
                actions.append(f"{LABELS[f]} ({v:.{d}f}{u}) is {word} the typical {rng} for {crop}. {tip}")
            else:
                verb = "Raise" if status == "low" else "Lower"
                actions.append(f"{verb} {LABELS[f].lower()} from {v:.{d}f}{u} toward {rng} (typical for {crop}).")
    off = sum(c["status"] != "ok" for c in checks)
    return {
        "crop": crop,
        "condition": "Healthy" if off == 0 else "Needs attention" if off <= 3 else "Poor fit",
        "out_of_range": off,
        "checks": checks,
        "actions": actions or [f"All readings fall inside the typical range for {crop}."],
        "method": "Readings compared to the 10th–90th percentile of observed conditions for this crop in the training data.",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "envirosense-api", "version": app.version}


@app.get("/metrics")
def metrics() -> dict:
    return METRICS


@app.get("/crops")
def crops() -> dict:
    return {"crops": CROPS, "profiles": PROFILES}


@app.post("/predict", response_model=Prediction)
def predict(reading: Reading) -> Prediction:
    values = reading.model_dump(exclude={"target_crop"})
    X = pd.DataFrame([values], columns=FEATURES)
    proba = rf.predict_proba(X)[0]
    ranked = sorted(zip(rf.classes_, proba), key=lambda t: -t[1])
    lr_proba = lr.predict_proba(X)[0]
    lr_best = max(zip(lr.classes_, lr_proba), key=lambda t: t[1])

    crop = ranked[0][0]
    if reading.target_crop:
        t = reading.target_crop.strip().lower()
        if t not in PROFILES:
            raise HTTPException(422, f"Unknown crop '{reading.target_crop}'. Options: {', '.join(CROPS)}")
        crop = t

    return Prediction(
        recommended_crop=ranked[0][0],
        confidence=round(float(ranked[0][1]), 4),
        top_crops=[{"crop": c, "probability": round(float(p), 4)} for c, p in ranked[:3]],
        baseline={"model": "logistic_regression", "crop": lr_best[0], "probability": round(float(lr_best[1]), 4)},
        assessment=assess(values, crop),
        model_version=f"rf-200 · trained {METRICS['trained_at'][:10]}",
    )


# Backwards compatible with the v1 app (flat body, "ph" key).
@app.post("/get_recommendations", include_in_schema=False)
def legacy(reading: Reading) -> Prediction:
    return predict(reading)
