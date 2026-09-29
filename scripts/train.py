"""Train and evaluate the EnviroSense crop-recommendation models.

Reproducible: `python scripts/train.py` rebuilds models/ and models/metrics.json
from data/crop_recommendation.csv. Every number the website shows comes from here.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crop_recommendation.csv"
OUT = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42


def crop_profiles(df: pd.DataFrame) -> dict:
    """Per-crop observed ranges (10th-90th percentile) used for field assessment."""
    profiles = {}
    for crop, g in df.groupby("label"):
        profiles[crop] = {
            f: {
                "low": round(float(g[f].quantile(0.10)), 2),
                "high": round(float(g[f].quantile(0.90)), 2),
                "median": round(float(g[f].median()), 2),
            }
            for f in FEATURES
        }
    return profiles


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        report = classification_report(y_te, pred, output_dict=True, zero_division=0)
        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 6),
            "cv_accuracy_std": round(float(cv_scores.std()), 6),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 6),
            "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 6),
            "per_class_f1": {c: round(v["f1-score"], 4) for c, v in report.items() if c in set(y)},
        }
        print(f"{name}: cv={cv_scores.mean():.4f}±{cv_scores.std():.4f} test={results[name]['test_accuracy']}")

    rf = models["random_forest"]
    imp = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": round(float(m), 4), "std": round(float(s), 4)}
         for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std)),
        key=lambda r: -r["importance"],
    )

    OUT.mkdir(exist_ok=True)
    joblib.dump(rf, OUT / "random_forest.joblib", compress=3)
    joblib.dump(models["logistic_regression"], OUT / "logistic_regression.joblib", compress=3)

    metrics = {
        "task": "crop_recommendation",
        "dataset": {
            "name": "Crop Recommendation Dataset (Kaggle, Atharva Ingle)",
            "rows": int(len(df)),
            "classes": int(y.nunique()),
            "features": FEATURES,
            "split": "80/20 stratified hold-out; 5-fold stratified CV on the training split",
        },
        "models": results,
        "permutation_importance": importance,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "crop_profiles.json").write_text(json.dumps(crop_profiles(df), indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
