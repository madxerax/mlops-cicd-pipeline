"""Entrenamiento del modelo.

Entrena el pipeline, evalúa en el set de test y persiste dos artefactos:
- ``model.joblib``: el modelo serializado (lo que se despliega).
- ``metrics.json``: las métricas de la corrida (lo que evalúa el quality gate).

Ambos son artefactos versionables que el pipeline de CI/CD sube como
build artifacts y que el model registry puede consumir.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .data import get_dataset
from .model import build_model

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def train(n_estimators: int = 200, max_depth: int | None = None) -> dict:
    """Entrena, evalúa y persiste el modelo y sus métricas."""
    data = get_dataset()
    model = build_model(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(data.X_train, data.y_train)

    proba = model.predict_proba(data.X_test)
    preds = proba.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(data.y_test, preds)),
        "f1_macro": float(f1_score(data.y_test, preds, average="macro")),
        "log_loss": float(log_loss(data.y_test, proba)),
        "n_train": int(len(data.X_train)),
        "n_test": int(len(data.X_test)),
        "params": {"n_estimators": n_estimators, "max_depth": max_depth},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Métricas: {json.dumps(metrics, indent=2)}")
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el modelo de clasificación de vinos")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(n_estimators=args.n_estimators, max_depth=args.max_depth)
