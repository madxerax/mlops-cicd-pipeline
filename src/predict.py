"""Inferencia.

Carga el modelo serializado y expone una función de predicción reutilizable
tanto por la API de serving como por los tests.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .data import EXPECTED_FEATURES
from .train import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH):
    """Carga el modelo desde disco una sola vez (cacheado)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo en {path}. Ejecuta 'python -m src.train' primero."
        )
    return joblib.load(path)


def predict(features: dict[str, float]) -> dict:
    """Predice la clase y la probabilidad para un único registro.

    ``features`` debe contener exactamente las claves de ``EXPECTED_FEATURES``.
    """
    missing = set(EXPECTED_FEATURES) - set(features)
    if missing:
        raise ValueError(f"Faltan features en la entrada: {sorted(missing)}")

    row = pd.DataFrame([{k: features[k] for k in EXPECTED_FEATURES}])
    model = load_model()
    proba = model.predict_proba(row)[0]
    label = int(proba.argmax())
    return {
        "prediction": label,
        "confidence": float(proba[label]),
        "probabilities": {str(i): float(p) for i, p in enumerate(proba)},
    }
