"""Definición del modelo.

El modelo se expresa como un ``Pipeline`` de scikit-learn (escalado +
clasificador) para que el preprocesamiento viaje siempre junto al estimador.
Esto evita training/serving skew: el mismo objeto serializado se usa en
entrenamiento e inferencia.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import RANDOM_STATE


def build_model(n_estimators: int = 200, max_depth: int | None = None) -> Pipeline:
    """Construye el pipeline de preprocesamiento + clasificador."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
