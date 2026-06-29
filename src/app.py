"""API de serving (FastAPI).

Expone el modelo como un servicio HTTP con health check y endpoint de
predicción. Esta es la pieza que se empaqueta en Docker y se despliega.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import EXPECTED_FEATURES
from .predict import load_model, predict

app = FastAPI(title="MLOps Wine Classifier", version="0.1.0")


class WineFeatures(BaseModel):
    """Esquema de entrada validado por Pydantic."""

    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0, alias="od280/od315_of_diluted_wines")
    proline: float = Field(..., ge=0)

    model_config = {"populate_by_name": True}


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe para Kubernetes o Cloud Run."""
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.post("/predict")
def predict_endpoint(payload: WineFeatures) -> dict:
    """Devuelve la clase predicha y las probabilidades."""
    features = {k: getattr(payload, k.replace("/", "_").replace(".", "_"), None) for k in EXPECTED_FEATURES}
    # Reconstruye usando los nombres canónicos (con alias) de Pydantic.
    raw = payload.model_dump(by_alias=True)
    features = {k: raw[k] for k in EXPECTED_FEATURES}
    try:
        return predict(features)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
