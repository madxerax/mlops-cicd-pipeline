"""Tests de inferencia: el modelo entrenado predice y respeta el contrato."""
from __future__ import annotations

import pytest

from src.data import EXPECTED_FEATURES, get_dataset
from src.predict import load_model, predict
from src.train import train


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_trained():
    """Garantiza que existe un modelo en disco antes de los tests."""
    train(n_estimators=100)
    load_model.cache_clear()


def _sample_features() -> dict[str, float]:
    data = get_dataset()
    return data.X_test.iloc[0].to_dict()


def test_predict_returns_valid_structure():
    out = predict(_sample_features())
    assert out["prediction"] in {0, 1, 2}
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6


def test_predict_rejects_missing_features():
    incomplete = _sample_features()
    incomplete.pop(EXPECTED_FEATURES[0])
    with pytest.raises(ValueError, match="Faltan features"):
        predict(incomplete)
