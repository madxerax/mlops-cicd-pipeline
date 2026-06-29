"""Tests del modelo: la corrida de entrenamiento debe superar umbrales.

Estos tests entrenan un modelo pequeño y verifican que la calidad mínima se
cumple. Actúan como una segunda capa del quality gate, ejecutándose dentro de
pytest en CI.
"""
from __future__ import annotations

from src.train import train
from src.validate_model import evaluate


def test_training_meets_quality_thresholds():
    metrics = train(n_estimators=100)
    assert metrics["accuracy"] >= 0.90
    assert metrics["f1_macro"] >= 0.90
    assert metrics["log_loss"] <= 0.5


def test_quality_gate_logic_flags_bad_model():
    bad = {"accuracy": 0.5, "f1_macro": 0.4, "log_loss": 1.2}
    failures = evaluate(bad, min_acc=0.9, min_f1=0.9, max_ll=0.4)
    assert len(failures) == 3


def test_quality_gate_logic_passes_good_model():
    good = {"accuracy": 0.98, "f1_macro": 0.98, "log_loss": 0.1}
    failures = evaluate(good, min_acc=0.9, min_f1=0.9, max_ll=0.4)
    assert failures == []
