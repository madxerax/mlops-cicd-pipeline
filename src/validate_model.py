"""Quality gate del modelo.

Este script es la compuerta del pipeline: lee ``artifacts/metrics.json`` y
compara contra umbrales mínimos. Si alguna métrica no cumple, sale con código
distinto de cero y CI marca el job como fallido, impidiendo que un modelo
degradado avance hacia producción.

Uso:
    python -m src.validate_model
    python -m src.validate_model --min-accuracy 0.95 --min-f1 0.95 --max-log-loss 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .train import METRICS_PATH

# Umbrales por defecto (también pueden definirse por variables de entorno en CI).
DEFAULT_MIN_ACCURACY = 0.92
DEFAULT_MIN_F1 = 0.92
DEFAULT_MAX_LOG_LOSS = 0.40


def evaluate(metrics: dict, min_acc: float, min_f1: float, max_ll: float) -> list[str]:
    """Devuelve la lista de fallos del quality gate (vacía si todo pasa)."""
    failures: list[str] = []
    if metrics["accuracy"] < min_acc:
        failures.append(f"accuracy {metrics['accuracy']:.4f} < umbral {min_acc}")
    if metrics["f1_macro"] < min_f1:
        failures.append(f"f1_macro {metrics['f1_macro']:.4f} < umbral {min_f1}")
    if metrics["log_loss"] > max_ll:
        failures.append(f"log_loss {metrics['log_loss']:.4f} > umbral {max_ll}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gate del modelo")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--max-log-loss", type=float, default=DEFAULT_MAX_LOG_LOSS)
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"ERROR: no se encontró {args.metrics}. Entrena el modelo primero.")
        return 2

    metrics = json.loads(args.metrics.read_text())
    failures = evaluate(metrics, args.min_accuracy, args.min_f1, args.max_log_loss)

    print("=== Quality Gate del Modelo ===")
    print(f"accuracy  : {metrics['accuracy']:.4f}  (min {args.min_accuracy})")
    print(f"f1_macro  : {metrics['f1_macro']:.4f}  (min {args.min_f1})")
    print(f"log_loss  : {metrics['log_loss']:.4f}  (max {args.max_log_loss})")

    if failures:
        print("\nGATE FALLIDO:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nGATE APROBADO: el modelo cumple todos los umbrales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
