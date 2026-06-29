"""Carga y validación de datos.

En MLOps, los datos son un artefacto de primera clase: se versionan, se
validan y se controlan con la misma rigurosidad que el código. Este módulo
define el esquema esperado y una validación que el pipeline de CI ejecuta
como compuerta: si los datos no cumplen el contrato, el pipeline se detiene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Contrato de datos: nombres de features esperados y rangos plausibles.
EXPECTED_FEATURES: list[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]
TARGET = "target"
N_CLASSES = 3
RANDOM_STATE = 42


@dataclass(frozen=True)
class Dataset:
    """Contenedor inmutable para un split de datos."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw() -> pd.DataFrame:
    """Carga el dataset crudo como DataFrame (features + target)."""
    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": TARGET})
    return df


class DataValidationError(ValueError):
    """Se lanza cuando los datos no cumplen el contrato esperado."""


def validate_schema(df: pd.DataFrame) -> None:
    """Valida esquema, nulos, tipos, rangos y balance de clases.

    Lanza ``DataValidationError`` ante el primer incumplimiento. El pipeline
    de CI llama a esta función como quality gate de datos.
    """
    # 1. Columnas presentes
    missing = set(EXPECTED_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas requeridas: {sorted(missing)}")

    # 2. Sin valores nulos
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise DataValidationError(f"Hay nulos en columnas: {null_cols}")

    # 3. Features numéricas y finitas
    for col in EXPECTED_FEATURES:
        if not np.issubdtype(df[col].dtype, np.number):
            raise DataValidationError(f"La feature '{col}' no es numérica")
        if not np.isfinite(df[col].to_numpy()).all():
            raise DataValidationError(f"La feature '{col}' contiene inf/-inf")
        if (df[col] < 0).any():
            raise DataValidationError(f"La feature '{col}' tiene valores negativos")

    # 4. Target válido: enteros en [0, N_CLASSES)
    classes = sorted(df[TARGET].unique().tolist())
    if classes != list(range(N_CLASSES)):
        raise DataValidationError(
            f"Clases inesperadas en target: {classes} (se esperaban {list(range(N_CLASSES))})"
        )

    # 5. Balance mínimo: ninguna clase por debajo del 10% del total
    min_share = df[TARGET].value_counts(normalize=True).min()
    if min_share < 0.10:
        raise DataValidationError(
            f"Clase minoritaria con solo {min_share:.1%} de las muestras (umbral 10%)"
        )


def get_dataset(test_size: float = 0.2) -> Dataset:
    """Devuelve un split train/test ya validado y estratificado."""
    df = load_raw()
    validate_schema(df)
    X = df[EXPECTED_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return Dataset(X_train, X_test, y_train, y_test)
