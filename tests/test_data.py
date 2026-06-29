"""Tests de datos: contrato de esquema y validaciones."""
from __future__ import annotations

import pytest

from src.data import (
    EXPECTED_FEATURES,
    N_CLASSES,
    TARGET,
    DataValidationError,
    get_dataset,
    load_raw,
    validate_schema,
)


def test_load_raw_has_expected_columns():
    df = load_raw()
    for col in EXPECTED_FEATURES + [TARGET]:
        assert col in df.columns


def test_validate_schema_passes_on_clean_data():
    df = load_raw()
    validate_schema(df)  # no debe lanzar


def test_validate_schema_detects_nulls():
    df = load_raw()
    df.loc[0, "alcohol"] = None
    with pytest.raises(DataValidationError, match="nulos"):
        validate_schema(df)


def test_validate_schema_detects_missing_column():
    df = load_raw().drop(columns=["hue"])
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        validate_schema(df)


def test_validate_schema_detects_negative_feature():
    df = load_raw()
    df.loc[0, "alcohol"] = -1.0
    with pytest.raises(DataValidationError, match="negativos"):
        validate_schema(df)


def test_validate_schema_detects_unexpected_class():
    df = load_raw()
    df.loc[0, TARGET] = 99
    with pytest.raises(DataValidationError, match="Clases inesperadas"):
        validate_schema(df)


def test_get_dataset_is_stratified_and_split():
    data = get_dataset(test_size=0.2)
    assert len(data.X_train) > len(data.X_test)
    assert list(data.X_train.columns) == EXPECTED_FEATURES
    # Las 3 clases deben aparecer en ambos splits.
    assert data.y_train.nunique() == N_CLASSES
    assert data.y_test.nunique() == N_CLASSES
