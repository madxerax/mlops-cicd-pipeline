# MLOps Pipeline — CI/CD con GitHub Actions

Proyecto de ejemplo, completo y funcional, que muestra un pipeline de
**CI/CD/CT** para Machine Learning. Entrena un clasificador (dataset *wine* de
scikit-learn), valida datos y modelo con quality gates, lo sirve vía FastAPI y
lo empaqueta en Docker. Pensado como material de curso de MLOps.

## Estructura

```
mlops-pipeline/
├── .github/workflows/
│   ├── ci.yml          # Lint -> Tests (matriz) -> Entrenar+Gate -> Build Docker
│   └── cd.yml          # Build+push de imagen y deploy canary (al pasar CI en main)
├── src/
│   ├── data.py         # Carga + validación de esquema (contrato de datos)
│   ├── model.py        # Pipeline scikit-learn (scaler + RandomForest)
│   ├── train.py        # Entrena, evalúa y persiste model.joblib + metrics.json
│   ├── validate_model.py  # Quality gate: falla CI si el modelo no cumple umbrales
│   ├── predict.py      # Inferencia reutilizable
│   └── app.py          # API de serving (FastAPI): /health y /predict
├── tests/
│   ├── test_data.py    # Tests del contrato de datos
│   ├── test_model.py   # Tests de calidad del modelo
│   └── test_predict.py # Tests de inferencia
├── Dockerfile
├── Makefile
├── pyproject.toml      # Config de ruff y pytest
├── requirements.txt
└── requirements-dev.txt
```

## El pipeline de CI (`ci.yml`), job por job

| Job | Depende de | Qué hace |
|-----|-----------|----------|
| `lint` | — | Análisis estático con ruff. Falla rápido y barato. |
| `test` | `lint` | Suite de pytest en matriz (Python 3.10/3.11/3.12) con cobertura. |
| `train-and-gate` | `test` | Entrena el modelo y aplica el **quality gate** (accuracy, F1, log-loss). Sube `artifacts/`. |
| `docker-build` | `train-and-gate` | Construye la imagen de serving y hace smoke test al endpoint `/health`. |

Cada flecha es una compuerta: si el lint, un test o el quality gate falla, el
pipeline se detiene y nada avanza. Así el despliegue es seguro por construcción.

## Uso local

```bash
# 1. Instalar dependencias de desarrollo
make install            # o: pip install -r requirements-dev.txt

# 2. Lint + tests + entrenamiento + quality gate (todo el flujo de CI)
make all

# Pasos individuales:
make lint               # ruff check
make test               # pytest con cobertura
make train              # genera artifacts/model.joblib y metrics.json
make gate               # valida métricas contra umbrales

# 3. Levantar la API de serving
make serve              # http://localhost:8000/docs
```

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Respuesta:

```json
{
  "prediction": 0,
  "confidence": 0.99,
  "probabilities": {"0": 0.99, "1": 0.005, "2": 0.005}
}
```

## Docker

```bash
make train              # el modelo debe existir antes de construir la imagen
make docker             # docker build -t mlops-pipeline:local .
docker run -p 8000:8000 mlops-pipeline:local
```

## Quality gate del modelo

Los umbrales viven en `ci.yml` (variables `MIN_ACCURACY`, `MIN_F1`,
`MAX_LOG_LOSS`) y se pasan a `src/validate_model.py`. Subirlos o bajarlos
ajusta cuán estricta es la compuerta sin tocar el código del modelo.

## CD (`cd.yml`)

Se dispara automáticamente cuando CI termina con éxito en `main`. Entrena el
modelo de release, publica la imagen en GitHub Container Registry (GHCR) y
despliega en estrategia **canary** (10% → verificación → 100%). Los comandos de
deploy están como plantilla comentada para que conectes tu destino real
(Cloud Run, Kubernetes, etc.).

---

Material didáctico · Xerax × UAI · Programa de Postgrado en IA.
