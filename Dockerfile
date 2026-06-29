FROM python:3.12-slim

# Evita .pyc y fuerza logs sin buffer (mejor para contenedores).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependencias primero para aprovechar la cache de capas de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código y el modelo entrenado (artefacto producido por CI).
COPY src/ ./src/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Usuario no-root por seguridad.
RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
