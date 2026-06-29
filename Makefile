.PHONY: install lint test train gate serve docker clean all

install:
	pip install -r requirements-dev.txt

lint:
	ruff check src tests

test:
	pytest

train:
	python -m src.train

gate:
	python -m src.validate_model

serve:
	uvicorn src.app:app --reload --port 8000

docker:
	docker build -t mlops-pipeline:local .

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .coverage __pycache__ src/__pycache__ tests/__pycache__

all: lint test train gate
