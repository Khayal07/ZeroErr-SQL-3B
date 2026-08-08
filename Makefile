.PHONY: install install-dev data prep fixtures test lint bench api serve-docs docker-up clean

install:
	pip install -e ".[train]"

install-dev:
	pip install -e ".[dev]"

data:
	python scripts/download_dataset.py

prep:
	python -m zeroerr.data.prep -i data/raw/spider_train.jsonl -o data/chatml/train.jsonl --per-bucket 2000 --with-repairs --val-fraction 0.1

fixtures:
	python -m eval.fixtures.build

test:
	pytest tests/ -v

lint:
	ruff check src tests eval | true
	ruff format --check src tests eval | true

bench:
	python eval/run_benchmark.py --split dev

api:
	uvicorn zeroerr.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose -f docker/docker-compose.yml up --build

clean:
	@rm -rf .pytest_cache .ruff_cache build dist **/__pycache__