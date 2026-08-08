.PHONY: install install-dev data prep prep-small train-local gguf ollama-import fixtures test lint bench api serve-docs docker-up clean

install:
	pip install -e ".[train]"

install-dev:
	pip install -e ".[dev]"

install-local:
	pip install -e ".[local,dev]"
	pip install torch --index-url https://download.pytorch.org/whl/cu124

data:
	python scripts/download_dataset.py

prep:
	python -m zeroerr.data.prep -i data/raw/spider_train.jsonl -o data/chatml/train.jsonl --per-bucket 2000 --with-repairs --val-fraction 0.1

prep-small:
	python -m zeroerr.data.prep -i data/raw/spider_train.jsonl -o data/chatml/train_local.jsonl --per-bucket 600 --with-repairs

train-local:
	python scripts/train_local.py --data data/chatml/train_local.jsonl --model 1.5b --out checkpoints/zeroerr-1.5b-merged

gguf:
	python scripts/convert_gguf.py --input checkpoints/zeroerr-1.5b-merged --output gguf/zeroerr-1.5b-q8_0.gguf --ftype q8_0

ollama-import:
	bash scripts/setup_ollama.sh gguf/zeroerr-1.5b-q8_0.gguf

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