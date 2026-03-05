.PHONY: install install-dev lint format test \
        data-generate data-preprocess \
        train-lora-ner train-qlora-ner train-qlora-clause train-qlora-risk train-all \
        evaluate compare serve demo \
        docker-build docker-run clean

# ── Environment ──────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ data/

format:
	ruff format src/ tests/ data/

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Data Pipeline ─────────────────────────────────────────────────────────────

data-generate:
	python data/synthetic/generate_loan_docs.py --num_samples 1000 --output data/raw/loan_docs.jsonl
	python data/synthetic/generate_credit_memos.py --num_samples 500 --output data/raw/credit_memos.jsonl
	python data/synthetic/generate_kyc_forms.py --num_samples 500 --output data/raw/kyc_forms.jsonl

data-preprocess:
	python data/preprocessing.py --input_dir data/raw/ --output_dir data/processed/

# ── Training ──────────────────────────────────────────────────────────────────

train-lora-ner:
	python src/models/lora_trainer.py --config configs/lora_ner.yaml

train-qlora-ner:
	python src/models/qlora_trainer.py --config configs/qlora_ner.yaml

train-qlora-clause:
	python src/models/qlora_trainer.py --config configs/qlora_clause.yaml

train-qlora-risk:
	python src/models/qlora_trainer.py --config configs/qlora_risk.yaml

train-all: train-qlora-ner train-qlora-clause train-qlora-risk

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluate:
	python src/evaluation/benchmark.py --adapters results/adapters/ --output results/metrics/

compare:
	python src/evaluation/compare_adapters.py --metrics_dir results/metrics/ --output results/plots/

# ── Serving ───────────────────────────────────────────────────────────────────

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

demo:
	python demo/gradio_app.py

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t bankdoc-ai-pipeline:latest .

docker-run:
	docker compose up

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ coverage.xml
