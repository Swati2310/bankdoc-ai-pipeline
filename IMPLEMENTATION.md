# BankDocAI — Complete Implementation Guide

> **Use this file in your Claude Code terminal.**
> Feed this to Claude Code and follow each phase sequentially.
> Every phase ends with a git commit — your GitHub stays green.

---

## PROJECT OVERVIEW

**Project:** Enterprise Document Intelligence Pipeline
**Repo Name:** `bankdoc-ai-pipeline`
**Goal:** Fine-tune LoRA & QLoRA adapters on a single base model (Mistral-7B) for 3 banking NLP tasks — entity extraction, clause classification, and risk scoring — with full MLOps pipeline.

**Runtime:** Google Colab free tier (T4 GPU, 16GB VRAM) for training. Local machine for everything else.

---

## PHASE 0 — ENVIRONMENT & REPO SETUP (Day 1)

### Step 0.1: Create the GitHub repo

```bash
mkdir bankdoc-ai-pipeline && cd bankdoc-ai-pipeline
git init
```

### Step 0.2: Create the project structure

Create the following directory tree:

```
bankdoc-ai-pipeline/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lora_trainer.py
│   │   ├── qlora_trainer.py
│   │   └── adapter_manager.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── compare_adapters.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── adapter_router.py
│   │   └── schemas.py
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_detector.py
│       ├── quality_monitor.py
│       └── alerting.py
├── pipelines/
│   ├── train_pipeline.py
│   ├── eval_pipeline.py
│   └── retrain_trigger.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_data_generation.ipynb
│   ├── 03_lora_finetuning.ipynb
│   ├── 04_qlora_finetuning.ipynb
│   ├── 05_evaluation_comparison.ipynb
│   ├── 06_adapter_merging_export.ipynb
│   └── 07_end_to_end_demo.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       ├── lint_test.yml
│       ├── train_on_push.yml
│       └── deploy.yml
├── mlflow/
│   └── tracking_config.py
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   └── api_reference.md
├── demo/
│   └── gradio_app.py
├── results/
│   ├── adapters/
│   ├── metrics/
│   └── plots/
├── README.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── LICENSE
```

### Step 0.3: Create `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "bankdoc-ai-pipeline"
version = "0.1.0"
description = "Enterprise Document Intelligence Pipeline using LoRA & QLoRA"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"

dependencies = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "peft>=0.9.0",
    "trl>=0.7.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.18.0",
    "accelerate>=0.27.0",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25.0",
    "scipy>=1.12.0",
    "faker>=24.0.0",
    "pymupdf>=1.23.0",
    "python-docx>=1.1.0",
    "mlflow>=2.11.0",
    "wandb>=0.16.0",
    "dvc>=3.47.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.28.0",
    "pydantic>=2.6.0",
    "gradio>=4.20.0",
    "evidently>=0.4.15",
    "pyyaml>=6.0.1",
    "rich>=13.7.0",
    "typer>=0.9.0",
    "loguru>=0.7.2",
    "scikit-learn>=1.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
    "ipykernel>=6.29.0",
    "jupyter>=1.0.0",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --tb=short"
```

### Step 0.4: Create `.gitignore`

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
.vscode/
.idea/
data/raw/*.jsonl
data/processed/*.jsonl
results/adapters/
results/metrics/*.json
results/plots/*.png
mlruns/
.env
wandb/
.DS_Store
.ipynb_checkpoints/
.coverage
htmlcov/
coverage.xml
*.log
```

### Step 0.5: Create `Makefile`

Create a Makefile with targets for: `install`, `install-dev`, `lint`, `format`, `test`, `data-generate`, `data-preprocess`, `train-lora-ner`, `train-qlora-ner`, `train-qlora-clause`, `train-qlora-risk`, `train-all`, `evaluate`, `compare`, `serve`, `demo`, `docker-build`, `docker-run`, `clean`.

Each target should call the corresponding Python script with appropriate arguments.

### Step 0.6: Create initial README.md

Write a comprehensive README with:
- Project title and badges (Python version, License, CI status)
- One-paragraph description
- ASCII architecture diagram showing the 7-stage pipeline: Ingestion → Preprocessing → Training → Evaluation → Registry → Serving → Monitoring
- Quick Start section with installation, data generation, training, and serving commands
- Results table (empty placeholder — fill after training)
- Tech stack table
- Project structure tree
- License section

### Step 0.7: First commit

```bash
git add -A
git commit -m "init: project scaffold with pyproject.toml, Makefile, README"
git remote add origin https://github.com/YOUR_USERNAME/bankdoc-ai-pipeline.git
git branch -M main
git push -u origin main
```

---

## PHASE 1 — SYNTHETIC DATA GENERATION (Days 2-4)

### Step 1.1: Loan Agreement Generator (`data/synthetic/generate_loan_docs.py`)

Build a script that generates realistic loan agreement documents with NER annotations.

**Requirements:**
- Use the `faker` library for realistic names, addresses, dates
- Create 5+ diverse templates (formal agreement, facility agreement, promissory note, term sheet, narrative style)
- Generate these entity types with position annotations: `BORROWER`, `LENDER`, `AMOUNT`, `DATE`, `MATURITY_DATE`, `INTEREST_RATE`, `COLLATERAL`, `LOAN_TYPE`
- Each sample must include:
  - `id`: unique UUID
  - `text`: the raw loan document text
  - `entities`: list of `{start, end, label, text}` annotations
  - `instruction_formatted`: Alpaca-style format with `### Instruction`, `### Input`, `### Response` sections
  - `metadata`: loan_type, amount, rate_type, template_idx
- Include realistic loan types: Commercial Real Estate, Term Loan, Revolving Credit, Equipment Financing, Bridge Loan, SBA 7(a), Syndicated, Mezzanine, etc.
- Include realistic collateral types: real estate, equipment, receivables, inventory, securities, personal guarantee, blanket lien
- Amount ranges: $50K-$500M with weighted distribution (more small/mid, fewer large)
- Interest rates: fixed (3.5-12%) or variable (spread + SOFR/Prime)
- CLI: `--num_samples`, `--output`, `--seed` arguments
- Output: JSONL format
- Log distribution stats on completion

```bash
git add data/synthetic/generate_loan_docs.py
git commit -m "feat(data): synthetic loan agreement generator with NER annotations"
```

### Step 1.2: Credit Memo Generator (`data/synthetic/generate_credit_memos.py`)

Build a script that generates credit memorandums with risk level labels (LOW, MEDIUM, HIGH).

**Requirements:**
- Define risk profiles for each level with specific language patterns:
  - **LOW**: strong revenue growth, low debt ratio (0.1-0.5), robust cash flow, stable industry, experienced management, overcollateralized, perfect payment history
  - **MEDIUM**: moderate growth, moderate debt (0.5-1.5), adequate cash flow, moderately cyclical industry, newer management, adequate collateral, minor payment issues
  - **HIGH**: declining revenue, high debt (1.5-4.0), negative cash flow, disrupted industry, management turnover, undercollateralized, delinquencies
- 3+ templates (structured memo, narrative style, bullet-point assessment)
- Facility types: Term Loan, Revolving Credit, Commercial Mortgage, Equipment Finance, Working Capital Line, etc.
- Risk-appropriate recommendations: APPROVE / APPROVE WITH CONDITIONS / DECLINE
- Instruction format for risk scoring task
- Distribution: 30% LOW, 40% MEDIUM, 30% HIGH
- CLI arguments, JSONL output, stats logging

```bash
git add data/synthetic/generate_credit_memos.py
git commit -m "feat(data): synthetic credit memo generator with risk labels"
```

### Step 1.3: KYC Form Generator (`data/synthetic/generate_kyc_forms.py`)

Build a script that generates Know Your Customer forms with entity annotations.

**Requirements:**
- Entity types: `COMPANY_NAME`, `ENTITY_TYPE`, `STATE`, `EIN`, `INDUSTRY`, `OWNER_NAME`, `OWNER_TITLE`, `DOB`, `SSN` (masked), `ADDRESS`, `REVENUE`, `RISK_RATING`
- Business types: Corporation, LLC, Partnership, Sole Proprietorship, Trust, Non-Profit
- Industries: Manufacturing, Technology, Healthcare, Real Estate, Financial Services, Retail, etc.
- Revenue ranges from Under $1M to Over $500M
- Include PEP status and OFAC screening fields
- 2+ templates (structured form, narrative report)
- Position annotations for each entity
- Instruction format for NER task

```bash
git add data/synthetic/generate_kyc_forms.py
git commit -m "feat(data): KYC form generator with entity annotations"
```

### Step 1.4: Data Preprocessing Pipeline (`data/preprocessing.py`)

Build the pipeline that converts raw synthetic data into training-ready format.

**Requirements:**
- `load_jsonl(filepath)` — load JSONL into list of dicts
- `format_ner_samples(loan_docs, kyc_forms)` — combine loan and KYC NER samples, shuffle
- `format_clause_samples(loan_docs)` — generate clause classification samples with 5 categories: `default_trigger`, `covenant`, `termination`, `indemnification`, `representation`. Create multiple variations of each clause type (5-15 per seed clause). Format as instruction with categories listed.
- `format_risk_samples(credit_memos)` — format credit memos for risk scoring
- `split_data(samples, eval_ratio=0.15, seed=42)` — reproducible train/eval split
- `save_jsonl(samples, filepath)` — write JSONL output
- Main pipeline: load all raw files → format per task → split → save to `data/processed/`
- CLI: `--input_dir`, `--output_dir`, `--tokenizer`, `--eval_ratio`, `--seed`
- Log summary: sample counts per task, train/eval split sizes

```bash
git add data/preprocessing.py
git commit -m "feat(data): preprocessing pipeline with train/eval splits"
```

### Step 1.5: Run the data generation pipeline and verify

```bash
python data/synthetic/generate_loan_docs.py --num_samples 1000 --output data/raw/loan_docs.jsonl
python data/synthetic/generate_credit_memos.py --num_samples 500 --output data/raw/credit_memos.jsonl
python data/synthetic/generate_kyc_forms.py --num_samples 500 --output data/raw/kyc_forms.jsonl
python data/preprocessing.py --input_dir data/raw/ --output_dir data/processed/
```

Verify outputs exist and look correct. Add .gitkeep files to data dirs.

```bash
git add data/raw/.gitkeep data/processed/.gitkeep
git commit -m "feat(data): verified data pipeline — 2000 samples across 3 tasks"
```

---

## PHASE 2 — DATA VERSIONING & EXPLORATION (Days 5-6)

### Step 2.1: DVC Integration (`data/dvc.yaml`)

Set up DVC for data versioning.

```bash
pip install dvc
dvc init
```

Create `data/dvc.yaml` tracking the raw and processed data directories. Create `.dvc` files. Add DVC remote configuration (can be local or S3).

```bash
git add data/dvc.yaml .dvc/ .dvcignore
git commit -m "feat(data): DVC integration for data versioning"
```

### Step 2.2: Data Exploration Notebook (`notebooks/01_data_exploration.ipynb`)

Create a Jupyter notebook that:
- Loads all 3 raw datasets
- Shows sample documents from each
- Plots distribution charts: loan types, risk levels, entity type frequencies, clause type distribution
- Analyzes text lengths (tokens) per task
- Shows instruction format examples
- Verifies entity annotation quality (spot-check positions)

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "feat(notebook): data exploration with distribution analysis"
```

### Step 2.3: Data Loader Module (`src/data/loader.py`)

Build the HuggingFace Datasets loader.

**Requirements:**
- `load_jsonl_as_dataset(filepath, text_field="text")` — load JSONL into HF Dataset
- `load_task_datasets(processed_dir, task)` — load train and eval datasets for a task ("ner", "clause", "risk")
- Proper error handling with `FileNotFoundError`
- Logging with loguru

```bash
git add src/data/loader.py
git commit -m "feat(data): HuggingFace dataset loader for task datasets"
```

---

## PHASE 3 — TRAINING CONFIGS & INFRASTRUCTURE (Days 7-9)

### Step 3.1: LoRA Config for NER (`configs/lora_ner.yaml`)

```yaml
model:
  name: "mistralai/Mistral-7B-v0.3"
  trust_remote_code: true
  torch_dtype: "bfloat16"

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

quantization:
  enabled: false

training:
  output_dir: "results/adapters/lora_ner"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.06
  lr_scheduler_type: "cosine"
  max_seq_length: 1024
  eval_strategy: "steps"
  eval_steps: 50
  save_strategy: "steps"
  save_steps: 100
  bf16: true
  gradient_checkpointing: true
  report_to: "wandb"

data:
  task: "ner"
  train_file: "data/processed/ner_train.jsonl"
  eval_file: "data/processed/ner_eval.jsonl"

tracking:
  project: "bankdoc-ai"
  experiment_name: "lora_ner"
  tags: ["lora", "ner", "mistral-7b"]
```

### Step 3.2: Create remaining configs

Create these additional YAML configs following the same pattern:
- `configs/qlora_ner.yaml` — Same as lora_ner but with `quantization.enabled: true`, `load_in_4bit: true`, `bnb_4bit_quant_type: "nf4"`, `bnb_4bit_compute_dtype: "bfloat16"`, `bnb_4bit_use_double_quant: true`
- `configs/qlora_clause.yaml` — QLoRA for clause classification, `r: 32`, `lora_alpha: 64`, `max_seq_length: 2048`, `num_train_epochs: 5`, `batch_size: 2`, `grad_accum: 8`
- `configs/qlora_risk.yaml` — QLoRA for risk scoring, `lora_dropout: 0.1`, `learning_rate: 1.0e-4`, `weight_decay: 0.02`
- `configs/lora_clause.yaml` — LoRA variant for clause comparison

```bash
git add configs/
git commit -m "feat(config): LoRA and QLoRA configs for all 3 tasks"
```

### Step 3.3: QLoRA Trainer (`src/models/qlora_trainer.py`)

Build the core training script.

**Requirements:**
- `load_config(config_path)` — load YAML config
- `setup_quantization(config)` — create `BitsAndBytesConfig` for 4-bit NF4 quantization (or return None if disabled)
- `load_base_model(config, bnb_config)` — load model with `AutoModelForCausalLM.from_pretrained()`, apply `prepare_model_for_kbit_training()` if quantized
- `load_tokenizer(config)` — load tokenizer, set `pad_token = eos_token` if missing
- `setup_lora(model, config)` — create `LoraConfig`, apply with `get_peft_model()`, log trainable params percentage
- `setup_training_args(config)` — create `TrainingArguments` from config, use `optim="paged_adamw_8bit"`, `max_grad_norm=0.3`, `group_by_length=True`
- `train(config)` — orchestrate full pipeline: quantization → model → tokenizer → LoRA → data → trainer → train → save → evaluate
- Use `SFTTrainer` from `trl` with `dataset_text_field="text"`
- CLI: `--config` argument
- Save adapter and tokenizer to output_dir on completion

```bash
git add src/models/qlora_trainer.py
git commit -m "feat(models): QLoRA trainer with SFTTrainer and BitsAndBytes"
```

### Step 3.4: LoRA Trainer (`src/models/lora_trainer.py`)

Create a thin wrapper that reuses the QLoRA trainer but verifies quantization is disabled. This ensures consistent training pipeline for fair comparison.

```bash
git add src/models/lora_trainer.py
git commit -m "feat(models): LoRA trainer (full precision) for comparison"
```

### Step 3.5: W&B Integration

Add Weights & Biases experiment tracking to the trainer. When `report_to: "wandb"` is set in config, the trainer should:
- Initialize W&B run with project name, experiment name, and tags from config
- Log hyperparameters
- Log training loss, eval loss, learning rate per step
- Log final metrics on completion

```bash
git commit -m "feat: Weights & Biases experiment tracking integration"
```

### Step 3.6: GitHub Actions CI (``.github/workflows/lint_test.yml`)

Create a workflow that on push/PR:
- Sets up Python 3.10 and 3.11
- Caches pip packages
- Installs ruff, pytest, and lightweight deps (faker, loguru, pyyaml, scikit-learn, pydantic, fastapi)
- Runs `ruff check src/ tests/ data/`
- Runs `pytest tests/ -v --tb=short --cov=src`

```bash
git add .github/workflows/lint_test.yml
git commit -m "ci: GitHub Actions lint and test workflow"
```

---

## PHASE 4 — TRAINING ON COLAB (Days 10-16)

> **All training happens on Google Colab. Upload your repo or mount from Google Drive.**

### Step 4.1: LoRA Fine-Tuning Notebook (`notebooks/03_lora_finetuning.ipynb`)

Create a Colab-ready notebook that:
1. Installs dependencies: `pip install transformers peft trl bitsandbytes accelerate datasets wandb loguru pyyaml`
2. Clones your repo or mounts Drive
3. Loads the NER dataset using `src.data.loader`
4. Shows GPU info: `torch.cuda.get_device_name()`, memory
5. Loads Mistral-7B with LoRA config (no quantization)
6. Prints trainable parameters
7. Trains with SFTTrainer
8. Logs to W&B
9. Saves adapter
10. Runs quick inference test
11. Profiles GPU memory usage

**Note:** LoRA on 7B may OOM on T4. If so, either:
- Use Phi-3-mini (3.8B) for the LoRA comparison
- Use gradient_checkpointing + batch_size=1 + grad_accum=16

```bash
git add notebooks/03_lora_finetuning.ipynb
git commit -m "feat(notebook): LoRA fine-tuning notebook for Colab"
```

### Step 4.2: QLoRA Fine-Tuning Notebook (`notebooks/04_qlora_finetuning.ipynb`)

Create a Colab-ready notebook that:
1. Same setup as above
2. Loads Mistral-7B with 4-bit NF4 quantization (QLoRA)
3. Shows memory footprint (should be ~6GB vs ~14GB for LoRA)
4. Trains NER adapter with QLoRA
5. Trains Clause adapter with QLoRA
6. Trains Risk adapter with QLoRA
7. Saves all 3 adapters
8. Quick inference test per adapter
9. Memory profiling per training run

```bash
git add notebooks/04_qlora_finetuning.ipynb
git commit -m "feat(notebook): QLoRA fine-tuning notebook — all 3 tasks"
```

### Step 4.3: Train all adapters

Run training for each task. Expected training times on T4:
- QLoRA NER: ~30-45 min
- QLoRA Clause: ~45-60 min
- QLoRA Risk: ~30-45 min

Save all adapters to `results/adapters/`. Each adapter will be ~30-50MB.

```bash
git add results/adapters/.gitkeep
git commit -m "feat(models): trained QLoRA adapters for NER, clause, risk tasks"
```

---

## PHASE 5 — EVALUATION & BENCHMARKING (Days 17-21)

### Step 5.1: Evaluation Metrics (`src/evaluation/metrics.py`)

Build task-specific evaluation functions.

**Requirements:**

**`evaluate_ner(predictions, references)`:**
- Parse JSON predictions (handle malformed output gracefully)
- Calculate per-entity Precision, Recall, F1 using fuzzy matching (exact match or substring match)
- Entity types: BORROWER, LENDER, AMOUNT, DATE, MATURITY_DATE, INTEREST_RATE, COLLATERAL, LOAN_TYPE, COMPANY_NAME, EIN, OWNER_NAME, INDUSTRY
- Return per-entity and overall metrics

**`evaluate_classification(predictions, references, task_name)`:**
- Clean predictions: extract label from formatted model output using regex patterns
- Handle label mismatches gracefully (closest match fallback)
- Calculate: accuracy, macro F1, weighted F1, macro precision, macro recall
- Per-class metrics using `sklearn.classification_report`
- Confusion matrix
- Return full results dict

**`profile_gpu_memory()`:**
- Return GPU name, allocated/reserved/max memory in GB

```bash
git add src/evaluation/metrics.py
git commit -m "feat(eval): task-specific metrics — NER F1, classification F1, GPU profiling"
```

### Step 5.2: Benchmark Runner (`src/evaluation/benchmark.py`)

Build a script that runs evaluation across all adapters.

**Requirements:**
- Load base model with QLoRA config
- For each adapter in `results/adapters/`:
  - Load adapter
  - Run inference on eval set
  - Calculate task-specific metrics
  - Profile GPU memory
  - Time inference latency (per-sample and total)
- Save results to `results/metrics/{adapter_name}_metrics.json`
- CLI: `--adapters`, `--output`, `--max_samples`

```bash
git add src/evaluation/benchmark.py
git commit -m "feat(eval): benchmark runner for all adapters"
```

### Step 5.3: Adapter Comparison (`src/evaluation/compare_adapters.py`)

Build a script that generates comparison plots and tables.

**Requirements:**
- Load all metrics JSON files from `results/metrics/`
- Generate plots using matplotlib:
  - Bar chart: F1 scores per task (LoRA vs QLoRA)
  - Bar chart: GPU memory usage comparison
  - Bar chart: Training time comparison
  - Bar chart: Inference latency comparison
  - Confusion matrices per task (heatmap)
  - Radar chart: overall comparison across all dimensions
- Generate markdown summary table
- Save plots to `results/plots/`
- CLI: `--metrics_dir`, `--output`

```bash
git add src/evaluation/compare_adapters.py
git commit -m "feat(eval): LoRA vs QLoRA comparison with plots"
```

### Step 5.4: Evaluation Notebook (`notebooks/05_evaluation_comparison.ipynb`)

Create a notebook that:
- Runs benchmark on all adapters
- Displays all comparison plots inline
- Shows detailed per-class metrics
- Analyzes error patterns (what does the model get wrong?)
- Summarizes findings with business impact interpretation

```bash
git add notebooks/05_evaluation_comparison.ipynb
git commit -m "feat(notebook): evaluation comparison — LoRA vs QLoRA analysis"
```

### Step 5.5: Update README with results

Fill in the results table with actual metrics from training runs. Add key findings summary.

```bash
git commit -m "docs: results table and benchmark findings in README"
```

---

## PHASE 6 — ADAPTER MANAGEMENT & MLFLOW (Days 22-26)

### Step 6.1: Adapter Manager (`src/models/adapter_manager.py`)

Build the core adapter management system.

**Requirements:**

**`class AdapterManager`:**
- `__init__(base_model_name, adapters_dir, load_in_4bit, device_map)` — load base model (optionally quantized)
- `list_available_adapters()` — scan adapters_dir for directories with `adapter_config.json`
- `load_adapter(adapter_name)` — load LoRA adapter using `PeftModel.from_pretrained()` for first adapter, `model.load_adapter()` for subsequent ones
- `switch_adapter(adapter_name)` — hot-swap to a different loaded adapter using `model.set_adapter()`
- `merge_and_save(adapter_name, output_dir)` — merge adapter into base model with `model.merge_and_unload()`, save full model
- `generate(prompt, max_new_tokens, temperature, top_p)` — run inference with current adapter using `torch.inference_mode()`

```bash
git add src/models/adapter_manager.py
git commit -m "feat(models): adapter manager — load, swap, merge adapters"
```

### Step 6.2: MLflow Tracking (`mlflow/tracking_config.py`)

Set up MLflow experiment tracking.

**Requirements:**
- Configure MLflow tracking URI (local `mlruns/` or remote server)
- Auto-log function that logs: config, hyperparameters, training metrics, adapter artifacts
- Model registration in MLflow Model Registry
- Comparison dashboard setup

```bash
git add mlflow/tracking_config.py
git commit -m "feat(mlflow): MLflow tracking server configuration"
```

### Step 6.3: Adapter Merging Notebook (`notebooks/06_adapter_merging_export.ipynb`)

Create a notebook that:
- Loads trained adapters
- Demonstrates hot-swapping between tasks
- Merges best adapter into base model
- Exports merged model in HuggingFace format
- Optionally exports to GGUF format for llama.cpp inference
- Shows size comparisons (adapter vs merged model)

```bash
git add notebooks/06_adapter_merging_export.ipynb
git commit -m "feat(notebook): adapter merging and export walkthrough"
```

### Step 6.4: Training Pipeline Orchestrator (`pipelines/train_pipeline.py`)

Build an orchestrated pipeline that runs the full flow.

**Requirements:**
- Step 1: Verify data exists, run preprocessing if needed
- Step 2: Load config
- Step 3: Train adapter
- Step 4: Evaluate adapter
- Step 5: Register in MLflow with metrics
- Step 6: Compare with existing adapters
- Support for `--task`, `--method` (lora/qlora), `--config` arguments
- Rich console output with progress

```bash
git add pipelines/train_pipeline.py
git commit -m "feat(pipeline): orchestrated training pipeline (data → train → eval → register)"
```

---

## PHASE 7 — SERVING LAYER & API (Days 27-31)

### Step 7.1: API Schemas (`src/serving/schemas.py`)

Define Pydantic models:
- `TaskType` enum: ner, clause_classification, risk_scoring
- `PredictionRequest`: text, task, max_new_tokens, temperature
- `BatchPredictionRequest`: list of PredictionRequest (max 50)
- `PredictionResponse`: task, adapter, input_preview, output, inference_time_ms
- `BatchPredictionResponse`: results list, total_time_ms
- `HealthResponse`: status, model, available/loaded adapters, current adapter

### Step 7.2: Adapter Router (`src/serving/adapter_router.py`)

Build the routing logic.

**Requirements:**
- Map task types to adapter names: `{"ner": "qlora_ner", "clause_classification": "qlora_clause", "risk_scoring": "qlora_risk"}`
- Task-specific prompt templates (Alpaca format for each task)
- Route incoming requests to correct adapter
- Handle adapter not found gracefully

### Step 7.3: FastAPI Application (`src/serving/api.py`)

Build the API server.

**Requirements:**
- Lifespan handler: load base model and pre-load all adapters on startup
- `GET /health` — health check with model status
- `POST /predict` — single document inference with automatic adapter routing
- `POST /predict/batch` — batch inference (up to 50 documents)
- `GET /adapters` — list available and loaded adapters
- Proper error handling: 503 if model not loaded, 404 if adapter missing, 400 for invalid task
- Request timing in response
- Graceful degradation (demo mode if no model loaded)

```bash
git add src/serving/
git commit -m "feat(serving): FastAPI inference endpoint with adapter routing"
```

### Step 7.4: Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /app
# Install Python, deps, copy source, expose 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add `docker-compose.yml` with the API service.

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Dockerfile and docker-compose for deployment"
```

### Step 7.5: Gradio Demo (`demo/gradio_app.py`)

Build an interactive demo UI.

**Requirements:**
- File upload for PDF/text documents
- Task selector dropdown
- "Analyze" button
- Output display with formatted results
- Side-by-side LoRA vs QLoRA comparison mode
- Example documents pre-loaded

```bash
git add demo/gradio_app.py
git commit -m "feat(demo): Gradio interactive demo with document upload"
```

### Step 7.6: API Tests (`tests/test_api.py`)

Write tests for all API endpoints using `pytest` and `httpx` with `TestClient`.

```bash
git add tests/test_api.py
git commit -m "test: API endpoint tests with pytest"
```

---

## PHASE 8 — MONITORING & AUTO-RETRAIN (Days 32-37)

### Step 8.1: Drift Detector (`src/monitoring/drift_detector.py`)

Build prediction drift detection.

**Requirements:**
- Use Evidently AI for drift analysis
- Monitor: prediction distribution, confidence scores, input text length distribution
- `detect_drift(recent_predictions, reference_predictions)` — compare distributions
- Return drift score and boolean alert
- Configurable threshold (default: 0.05 for statistical test p-value)

```bash
git add src/monitoring/drift_detector.py
git commit -m "feat(monitor): prediction drift detector using Evidently"
```

### Step 8.2: Quality Monitor (`src/monitoring/quality_monitor.py`)

Build prediction quality monitoring.

**Requirements:**
- Track: average confidence score, percentage of low-confidence predictions, edge case detection (very short/long inputs, unusual entity combinations)
- Rolling window statistics
- Alert when quality metrics breach thresholds

```bash
git add src/monitoring/quality_monitor.py
git commit -m "feat(monitor): quality monitor — confidence scores, edge cases"
```

### Step 8.3: Auto-Retrain Trigger (`pipelines/retrain_trigger.py`)

Build automated retraining.

**Requirements:**
- Check drift score from monitoring
- If drift exceeds threshold:
  - Collect recent labeled data (from human review feedback)
  - Append to training set
  - Trigger training pipeline
  - Evaluate new adapter vs current
  - Promote if better (A/B comparison)
- Configurable schedule (check every N hours or N predictions)
- Logging and alerting

```bash
git add pipelines/retrain_trigger.py
git commit -m "feat(pipeline): auto-retrain trigger on drift threshold breach"
```

### Step 8.4: Alerting System (`src/monitoring/alerting.py`)

Build alerting for quality degradation.

**Requirements:**
- Alert channels: console logging, file logging, optional webhook (Slack/email)
- Alert levels: INFO, WARNING, CRITICAL
- Alert conditions: drift detected, quality below threshold, high latency, adapter load failure
- Configurable cooldown (don't spam alerts)

```bash
git add src/monitoring/alerting.py
git commit -m "feat(monitor): alerting system for quality degradation"
```

---

## PHASE 9 — TESTS & DOCUMENTATION (Days 38-40)

### Step 9.1: Data Pipeline Tests (`tests/test_data_pipeline.py`)

Write comprehensive tests:
- Test each generator produces valid samples with correct fields
- Test entity position annotations match text
- Test instruction format contains required sections
- Test multiple samples are unique
- Test credit memos have correct risk levels in output
- Test preprocessing: format functions, split reproducibility, JSONL save/load
- Test end-to-end: generate → preprocess → verify

### Step 9.2: Training Tests (`tests/test_training.py`)

Write tests (mock heavy GPU operations):
- Test config loading and validation
- Test BitsAndBytesConfig creation
- Test LoRA config creation with correct parameters
- Test training args creation from config

### Step 9.3: Inference Tests (`tests/test_inference.py`)

Write tests:
- Test adapter manager initialization
- Test adapter listing
- Test prompt template formatting

### Step 9.4: Full test suite

```bash
git add tests/
git commit -m "test: comprehensive test suite — data, training, inference, API"
```

### Step 9.5: Architecture Documentation (`docs/architecture.md`)

Write detailed architecture doc covering:
- System overview and design decisions
- Data flow diagram
- Adapter routing strategy
- MLOps lifecycle
- Monitoring and retraining loop
- Deployment architecture

### Step 9.6: Model Card (`docs/model_card.md`)

Create a model card for each adapter covering:
- Model description and intended use
- Training data and procedure
- Evaluation results
- Limitations and biases
- Ethical considerations for banking AI

### Step 9.7: API Reference (`docs/api_reference.md`)

Document all API endpoints with request/response examples.

```bash
git add docs/
git commit -m "docs: architecture, model cards, and API reference"
```

---

## PHASE 10 — END-TO-END DEMO & POLISH (Days 41-42)

### Step 10.1: End-to-End Demo Notebook (`notebooks/07_end_to_end_demo.ipynb`)

Create a showcase notebook that:
1. Loads base model with QLoRA
2. Loads all 3 adapters
3. Takes a sample loan agreement → runs NER → extracts entities
4. Takes sample clauses → classifies each
5. Takes a sample credit memo → scores risk
6. Shows adapter hot-swapping in action
7. Demonstrates the combined pipeline: document in → structured output out
8. Shows latency benchmarks

```bash
git add notebooks/07_end_to_end_demo.ipynb
git commit -m "feat(notebook): end-to-end demo — full pipeline showcase"
```

### Step 10.2: Final README polish

Update README with:
- All results filled in
- Badges: Python version, License, CI, Coverage
- GIF or screenshot of Gradio demo (if available)
- Clear "Results" section with comparison table
- "Key Findings" section summarizing LoRA vs QLoRA
- "Business Impact" section

### Step 10.3: Final cleanup and tag

```bash
ruff check src/ tests/ data/ --fix
ruff format src/ tests/ data/
pytest tests/ -v

git add -A
git commit -m "chore: final cleanup, lint fixes, version tag v1.0.0"
git tag v1.0.0
git push origin main --tags
```

---

## GIT COMMIT STRATEGY — KEEPING GITHUB GREEN

### Rules for daily commits:
1. **Commit at least once per working day** — even small improvements count
2. **Use conventional commit messages**: `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `refactor:`, `chore:`
3. **Every commit should be atomic** — one logical change per commit
4. **Never commit broken code** — each commit should pass lint and tests

### Commit schedule overview:

| Week | Focus | Commits |
|------|-------|---------|
| 1 | Scaffold + Data Pipeline | 8-9 commits |
| 2 | LoRA Fine-Tuning | 8-9 commits |
| 3 | QLoRA Fine-Tuning + Comparison | 8-9 commits |
| 4 | MLflow + Adapter Management | 7-8 commits |
| 5 | API + Serving + Docker | 7-8 commits |
| 6 | Monitoring + Docs + Polish | 8-9 commits |

**Total: 50+ commits across 6 weeks**

### Quick commit ideas when you need a green square:
- Update docstrings
- Add type hints
- Improve error messages
- Add a test case
- Update README
- Fix a linting warning
- Add logging to a function
- Improve CLI help text

---

## COLAB-SPECIFIC NOTES

### GPU Memory Management:
```python
# Always clear GPU cache between training runs
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Saving adapters from Colab:
```python
# Option 1: Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
trainer.save_model('/content/drive/MyDrive/bankdoc-ai/results/adapters/qlora_ner')

# Option 2: Push to HuggingFace Hub
model.push_to_hub("your-username/bankdoc-qlora-ner")
```

### If T4 runs out of memory:
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` to 16
3. Reduce `max_seq_length` to 512
4. Use `gradient_checkpointing: true`
5. If still OOM: use Phi-3-mini (3.8B) instead of Mistral-7B for LoRA comparison

---

## KEY LIBRARIES REFERENCE

```
transformers     — Model loading, tokenization
peft             — LoRA/QLoRA adapter creation and management
bitsandbytes     — 4-bit quantization for QLoRA
trl              — SFTTrainer for supervised fine-tuning
datasets         — HuggingFace data loading
accelerate       — Multi-GPU and mixed precision
mlflow           — Experiment tracking and model registry
wandb            — Training visualization
dvc              — Data versioning
fastapi          — REST API serving
gradio           — Demo UI
evidently        — Drift detection
faker            — Synthetic data generation
scikit-learn     — Evaluation metrics
```

---

**You now have everything to build this project from scratch. Feed each phase to Claude Code, commit after each step, and your GitHub will be green for 6 weeks straight. Good luck!**
