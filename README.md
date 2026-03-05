# BankDocAI Pipeline

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Swati2310/bankdoc-ai-pipeline/actions/workflows/lint_test.yml/badge.svg)

Enterprise Document Intelligence Pipeline — fine-tunes LoRA and QLoRA adapters on Mistral-7B for three banking NLP tasks: named entity recognition in loan agreements, clause classification, and risk scoring of credit memos. Includes a full MLOps stack with data versioning (DVC), experiment tracking (MLflow + W&B), a FastAPI inference server with adapter hot-swapping, drift monitoring (Evidently), and an auto-retrain pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BankDocAI — 7-Stage Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘

  Stage 1          Stage 2          Stage 3          Stage 4
┌──────────┐     ┌────────────┐   ┌──────────┐     ┌────────────┐
│Ingestion │────▶│Preprocessing│──▶│ Training │────▶│Evaluation  │
│          │     │            │   │LoRA /    │     │NER F1      │
│PDF/DOCX  │     │Tokenize    │   │QLoRA     │     │Clause F1   │
│Synthetic │     │Format      │   │Mistral-7B│     │Risk Acc.   │
│Generators│     │Split       │   │          │     │            │
└──────────┘     └────────────┘   └──────────┘     └─────┬──────┘
                                                          │
  Stage 7          Stage 6          Stage 5               │
┌──────────┐     ┌────────────┐   ┌──────────┐           │
│Monitoring│◀────│  Serving   │◀──│ Registry │◀──────────┘
│          │     │            │   │          │
│Evidently │     │FastAPI     │   │MLflow    │
│Drift Det.│     │Adapter     │   │Model     │
│Auto-     │     │Router      │   │Registry  │
│Retrain   │     │Gradio Demo │   │          │
└──────────┘     └────────────┘   └──────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Swati2310/bankdoc-ai-pipeline.git
cd bankdoc-ai-pipeline
make install-dev
```

### 2. Generate Synthetic Data

```bash
make data-generate
make data-preprocess
```

### 3. Train Adapters (on Colab T4 GPU)

```bash
# Open notebooks/03_lora_finetuning.ipynb  — LoRA NER
# Open notebooks/04_qlora_finetuning.ipynb — QLoRA all tasks

# Or via CLI (requires GPU):
make train-qlora-ner
make train-qlora-clause
make train-qlora-risk
```

### 4. Evaluate & Compare

```bash
make evaluate
make compare
```

### 5. Serve

```bash
make serve          # FastAPI at http://localhost:8000
make demo           # Gradio UI
```

---

## Results

> Fill in after training runs complete.

| Adapter | Task | F1 / Acc | GPU Mem (GB) | Inference (ms/sample) |
|---------|------|----------|--------------|-----------------------|
| LoRA | NER | — | — | — |
| QLoRA | NER | — | — | — |
| QLoRA | Clause Classification | — | — | — |
| QLoRA | Risk Scoring | — | — | — |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Fine-tuning | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate` |
| Base Model | `mistralai/Mistral-7B-v0.3` |
| Data | `datasets`, `faker`, `dvc` |
| Experiment Tracking | `mlflow`, `wandb` |
| Serving | `fastapi`, `uvicorn`, `pydantic` |
| Demo UI | `gradio` |
| Monitoring | `evidently` |
| Testing | `pytest`, `ruff` |
| Containerization | `docker`, `docker-compose` |

---

## Project Structure

```
bankdoc-ai-pipeline/
├── configs/                    # LoRA & QLoRA YAML configs per task
├── data/
│   ├── raw/                    # Generated JSONL documents
│   ├── processed/              # Train/eval splits per task
│   └── synthetic/              # Generator scripts
├── src/
│   ├── data/loader.py          # HuggingFace dataset loader
│   ├── models/                 # LoRA, QLoRA trainers, adapter manager
│   ├── evaluation/             # Metrics, benchmark runner, comparison
│   ├── serving/                # FastAPI app, adapter router, schemas
│   └── monitoring/             # Drift detector, quality monitor, alerts
├── pipelines/                  # Orchestrated train/eval/retrain pipelines
├── notebooks/                  # Colab-ready training & analysis notebooks
├── tests/                      # pytest test suite
├── mlflow/                     # MLflow tracking config
├── docs/                       # Architecture, model cards, API reference
├── demo/gradio_app.py          # Interactive Gradio demo
├── results/                    # Adapters, metrics, plots
├── .github/workflows/          # CI: lint, test, deploy
├── pyproject.toml
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

---

## License

MIT — see [LICENSE](LICENSE).
