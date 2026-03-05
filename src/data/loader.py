"""
HuggingFace Datasets loader for BankDocAI task datasets.
"""

from pathlib import Path

from datasets import Dataset
from loguru import logger


def load_jsonl_as_dataset(filepath: str, text_field: str = "text") -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset.

    Args:
        filepath: Path to the .jsonl file.
        text_field: Name of the field to use as the primary text column.

    Returns:
        HuggingFace Dataset object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info(f"Loading dataset from {filepath}")
    dataset = Dataset.from_json(str(path))
    logger.info(f"Loaded {len(dataset):,} samples (text_field='{text_field}')")
    return dataset


def load_task_datasets(processed_dir: str, task: str) -> tuple[Dataset, Dataset]:
    """Load train and eval datasets for a given task.

    Args:
        processed_dir: Directory containing processed JSONL files.
        task: One of 'ner', 'clause', 'risk'.

    Returns:
        Tuple of (train_dataset, eval_dataset).

    Raises:
        ValueError: If task is not one of the supported tasks.
        FileNotFoundError: If the dataset files do not exist.
    """
    supported_tasks = {"ner", "clause", "risk"}
    if task not in supported_tasks:
        raise ValueError(f"Unsupported task '{task}'. Must be one of: {supported_tasks}")

    base = Path(processed_dir)
    train_path = base / f"{task}_train.jsonl"
    eval_path = base / f"{task}_eval.jsonl"

    logger.info(f"Loading '{task}' task datasets from {processed_dir}")
    train_dataset = load_jsonl_as_dataset(train_path)
    eval_dataset = load_jsonl_as_dataset(eval_path)

    logger.info(
        f"Task '{task}': {len(train_dataset):,} train / {len(eval_dataset):,} eval samples"
    )
    return train_dataset, eval_dataset
