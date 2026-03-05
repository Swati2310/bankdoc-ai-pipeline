"""
Data Preprocessing Pipeline.
Converts raw synthetic JSONL data into training-ready format for each task.

Usage:
    python data/preprocessing.py --input_dir data/raw/ --output_dir data/processed/
"""

import argparse
import json
import random
from pathlib import Path

CLAUSE_SEEDS = {
    "default_trigger": [
        "Borrower shall be in default if it fails to make any payment when due.",
        "An Event of Default shall occur upon failure to pay principal or interest within 5 business days.",
        "Default is triggered by non-payment of any scheduled installment.",
        "The occurrence of any payment default shall constitute an Event of Default hereunder.",
        "Failure to remit any amount due under this Agreement within 10 days constitutes default.",
    ],
    "covenant": [
        "Borrower shall maintain a minimum debt service coverage ratio of 1.25x at all times.",
        "The Borrower covenants to maintain a current ratio of not less than 1.5 to 1.0.",
        "Borrower shall not permit its total leverage ratio to exceed 3.0x EBITDA.",
        "The Borrower shall maintain minimum liquidity of $1,000,000 at all times.",
        "Borrower agrees to maintain tangible net worth of no less than $5,000,000.",
    ],
    "termination": [
        "Either party may terminate this Agreement upon 30 days written notice.",
        "Lender may terminate the facility immediately upon occurrence of an Event of Default.",
        "This Agreement shall terminate automatically upon the Maturity Date.",
        "The credit facility may be terminated by Lender in the event of material adverse change.",
        "Borrower may prepay and terminate this facility without penalty after 12 months.",
    ],
    "indemnification": [
        "Borrower shall indemnify Lender against all losses arising from breach of this Agreement.",
        "The Borrower agrees to hold harmless the Lender from any third-party claims.",
        "Borrower shall defend, indemnify, and hold harmless Lender from all liabilities.",
        "Indemnification obligations of Borrower shall survive termination of this Agreement.",
        "Borrower shall reimburse Lender for all costs incurred in enforcing this Agreement.",
    ],
    "representation": [
        "Borrower represents that all financial statements provided are true and accurate.",
        "The Borrower warrants that no material adverse change has occurred since the last audit.",
        "Borrower represents that it is duly organized and validly existing under applicable law.",
        "The Borrower represents that there is no pending litigation that would materially affect operations.",
        "Borrower warrants that the execution of this Agreement does not violate any other obligation.",
    ],
}

CLAUSE_VARIATIONS = [
    "As set forth herein, {base}",
    "For the avoidance of doubt, {base}",
    "Notwithstanding anything to the contrary, {base}",
    "In accordance with market practice, {base}",
    "Subject to the terms of this Agreement, {base}",
    "Pursuant to applicable law, {base}",
    "As a material inducement to Lender, {base}",
    "{base} This provision shall be strictly construed.",
    "{base} Time is of the essence with respect to this obligation.",
    "The parties agree that {base_lower}",
]


def load_jsonl(filepath):
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def split_data(samples, eval_ratio=0.15, seed=42):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def format_ner_samples(loan_docs, kyc_forms):
    samples = []
    for doc in loan_docs:
        samples.append({
            "text": doc["instruction_formatted"],
            "task": "ner",
            "source": "loan_doc",
            "id": doc["id"],
        })
    for form in kyc_forms:
        samples.append({
            "text": form["instruction_formatted"],
            "task": "ner",
            "source": "kyc_form",
            "id": form["id"],
        })
    random.shuffle(samples)
    return samples


def format_clause_samples(loan_docs, num_variations=8, seed=42):
    rng = random.Random(seed)
    samples = []
    categories = list(CLAUSE_SEEDS.keys())
    category_list = ", ".join(categories)

    for category, seed_clauses in CLAUSE_SEEDS.items():
        for seed_clause in seed_clauses:
            for _ in range(num_variations):
                variation_template = rng.choice(CLAUSE_VARIATIONS)
                if "{base_lower}" in variation_template:
                    text = variation_template.replace("{base_lower}", seed_clause[0].lower() + seed_clause[1:])
                else:
                    text = variation_template.replace("{base}", seed_clause)

                instruction = (
                    "### Instruction\n"
                    f"Classify the following loan agreement clause into one of these categories: {category_list}.\n\n"
                    "### Input\n"
                    f"{text}\n\n"
                    "### Response\n"
                    f"{category}"
                )

                samples.append({
                    "text": instruction,
                    "task": "clause",
                    "label": category,
                    "clause_text": text,
                    "id": f"clause_{category}_{len(samples)}",
                })

    rng.shuffle(samples)
    return samples


def format_risk_samples(credit_memos):
    samples = []
    for memo in credit_memos:
        samples.append({
            "text": memo["instruction_formatted"],
            "task": "risk",
            "risk_level": memo["risk_level"],
            "recommendation": memo["recommendation"],
            "id": memo["id"],
        })
    random.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw synthetic data into training sets")
    parser.add_argument("--input_dir", type=str, default="data/raw/")
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    parser.add_argument("--eval_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    loan_docs = load_jsonl(input_dir / "loan_docs.jsonl")
    credit_memos = load_jsonl(input_dir / "credit_memos.jsonl")
    kyc_forms = load_jsonl(input_dir / "kyc_forms.jsonl")
    print(f"  Loan docs:     {len(loan_docs)}")
    print(f"  Credit memos:  {len(credit_memos)}")
    print(f"  KYC forms:     {len(kyc_forms)}")

    print("\nFormatting NER samples...")
    ner_samples = format_ner_samples(loan_docs, kyc_forms)
    print(f"  Total NER samples: {len(ner_samples)}")

    print("Formatting clause classification samples...")
    clause_samples = format_clause_samples(loan_docs, seed=args.seed)
    print(f"  Total clause samples: {len(clause_samples)}")

    print("Formatting risk scoring samples...")
    risk_samples = format_risk_samples(credit_memos)
    print(f"  Total risk samples: {len(risk_samples)}")

    tasks = {
        "ner": ner_samples,
        "clause": clause_samples,
        "risk": risk_samples,
    }

    print(f"\nSplitting with eval_ratio={args.eval_ratio}, seed={args.seed}...")
    for task_name, samples in tasks.items():
        train, eval_ = split_data(samples, eval_ratio=args.eval_ratio, seed=args.seed)
        save_jsonl(train, output_dir / f"{task_name}_train.jsonl")
        save_jsonl(eval_, output_dir / f"{task_name}_eval.jsonl")
        print(f"  {task_name}: {len(train)} train / {len(eval_)} eval")

    total_train = sum(
        len(split_data(s, args.eval_ratio, args.seed)[0]) for s in tasks.values()
    )
    total_eval = sum(
        len(split_data(s, args.eval_ratio, args.seed)[1]) for s in tasks.values()
    )
    print(f"\nTotal: {total_train} train samples, {total_eval} eval samples")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
