"""
Synthetic Loan Agreement Generator with NER annotations.
Generates realistic loan documents in Alpaca instruction format.

Usage:
    python data/synthetic/generate_loan_docs.py --num_samples 1000 --output data/raw/loan_docs.jsonl
"""

import argparse
import json
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()

LOAN_TYPES = [
    "Commercial Real Estate Loan",
    "Term Loan",
    "Revolving Credit Facility",
    "Equipment Financing",
    "Bridge Loan",
    "SBA 7(a) Loan",
    "Syndicated Loan",
    "Mezzanine Loan",
    "Working Capital Line of Credit",
    "Construction Loan",
]

COLLATERAL_TYPES = [
    "commercial real estate located at {address}",
    "manufacturing equipment and machinery",
    "accounts receivable and inventory",
    "marketable securities and investments",
    "personal guarantee of {name}",
    "blanket lien on all business assets",
    "commercial vehicles and fleet assets",
    "intellectual property and trademarks",
]

LENDERS = [
    "First National Bank",
    "Pacific Coast Capital Group",
    "Meridian Commercial Lending",
    "Apex Financial Partners",
    "Heritage Bank & Trust",
    "Summit Capital Advisors",
    "Cornerstone Business Finance",
    "Keystone Lending Solutions",
    "United Commercial Bank",
    "Liberty Financial Group",
]

AMOUNT_RANGES = [
    (50_000, 500_000, 0.35),
    (500_000, 5_000_000, 0.40),
    (5_000_000, 50_000_000, 0.18),
    (50_000_000, 500_000_000, 0.07),
]

RATE_TYPES = ["fixed", "variable"]
SOFR_SPREAD = (1.5, 4.5)
PRIME_SPREAD = (0.5, 3.0)
FIXED_RATE_RANGE = (3.5, 12.0)


def generate_amount():
    rand = random.random()
    cumulative = 0.0
    for low, high, weight in AMOUNT_RANGES:
        cumulative += weight
        if rand <= cumulative:
            amount = random.uniform(low, high)
            if amount >= 1_000_000:
                amount = round(amount / 100_000) * 100_000
            else:
                amount = round(amount / 5_000) * 5_000
            return amount
    return random.uniform(500_000, 5_000_000)


def format_amount(amount):
    if amount >= 1_000_000:
        return f"${amount:,.0f}"
    return f"${amount:,.0f}"


def generate_rate(rate_type):
    if rate_type == "fixed":
        rate = round(random.uniform(*FIXED_RATE_RANGE), 2)
        return f"{rate}% per annum (fixed)"
    else:
        base = random.choice(["SOFR", "Prime Rate"])
        if base == "SOFR":
            spread = round(random.uniform(*SOFR_SPREAD), 2)
        else:
            spread = round(random.uniform(*PRIME_SPREAD), 2)
        return f"{base} plus {spread}% per annum (variable)"


def generate_date(start_offset_days=0, end_offset_days=30):
    base = datetime.now()
    delta = random.randint(start_offset_days, end_offset_days)
    return (base + timedelta(days=delta)).strftime("%B %d, %Y")


def generate_maturity_date(term_years=None):
    if term_years is None:
        term_years = random.choice([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    base = datetime.now()
    maturity = base + timedelta(days=365 * term_years)
    return maturity.strftime("%B %d, %Y"), term_years


def find_span(text, substring):
    idx = text.find(substring)
    if idx == -1:
        return None, None
    return idx, idx + len(substring)


def build_entities(text, fields):
    entities = []
    for label, value in fields.items():
        start, end = find_span(text, value)
        if start is not None:
            entities.append({"start": start, "end": end, "label": label, "text": value})
    return entities


# ── Templates ────────────────────────────────────────────────────────────────

def template_formal_agreement(borrower, lender, amount_str, loan_type, rate_str,
                               date, maturity_date, collateral):
    return f"""LOAN AGREEMENT

This Loan Agreement ("Agreement") is entered into as of {date}, by and between:

BORROWER: {borrower}, a corporation duly organized and existing under the laws of the State of {fake.state()}, with its principal place of business at {fake.address().replace(chr(10), ', ')} ("Borrower");

LENDER: {lender}, a financial institution organized under the laws of the United States, with offices at {fake.address().replace(chr(10), ', ')} ("Lender").

1. LOAN AMOUNT
Lender agrees to make a {loan_type} to Borrower in the principal amount of {amount_str} (the "Loan").

2. INTEREST RATE
The outstanding principal balance of the Loan shall bear interest at a rate of {rate_str}.

3. MATURITY DATE
The entire outstanding principal balance, together with all accrued and unpaid interest, shall be due and payable in full on {maturity_date} (the "Maturity Date").

4. COLLATERAL
As security for the Loan, Borrower hereby grants Lender a first priority security interest in {collateral}.

5. REPRESENTATIONS AND WARRANTIES
Borrower represents and warrants that it has full legal authority to enter into this Agreement and that no event of default exists or is reasonably likely to occur.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

{borrower}
By: ___________________________

{lender}
By: ___________________________
"""


def template_facility_agreement(borrower, lender, amount_str, loan_type, rate_str,
                                 date, maturity_date, collateral):
    return f"""CREDIT FACILITY AGREEMENT

Date: {date}

Parties:
  Borrower:  {borrower}
  Facility Agent / Lender:  {lender}

Facility Details:
  Type:           {loan_type}
  Commitment:     {amount_str}
  Interest Rate:  {rate_str}
  Facility Expiry: {maturity_date}

Security Package:
  The Borrower shall provide the following collateral: {collateral}.

Conditions Precedent:
  1. Delivery of duly executed facility documents.
  2. No material adverse change in Borrower's financial condition.
  3. Evidence of insurance satisfactory to the Lender.

This Facility Agreement is governed by the laws of the State of {fake.state()}.

Executed by:
  {borrower} ___________________
  {lender}   ___________________
"""


def template_promissory_note(borrower, lender, amount_str, loan_type, rate_str,
                              date, maturity_date, collateral):
    return f"""PROMISSORY NOTE

Principal Amount: {amount_str}
Date: {date}

FOR VALUE RECEIVED, {borrower} ("Maker") promises to pay to the order of {lender} ("Holder") the principal sum of {amount_str}, with interest accruing at {rate_str}.

Loan Type: {loan_type}

This Note shall be secured by {collateral}.

The entire unpaid principal balance plus accrued interest shall be due and payable on {maturity_date}.

Default: Failure to pay any amount due within 10 days of its due date shall constitute an Event of Default.

Maker: {borrower}
Date: {date}
"""


def template_term_sheet(borrower, lender, amount_str, loan_type, rate_str,
                         date, maturity_date, collateral):
    return f"""TERM SHEET — NON-BINDING SUMMARY OF PROPOSED TERMS

Prepared: {date}

BORROWER:         {borrower}
LENDER:           {lender}
FACILITY TYPE:    {loan_type}
LOAN AMOUNT:      {amount_str}
INTEREST RATE:    {rate_str}
MATURITY:         {maturity_date}
COLLATERAL:       {collateral}
GOVERNING LAW:    State of {fake.state()}

This Term Sheet is for discussion purposes only and does not constitute a binding commitment.
"""


def template_narrative(borrower, lender, amount_str, loan_type, rate_str,
                        date, maturity_date, collateral):
    return f"""On {date}, {lender} extended a {loan_type} to {borrower} in the amount of {amount_str}. \
The loan carries an interest rate of {rate_str} and matures on {maturity_date}. \
The facility is secured by {collateral}. Both parties agreed to the terms outlined \
in the definitive loan documentation executed on the above date.
"""


TEMPLATES = [
    template_formal_agreement,
    template_facility_agreement,
    template_promissory_note,
    template_term_sheet,
    template_narrative,
]


def generate_sample(template_idx=None):
    borrower = fake.company()
    lender = random.choice(LENDERS)
    amount = generate_amount()
    amount_str = format_amount(amount)
    loan_type = random.choice(LOAN_TYPES)
    rate_type = random.choice(RATE_TYPES)
    rate_str = generate_rate(rate_type)
    date = generate_date(-30, 0)
    maturity_date, term_years = generate_maturity_date()

    collateral_template = random.choice(COLLATERAL_TYPES)
    if "{address}" in collateral_template:
        collateral = collateral_template.format(address=fake.address().replace("\n", ", "))
    elif "{name}" in collateral_template:
        collateral = collateral_template.format(name=fake.name())
    else:
        collateral = collateral_template

    if template_idx is None:
        template_idx = random.randint(0, len(TEMPLATES) - 1)

    text = TEMPLATES[template_idx](
        borrower, lender, amount_str, loan_type, rate_str,
        date, maturity_date, collateral
    )

    fields = {
        "BORROWER": borrower,
        "LENDER": lender,
        "AMOUNT": amount_str,
        "DATE": date,
        "MATURITY_DATE": maturity_date,
        "INTEREST_RATE": rate_str,
        "COLLATERAL": collateral,
        "LOAN_TYPE": loan_type,
    }

    entities = build_entities(text, fields)

    entity_list = "\n".join(
        f'  - {e["label"]}: "{e["text"]}"' for e in entities
    )
    instruction_formatted = (
        "### Instruction\n"
        "Extract all named entities from the following banking document. "
        "Return a JSON list with fields: label, text.\n\n"
        "### Input\n"
        f"{text.strip()}\n\n"
        "### Response\n"
        f"{entity_list}"
    )

    return {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "entities": entities,
        "instruction_formatted": instruction_formatted,
        "metadata": {
            "loan_type": loan_type,
            "amount": amount,
            "rate_type": rate_type,
            "template_idx": template_idx,
            "term_years": term_years,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic loan agreement documents")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/raw/loan_docs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    samples = []
    for i in range(args.num_samples):
        template_idx = i % len(TEMPLATES)
        samples.append(generate_sample(template_idx))

    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    loan_types = Counter(s["metadata"]["loan_type"] for s in samples)
    rate_types = Counter(s["metadata"]["rate_type"] for s in samples)
    templates = Counter(s["metadata"]["template_idx"] for s in samples)

    print(f"\nGenerated {len(samples)} loan agreement samples -> {args.output}")
    print("\nLoan type distribution:")
    for lt, count in loan_types.most_common():
        print(f"  {lt}: {count}")
    print("\nRate type distribution:")
    for rt, count in rate_types.most_common():
        print(f"  {rt}: {count}")
    print("\nTemplate distribution:")
    for ti, count in templates.most_common():
        print(f"  Template {ti}: {count}")
    avg_entities = sum(len(s["entities"]) for s in samples) / len(samples)
    print(f"\nAvg entities per sample: {avg_entities:.1f}")


if __name__ == "__main__":
    main()
