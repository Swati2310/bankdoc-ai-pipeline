"""
Synthetic KYC (Know Your Customer) Form Generator with entity annotations.

Usage:
    python data/synthetic/generate_kyc_forms.py --num_samples 500 --output data/raw/kyc_forms.jsonl
"""

import argparse
import json
import random
import uuid
from collections import Counter

from faker import Faker

fake = Faker()

BUSINESS_TYPES = [
    "Corporation",
    "Limited Liability Company",
    "Partnership",
    "Sole Proprietorship",
    "Trust",
    "Non-Profit Organization",
]

INDUSTRIES = [
    "Manufacturing",
    "Technology",
    "Healthcare",
    "Real Estate",
    "Financial Services",
    "Retail",
    "Construction",
    "Transportation",
    "Agriculture",
    "Energy",
    "Hospitality",
    "Education",
]

REVENUE_RANGES = [
    "Under $1,000,000",
    "$1,000,000 - $5,000,000",
    "$5,000,000 - $25,000,000",
    "$25,000,000 - $100,000,000",
    "$100,000,000 - $500,000,000",
    "Over $500,000,000",
]

OWNER_TITLES = [
    "Chief Executive Officer",
    "President",
    "Managing Member",
    "General Partner",
    "Sole Proprietor",
    "Trustee",
    "Chairman",
    "Executive Director",
]

RISK_RATINGS = ["Low", "Medium", "High"]
RISK_WEIGHTS = [0.45, 0.35, 0.20]

PEP_STATUSES = ["No", "No", "No", "No", "Yes"]
OFAC_RESULTS = ["Clear", "Clear", "Clear", "Clear", "Match - Pending Review"]

STATES = [fake.state() for _ in range(10)]


def mask_ssn():
    return f"XXX-XX-{random.randint(1000, 9999)}"


def generate_ein():
    return f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"


def weighted_risk():
    return random.choices(RISK_RATINGS, weights=RISK_WEIGHTS, k=1)[0]


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


# ── Templates ─────────────────────────────────────────────────────────────────

def template_structured_form(company_name, entity_type, state, ein, industry,
                              owner_name, owner_title, dob, ssn, address,
                              revenue, risk_rating, pep_status, ofac_result):
    return f"""KNOW YOUR CUSTOMER (KYC) FORM

SECTION 1 — BUSINESS INFORMATION
  Company Name:       {company_name}
  Entity Type:        {entity_type}
  State of Formation: {state}
  EIN / Tax ID:       {ein}
  Industry:           {industry}
  Business Address:   {address}
  Annual Revenue:     {revenue}

SECTION 2 — BENEFICIAL OWNER INFORMATION
  Owner Name:         {owner_name}
  Title:              {owner_title}
  Date of Birth:      {dob}
  SSN (masked):       {ssn}

SECTION 3 — COMPLIANCE SCREENING
  PEP Status:         {pep_status}
  OFAC Screening:     {ofac_result}
  Risk Rating:        {risk_rating}

CERTIFICATION
I certify that the information provided above is accurate and complete.

Authorized Signature: ___________________________
Date: {fake.date_this_year().strftime("%B %d, %Y")}
"""


def template_narrative_report(company_name, entity_type, state, ein, industry,
                               owner_name, owner_title, dob, ssn, address,
                               revenue, risk_rating, pep_status, ofac_result):
    return f"""{company_name} is a {entity_type} formed in the state of {state}, \
operating primarily in the {industry} sector. \
The company's federal tax identification number (EIN) is {ein}, \
and its principal business address is {address}. \
Annual revenues are reported in the range of {revenue}. \
The beneficial owner, {owner_name}, serves as {owner_title} and was born on {dob}. \
The owner's SSN on file is {ssn}. \
PEP screening returned: {pep_status}. OFAC screening result: {ofac_result}. \
Based on the foregoing, the entity has been assigned a risk rating of {risk_rating}.
"""


def generate_sample(template_idx=None):
    company_name = fake.company()
    entity_type = random.choice(BUSINESS_TYPES)
    state = random.choice(STATES) if STATES else fake.state()
    ein = generate_ein()
    industry = random.choice(INDUSTRIES)
    owner_name = fake.name()
    owner_title = random.choice(OWNER_TITLES)
    dob = fake.date_of_birth(minimum_age=25, maximum_age=75).strftime("%B %d, %Y")
    ssn = mask_ssn()
    address = fake.address().replace("\n", ", ")
    revenue = random.choice(REVENUE_RANGES)
    risk_rating = weighted_risk()
    pep_status = random.choice(PEP_STATUSES)
    ofac_result = random.choice(OFAC_RESULTS)

    templates = [template_structured_form, template_narrative_report]
    if template_idx is None:
        template_idx = random.randint(0, len(templates) - 1)

    text = templates[template_idx](
        company_name, entity_type, state, ein, industry,
        owner_name, owner_title, dob, ssn, address,
        revenue, risk_rating, pep_status, ofac_result,
    )

    fields = {
        "COMPANY_NAME": company_name,
        "ENTITY_TYPE": entity_type,
        "STATE": state,
        "EIN": ein,
        "INDUSTRY": industry,
        "OWNER_NAME": owner_name,
        "OWNER_TITLE": owner_title,
        "DOB": dob,
        "SSN": ssn,
        "ADDRESS": address,
        "REVENUE": revenue,
        "RISK_RATING": risk_rating,
    }

    entities = build_entities(text, fields)

    entity_list = "\n".join(
        f'  - {e["label"]}: "{e["text"]}"' for e in entities
    )
    instruction_formatted = (
        "### Instruction\n"
        "Extract all named entities from the following KYC document. "
        "Entity types: COMPANY_NAME, ENTITY_TYPE, STATE, EIN, INDUSTRY, "
        "OWNER_NAME, OWNER_TITLE, DOB, SSN, ADDRESS, REVENUE, RISK_RATING. "
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
            "entity_type": entity_type,
            "industry": industry,
            "risk_rating": risk_rating,
            "template_idx": template_idx,
            "pep_status": pep_status,
            "ofac_result": ofac_result,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic KYC forms")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/raw/kyc_forms.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    samples = [generate_sample(i % 2) for i in range(args.num_samples)]

    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    entity_types = Counter(s["metadata"]["entity_type"] for s in samples)
    industries = Counter(s["metadata"]["industry"] for s in samples)
    risk_ratings = Counter(s["metadata"]["risk_rating"] for s in samples)

    print(f"\nGenerated {len(samples)} KYC form samples -> {args.output}")
    print("\nRisk rating distribution:")
    for rating in RISK_RATINGS:
        count = risk_ratings.get(rating, 0)
        print(f"  {rating}: {count} ({count/len(samples)*100:.1f}%)")
    print("\nEntity type distribution:")
    for et, count in entity_types.most_common():
        print(f"  {et}: {count}")
    print("\nIndustry distribution:")
    for ind, count in industries.most_common():
        print(f"  {ind}: {count}")
    avg_entities = sum(len(s["entities"]) for s in samples) / len(samples)
    print(f"\nAvg entities per sample: {avg_entities:.1f}")


if __name__ == "__main__":
    main()
