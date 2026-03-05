"""
Synthetic Credit Memorandum Generator with risk labels (LOW, MEDIUM, HIGH).

Usage:
    python data/synthetic/generate_credit_memos.py --num_samples 500 --output data/raw/credit_memos.jsonl
"""

import argparse
import json
import random
import uuid
from collections import Counter

from faker import Faker

fake = Faker()

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
RISK_DISTRIBUTION = {"LOW": 0.30, "MEDIUM": 0.40, "HIGH": 0.30}

FACILITY_TYPES = [
    "Term Loan",
    "Revolving Credit Facility",
    "Commercial Mortgage",
    "Equipment Finance",
    "Working Capital Line of Credit",
    "Bridge Loan",
    "SBA 7(a) Loan",
    "Acquisition Finance",
]

RISK_PROFILES = {
    "LOW": {
        "revenue_trend": ["strong revenue growth of {pct}% year-over-year",
                          "consistent revenue expansion of {pct}% over the past three years",
                          "robust top-line growth averaging {pct}% annually"],
        "debt_ratio": (0.1, 0.5),
        "cash_flow": ["robust free cash flow of ${amount}M",
                      "strong operating cash flow exceeding ${amount}M annually",
                      "healthy EBITDA margin of {pct}%"],
        "industry": ["stable and non-cyclical industry", "defensive consumer staples sector",
                     "regulated utility sector with predictable cash flows"],
        "management": ["highly experienced management team with {years}+ years in the industry",
                       "seasoned executive team with proven track record"],
        "collateral": ["overcollateralized at {ratio}x loan value",
                       "fully secured by prime commercial real estate valued at {ratio}x the loan amount"],
        "payment_history": ["perfect payment history with zero delinquencies",
                            "impeccable repayment track record over {years} years",
                            "no instances of late payment in borrower's credit history"],
        "recommendation": "APPROVE",
    },
    "MEDIUM": {
        "revenue_trend": ["moderate revenue growth of {pct}% year-over-year",
                          "steady but uneven revenue with {pct}% average growth",
                          "mixed revenue performance with {pct}% growth in recent periods"],
        "debt_ratio": (0.5, 1.5),
        "cash_flow": ["adequate cash flow of ${amount}M",
                      "sufficient operating cash flow of ${amount}M, though with seasonal variation",
                      "EBITDA margin of {pct}%, adequate to service debt obligations"],
        "industry": ["moderately cyclical industry subject to economic fluctuations",
                     "competitive market with moderate pricing pressure",
                     "industry facing some regulatory headwinds"],
        "management": ["relatively new management team in place for {years} years",
                       "management team with adequate but limited industry experience"],
        "collateral": ["adequately collateralized at {ratio}x loan value",
                       "collateral coverage of {ratio}x, meeting minimum requirements"],
        "payment_history": ["minor payment delays noted on {count} occasions in past 24 months",
                            "one instance of 30-day delinquency in the past 36 months",
                            "generally satisfactory payment history with minor irregularities"],
        "recommendation": "APPROVE WITH CONDITIONS",
    },
    "HIGH": {
        "revenue_trend": ["declining revenue of -{pct}% year-over-year",
                          "significant revenue contraction of -{pct}% over the past two years",
                          "deteriorating top-line performance with -{pct}% decline"],
        "debt_ratio": (1.5, 4.0),
        "cash_flow": ["negative free cash flow of -${amount}M",
                      "insufficient cash flow with operating deficit of ${amount}M",
                      "EBITDA margin compressed to {pct}%, insufficient to cover debt service"],
        "industry": ["highly disrupted industry facing structural challenges",
                     "sector experiencing significant technological disruption",
                     "industry in secular decline with limited recovery prospects"],
        "management": ["recent management turnover with new CEO appointed {months} months ago",
                       "management team lacking relevant experience in current market conditions",
                       "key-person dependency risk with recent departure of CFO"],
        "collateral": ["undercollateralized at {ratio}x loan value",
                       "collateral coverage of only {ratio}x, below required thresholds",
                       "collateral of uncertain value in current market conditions at {ratio}x"],
        "payment_history": ["multiple delinquencies totaling {count} instances in past 12 months",
                            "{count} instances of 60-day delinquency in recent history",
                            "pattern of late payments raising concerns about debt service capacity"],
        "recommendation": "DECLINE",
    },
}


def pick_profile_text(profile_key, risk_level):
    template = random.choice(RISK_PROFILES[risk_level][profile_key])
    pct = round(random.uniform(2, 8), 1) if risk_level == "LOW" else \
          round(random.uniform(1, 5), 1) if risk_level == "MEDIUM" else \
          round(random.uniform(3, 15), 1)
    amount = round(random.uniform(0.5, 20), 1)
    ratio = round(random.uniform(1.5, 3.0), 1) if risk_level == "LOW" else \
            round(random.uniform(1.0, 1.5), 1) if risk_level == "MEDIUM" else \
            round(random.uniform(0.4, 0.9), 1)
    years = random.randint(10, 25) if risk_level == "LOW" else random.randint(3, 8)
    months = random.randint(3, 18)
    count = random.randint(1, 2) if risk_level == "MEDIUM" else random.randint(3, 8)
    return (template
            .replace("{pct}", str(pct))
            .replace("{amount}", str(amount))
            .replace("{ratio}", str(ratio))
            .replace("{years}", str(years))
            .replace("{months}", str(months))
            .replace("{count}", str(count)))


def generate_debt_ratio(risk_level):
    low, high = RISK_PROFILES[risk_level]["debt_ratio"]
    return round(random.uniform(low, high), 2)


def template_structured_memo(company, facility_type, amount_str, risk_level, analyst):
    profile = RISK_PROFILES[risk_level]
    debt_ratio = generate_debt_ratio(risk_level)
    revenue_text = pick_profile_text("revenue_trend", risk_level)
    cash_flow_text = pick_profile_text("cash_flow", risk_level)
    industry_text = pick_profile_text("industry", risk_level)
    management_text = pick_profile_text("management", risk_level)
    collateral_text = pick_profile_text("collateral", risk_level)
    payment_text = pick_profile_text("payment_history", risk_level)
    recommendation = profile["recommendation"]

    return f"""CREDIT MEMORANDUM

Borrower:        {company}
Facility Type:   {facility_type}
Requested Amount:{amount_str}
Risk Rating:     {risk_level}
Prepared By:     {analyst}
Date:            {fake.date_this_year().strftime("%B %d, %Y")}

EXECUTIVE SUMMARY
This memorandum summarizes the credit analysis for {company}'s request for a {facility_type} \
in the amount of {amount_str}.

FINANCIAL ANALYSIS
Revenue: The borrower has demonstrated {revenue_text}.
Cash Flow: {cash_flow_text}.
Debt-to-Equity Ratio: {debt_ratio}x.

INDUSTRY & MARKET ASSESSMENT
{industry_text}.

MANAGEMENT ASSESSMENT
{management_text}.

COLLATERAL ANALYSIS
The proposed facility is {collateral_text}.

PAYMENT HISTORY
{payment_text}.

RECOMMENDATION: {recommendation}

Credit Officer: {analyst}
"""


def template_narrative_memo(company, facility_type, amount_str, risk_level, analyst):
    profile = RISK_PROFILES[risk_level]
    debt_ratio = generate_debt_ratio(risk_level)
    revenue_text = pick_profile_text("revenue_trend", risk_level)
    cash_flow_text = pick_profile_text("cash_flow", risk_level)
    industry_text = pick_profile_text("industry", risk_level)
    collateral_text = pick_profile_text("collateral", risk_level)
    recommendation = profile["recommendation"]

    return f"""{company} has applied for a {facility_type} of {amount_str}. \
The credit review indicates a risk rating of {risk_level}. \
Financially, the company has shown {revenue_text}, with {cash_flow_text}. \
The debt-to-equity ratio stands at {debt_ratio}x. \
The company operates in a {industry_text}. \
The loan is {collateral_text}. \
Based on the foregoing analysis, the recommendation is to {recommendation} this credit request. \
Review conducted by {analyst}.
"""


def template_bullet_memo(company, facility_type, amount_str, risk_level, analyst):
    profile = RISK_PROFILES[risk_level]
    debt_ratio = generate_debt_ratio(risk_level)
    revenue_text = pick_profile_text("revenue_trend", risk_level)
    cash_flow_text = pick_profile_text("cash_flow", risk_level)
    management_text = pick_profile_text("management", risk_level)
    collateral_text = pick_profile_text("collateral", risk_level)
    payment_text = pick_profile_text("payment_history", risk_level)
    recommendation = profile["recommendation"]

    return f"""CREDIT ASSESSMENT — {company}

  • Facility: {facility_type} | Amount: {amount_str} | Risk: {risk_level}
  • Revenue: {revenue_text}
  • Cash Flow: {cash_flow_text}
  • Debt/Equity: {debt_ratio}x
  • Management: {management_text}
  • Collateral: {collateral_text}
  • Payment History: {payment_text}

DECISION: {recommendation}
Analyst: {analyst}
"""


TEMPLATES = [template_structured_memo, template_narrative_memo, template_bullet_memo]


def weighted_risk_level():
    rand = random.random()
    cumulative = 0.0
    for level, weight in RISK_DISTRIBUTION.items():
        cumulative += weight
        if rand <= cumulative:
            return level
    return "MEDIUM"


def generate_amount_str():
    amount = random.choice([
        random.randint(1, 10) * 100_000,
        random.randint(1, 50) * 500_000,
        random.randint(1, 20) * 1_000_000,
        random.randint(5, 100) * 1_000_000,
    ])
    return f"${amount:,.0f}"


def generate_sample(template_idx=None):
    company = fake.company()
    facility_type = random.choice(FACILITY_TYPES)
    amount_str = generate_amount_str()
    risk_level = weighted_risk_level()
    analyst = fake.name()

    if template_idx is None:
        template_idx = random.randint(0, len(TEMPLATES) - 1)

    text = TEMPLATES[template_idx](company, facility_type, amount_str, risk_level, analyst)

    instruction_formatted = (
        "### Instruction\n"
        "Analyze the following credit memorandum and assign a risk rating. "
        "Return one of: LOW, MEDIUM, or HIGH. Then provide a one-sentence justification.\n\n"
        "### Input\n"
        f"{text.strip()}\n\n"
        "### Response\n"
        f"Risk Rating: {risk_level}\n"
        f"Recommendation: {RISK_PROFILES[risk_level]['recommendation']}"
    )

    return {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "risk_level": risk_level,
        "recommendation": RISK_PROFILES[risk_level]["recommendation"],
        "instruction_formatted": instruction_formatted,
        "metadata": {
            "company": company,
            "facility_type": facility_type,
            "amount_str": amount_str,
            "template_idx": template_idx,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit memoranda")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/raw/credit_memos.jsonl")
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

    risk_dist = Counter(s["risk_level"] for s in samples)
    facility_dist = Counter(s["metadata"]["facility_type"] for s in samples)

    print(f"\nGenerated {len(samples)} credit memo samples -> {args.output}")
    print("\nRisk level distribution:")
    for level in RISK_LEVELS:
        count = risk_dist.get(level, 0)
        pct = count / len(samples) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")
    print("\nFacility type distribution:")
    for ft, count in facility_dist.most_common():
        print(f"  {ft}: {count}")


if __name__ == "__main__":
    main()
