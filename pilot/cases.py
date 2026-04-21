"""Hand-crafted synthetic cases for Stage 1 smoke test.

Each case exposes:
  - documents: list of typed chunks the agent ingests sequentially
  - ground_truth.facts: {key: value} pairs the decision must reference exactly
  - ground_truth.reasoning_points: inferences the rationale must entail
  - ground_truth.decision: the correct decision label
  - ground_truth.required_provisions: for CRR, the reasons a compliant
    denial notice must cite

Cases are intentionally hand-crafted (not LLM-generated) for the smoke test
so ground truth is deterministic and costs stay low.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal


ChunkType = Literal["fact", "reasoning", "mixed"]


@dataclass
class Chunk:
    text: str
    type: ChunkType
    # Optional structured payload for facts: (key, value) pair
    key: str | None = None
    value: str | None = None


@dataclass
class GroundTruth:
    decision: str
    facts: Dict[str, str] = field(default_factory=dict)
    reasoning_points: List[str] = field(default_factory=list)
    required_provisions: List[str] = field(default_factory=list)
    rationale_exemplar: str = ""


@dataclass
class Case:
    case_id: str
    domain: Literal["loan", "claim"]
    documents: List[Chunk]
    ground_truth: GroundTruth
    task_prompt: str


# -----------------------------------------------------------------------------
# LOAN CASE: approve with an employment-gap explanation required in rationale.
# -----------------------------------------------------------------------------

LOAN_CASE = Case(
    case_id="loan_001",
    domain="loan",
    task_prompt=(
        "You are a mortgage underwriter. Based on the provided memory, decide "
        "whether to APPROVE or DENY this application. Produce (a) the decision, "
        "(b) a rationale memo that cites specific numeric facts from the file, "
        "and (c) if denied, an adverse action notice citing specific reasons."
    ),
    documents=[
        # Identity
        Chunk("Application received 2026-03-02. Applicant: Jane Doe, DOB 1988-06-15, "
              "primary residence: 421 Elm St, Portland OR.", "fact"),
        # Income facts
        Chunk("W-2 tax year 2025: gross annual wages $142,500 from Pacific Logistics Inc.",
              "fact", key="income_2025", value="$142,500"),
        Chunk("W-2 tax year 2024: gross annual wages $68,200 from Pacific Logistics Inc. "
              "(partial year — 5 months).",
              "fact", key="income_2024", value="$68,200"),
        Chunk("Pay stub dated 2026-02-14: YTD gross $24,807; employer Pacific Logistics Inc.",
              "fact", key="ytd_gross_2026", value="$24,807"),
        Chunk("Tax return 2025 filed 2026-02-10. AGI $138,405; filing status single.",
              "fact", key="agi_2025", value="$138,405"),
        # Credit
        Chunk("Credit report pulled 2026-03-05. FICO 8: 742. Total revolving utilization: 18%. "
              "No delinquencies in past 24 months.",
              "fact", key="fico_score", value="742"),
        Chunk("Open tradelines: 4. Oldest account: 2011. No collections.", "fact"),
        # Property
        Chunk("Property: 821 Oak Ave, Portland OR. Purchase price per contract: $585,000.",
              "fact", key="purchase_price", value="$585,000"),
        Chunk("Appraisal dated 2026-03-10 by Alpine Valuation. Appraised value: $592,000.",
              "fact", key="appraised_value", value="$592,000"),
        Chunk("Loan amount requested: $468,000. LTV: 80.0%.",
              "fact", key="loan_amount", value="$468,000"),
        # Bank statements
        Chunk("Chase checking statement Dec 2025: opening $48,210, closing $51,884. "
              "Payroll deposits bi-weekly.", "mixed"),
        Chunk("Chase checking statement Jan 2026: large deposit of $12,000 on 2026-01-18. "
              "Memo: 'from B. Doe'.", "mixed"),
        # Narrative / reasoning ambiguities that need resolution
        Chunk("Underwriter note 2026-03-06: 2024 W-2 shows only partial year income. "
              "Need explanation for employment gap Jan-Jul 2024.", "reasoning"),
        Chunk("Applicant letter of explanation dated 2026-03-08: 'I was on FMLA-protected "
              "medical leave from January through June 2024 following surgery. Documentation "
              "from Providence Medical attached. Returned to full duty 2024-07-01, same "
              "employer, same role, same base salary.'", "reasoning"),
        Chunk("Providence Medical leave verification letter dated 2026-03-09 confirms "
              "medical leave 2024-01-09 through 2024-06-28, return-to-work 2024-07-01.",
              "reasoning"),
        Chunk("Underwriter decision note 2026-03-10: employment gap is explained by "
              "documented medical leave; does not indicate income instability. "
              "Current W-2 and YTD figures consistent with stable $142k salary.",
              "reasoning"),
        # Large deposit resolution
        Chunk("Underwriter request 2026-03-11: source of $12,000 deposit 2026-01-18 "
              "required.", "reasoning"),
        Chunk("Applicant letter dated 2026-03-12: 'The $12,000 deposit on January 18 "
              "was a gift from my father Benjamin Doe, for use toward closing costs. "
              "Gift letter attached.'", "reasoning"),
        Chunk("Gift letter from Benjamin Doe dated 2026-03-12, signed: attests $12,000 "
              "is a gift, not a loan, no repayment expected. Donor bank statement provided.",
              "reasoning"),
        Chunk("Underwriter acceptance 2026-03-13: gift documented per Fannie Mae "
              "guidelines; funds sourced.", "reasoning"),
        # Rate lock
        Chunk("Rate lock executed 2026-03-14: 6.25% fixed, 30-year term, 45-day lock.",
              "fact", key="locked_rate", value="6.25%"),
        # Final
        Chunk("DTI calculated 2026-03-15: monthly debt $1,205 + proposed housing $3,420 "
              "= $4,625. Monthly income $11,875 (based on $142,500). DTI = 38.9%.",
              "mixed", key="dti", value="38.9%"),
    ],
    ground_truth=GroundTruth(
        decision="APPROVE",
        facts={
            "income_2025": "$142,500",
            "fico_score": "742",
            "appraised_value": "$592,000",
            "loan_amount": "$468,000",
            "dti": "38.9%",
            "locked_rate": "6.25%",
        },
        reasoning_points=[
            "The 2024 employment gap is explained by documented FMLA medical leave, "
            "not income instability.",
            "The $12,000 January 2026 deposit is a documented gift from Benjamin Doe, "
            "with executed gift letter and donor bank statement per Fannie Mae guidelines.",
            "DTI of 38.9% is within acceptable range; LTV of 80% meets conforming "
            "loan standards.",
        ],
        required_provisions=[],  # approve case — no adverse action notice required
        rationale_exemplar=(
            "APPROVE. Applicant Jane Doe requests a $468,000 loan against a property "
            "appraised at $592,000 (LTV 80.0%). Income of $142,500 (2025 W-2) and "
            "FICO 742 support the application. The 2024 employment gap is explained "
            "by documented FMLA medical leave (2024-01-09 to 2024-06-28) verified by "
            "Providence Medical, indicating a temporary interruption rather than income "
            "instability. The $12,000 deposit on 2026-01-18 is a documented gift from "
            "Benjamin Doe (father), executed per Fannie Mae guidelines. DTI calculates "
            "to 38.9% at the locked rate of 6.25%, within conforming guidelines."
        ),
    ),
)


# -----------------------------------------------------------------------------
# CLAIM CASE: partial pay with specific exclusion rationale required.
# -----------------------------------------------------------------------------

CLAIM_CASE = Case(
    case_id="claim_001",
    domain="claim",
    task_prompt=(
        "You are an insurance claim adjudicator. Based on the provided memory, "
        "decide whether to PAY in full, PARTIAL PAY, or DENY this claim. "
        "Produce (a) the decision, (b) a rationale memo citing specific policy "
        "provisions and dollar amounts, and (c) if denying or partial-paying, a "
        "denial/partial notice citing the specific provisions applied."
    ),
    documents=[
        # Policy
        Chunk("Policy HO-3 #POL-884219, insured: Maria Chen, property: 1142 Birch Ln, "
              "Austin TX. Effective 2025-04-01 to 2026-04-01.",
              "fact", key="policy_number", value="POL-884219"),
        Chunk("Coverage A (dwelling): $420,000. Coverage B (other structures): $42,000. "
              "Coverage C (personal property): $210,000. Deductible: $2,500.",
              "fact", key="dwelling_limit", value="$420,000"),
        Chunk("Coverage A Exclusion 7: 'We do not cover loss caused by wear, tear, "
              "deterioration, rust, mold, or gradual seepage of water over time.'",
              "fact", key="exclusion_7", value="wear/tear/gradual seepage"),
        Chunk("Coverage A Exclusion 9: 'We do not cover loss caused by settling, "
              "cracking, shrinking, or expansion of foundations.'",
              "fact", key="exclusion_9", value="settling/cracking"),
        # FNOL
        Chunk("First Notice of Loss 2026-02-07: insured reports 'water damage in "
              "basement discovered 2026-02-06; think a pipe burst'.",
              "mixed", key="date_of_loss_reported", value="2026-02-06"),
        # Investigation facts
        Chunk("Inspection by adjuster Tom Rivera 2026-02-09. Scope: basement wall "
              "staining, mildew smell, drywall damage on north wall bottom 18 inches.",
              "mixed"),
        Chunk("Plumber invoice dated 2026-02-10 from Austin Plumbing Pro: "
              "'No active leak found. Evidence of long-term moisture ingress "
              "through wall at grade. No burst pipe. $450 service fee.'",
              "reasoning"),
        Chunk("Repair estimate dated 2026-02-14 from CertifiedDry: $18,400 total "
              "(drywall $6,200, mold remediation $9,800, painting $2,400).",
              "fact", key="repair_estimate", value="$18,400"),
        Chunk("Moisture mapping report dated 2026-02-15: wall moisture 22-31%, "
              "consistent with prolonged exposure over weeks to months, not a "
              "single-event leak.", "reasoning"),
        # Prior claim history
        Chunk("Policy history check 2026-02-16: prior claim 2024-11, paid $3,200 "
              "for roof wind damage (unrelated).",
              "fact", key="prior_claim_amount", value="$3,200"),
        # Allocation reasoning
        Chunk("Adjuster note 2026-02-17: drywall and painting ($8,600) likely fall "
              "under Coverage A but causation analysis needed given moisture-mapping "
              "report.", "reasoning"),
        Chunk("Adjuster note 2026-02-18: mold remediation ($9,800) explicitly "
              "excluded under Exclusion 7 (mold/gradual seepage).",
              "reasoning"),
        Chunk("Claimant correspondence 2026-02-19: Maria Chen asserts she was "
              "unaware of any prior moisture and requests full payment.",
              "reasoning"),
        Chunk("Adjuster response 2026-02-20: even absent insured awareness, "
              "coverage turns on cause (gradual vs sudden), not knowledge. "
              "Moisture mapping establishes gradual cause.",
              "reasoning"),
        # Final facts
        Chunk("Final allocation 2026-02-22: drywall $6,200 and painting $2,400 "
              "paid under Coverage A (sudden-seeming surface damage judged "
              "incidental to gradual cause but covered under narrow reading). "
              "Mold remediation $9,800 denied under Exclusion 7. "
              "Deductible $2,500 applied. Net payment: $6,100.",
              "fact", key="net_payment", value="$6,100"),
        Chunk("Denial reason for mold portion: 'Exclusion 7: wear, tear, "
              "deterioration, rust, mold, or gradual seepage of water over time.'",
              "fact", key="denial_provision", value="Exclusion 7"),
    ],
    ground_truth=GroundTruth(
        decision="PARTIAL PAY",
        facts={
            "policy_number": "POL-884219",
            "dwelling_limit": "$420,000",
            "repair_estimate": "$18,400",
            "net_payment": "$6,100",
            "denial_provision": "Exclusion 7",
        },
        reasoning_points=[
            "The moisture mapping report establishes a gradual cause (weeks to months "
            "of exposure) rather than a single-event sudden loss, which determines "
            "coverage applicability.",
            "Mold remediation ($9,800) is explicitly excluded under Exclusion 7 "
            "(wear, tear, deterioration, rust, mold, or gradual seepage).",
            "Drywall and painting ($8,600 gross) are paid under Coverage A less the "
            "$2,500 deductible, yielding net $6,100.",
        ],
        required_provisions=["Exclusion 7"],
        rationale_exemplar=(
            "PARTIAL PAY. Claim POL-884219 for water damage at 1142 Birch Ln, "
            "date of loss reported 2026-02-06. Moisture mapping and plumber "
            "inspection establish a gradual cause (prolonged moisture ingress) "
            "rather than a sudden loss. Mold remediation of $9,800 is denied "
            "under Coverage A Exclusion 7 (wear, tear, deterioration, rust, mold, "
            "or gradual seepage). Drywall ($6,200) and painting ($2,400) totaling "
            "$8,600 are paid under Coverage A. Deductible of $2,500 is applied. "
            "Net payment: $6,100."
        ),
    ),
)


ALL_CASES: List[Case] = [LOAN_CASE, CLAIM_CASE]


if __name__ == "__main__":
    for c in ALL_CASES:
        total_chars = sum(len(d.text) for d in c.documents)
        print(f"{c.case_id}: {len(c.documents)} chunks, ~{total_chars} chars, "
              f"~{total_chars // 4} tokens (approx)")
