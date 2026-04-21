"""Stage 2 large synthetic cases.

Template-based generator: ground truth is injected by construction, not
inferred post-hoc. Each case exposes the same Case/Chunk API as cases.py
so the existing conditions/metrics pipeline runs unmodified.

Design principles (from paper Section 6.3 Ground Truth Construction):
  1. Sample the target decision and required facts/reasoning points first.
  2. Build supporting documents that contain those anchors verbatim.
  3. Pad with plausible filler (boilerplate, addenda, routing stamps) so
     total content exceeds 5000 tokens (~20k chars). Filler never
     contradicts anchors and is largely content-free from the policy's
     perspective.
  4. Chunk-level type labels ("fact"/"reasoning"/"mixed") come from the
     template, NOT from conditions.classify_chunk. Classifier-free types
     are stored in Chunk.type for ablation bookkeeping; conditions still
     re-classify at ingest as in the protocol spec.

The generator is deterministic: case_id seeds all randomness.
"""
from __future__ import annotations
import random
from typing import List

from cases import Case, Chunk, GroundTruth


# ---------------------------------------------------------------------------
# Filler snippets: boilerplate the real world produces around every doc.
# ---------------------------------------------------------------------------

LOAN_FILLER = [
    "Automated workflow notification (cycle {cycle}, queued {date}): task "
    "assigned to regional analyst pool team {team}. Service-level target 48 "
    "hours. No prior touches on this file. Workflow routing rules v3.2 applied. "
    "Inbound referral source: retail branch channel. Loan officer of record "
    "matches application intake record. No duplicate application detected "
    "within the past 90 days across the enterprise loan origination system.",
    "Standard disclosure packet delivered {date}: Loan Estimate, Closing "
    "Disclosure preview, Fair Lending pamphlet, Consumer Handbook on ARMs "
    "(marked not applicable for this fixed-rate product), Your Home Loan "
    "Toolkit, Privacy Notice. Delivery method: secure borrower portal with "
    "time-stamped acknowledgement. Read receipts received for all six "
    "documents within 24 hours. No borrower questions logged in the portal.",
    "Address validation run {date}: USPS standardized match confirmed with "
    "no deliverability exceptions. County parcel cross-referenced via public "
    "records; assessor parcel number matched to title commitment. Flood zone "
    "determination X (no mandatory flood insurance required under NFIP). "
    "School district and tax district assignment pulled for the borrower's "
    "disclosures. Environmental hazards database shows no adjacent site risks.",
    "Preliminary title report {date} from First American Title Insurance: "
    "fee simple estate, vesting consistent with borrower record. No open "
    "liens, judgments, or pending litigation on the property. One recorded "
    "easement for utility access on the western lot line — standard dominant "
    "estate held by the serving utility. Tax year paid through. Title "
    "insurance binder issued with standard ALTA 9 endorsement and lender's "
    "extended coverage. Recording fees estimated and disclosed.",
    "HMDA reporting flags populated {date}: loan purpose home purchase, "
    "occupancy primary residence, lien status first lien, preapproval not "
    "requested, HOEPA status not high-cost, rate spread computation pending "
    "final APR lock. Demographic information collected per applicant "
    "self-identification. Automated underwriting system recommendation "
    "received and cached for post-decision reconciliation. Reporting "
    "submission scheduled for quarter-end batch.",
    "Fraud services check {date}: LexisNexis RiskView score within normal "
    "range; ID analytics score normal; no OFAC hits on applicant or "
    "co-applicant; no SSA death master inconsistencies; DataVerify ID "
    "validation passed; employment verification vendor Equifax The Work "
    "Number returned a match with current employer; no fraud alerts from "
    "internal blacklists. All fraud gates cleared.",
    "Closing coordinator assignment {date}: estimated closing window within "
    "30 days of clear-to-close. Settlement agent: Summit Escrow Services, "
    "licensed in-state with E&O coverage verified. Wire instructions "
    "verified via callback to a known number on file for the settlement "
    "company. Closing fees quoted and disclosed on the Loan Estimate. "
    "Borrower-selected providers confirmed.",
    "Processor communication log excerpt: borrower contacted {date} regarding "
    "a missing second page of the verification of employment form. Page "
    "received same-day by secure upload. File status updated from initial "
    "submission to conditionally approved pending underwriter review. "
    "Outstanding conditions reduced to three: appraisal review completion, "
    "final VOE recertification within 10 business days of closing, and "
    "homeowner's insurance binder with lender as mortgagee.",
    "Compliance checklist snapshot {date}: TRID timing requirements on "
    "track; ECOA adverse action notice timers not yet triggered; RESPA "
    "3-day rule satisfied on original application; HMDA fields in sync "
    "with LOS record; Regulation Z tolerance analysis pending final "
    "disclosure; state-level high-cost and predatory lending thresholds "
    "cleared. All compliance timers green.",
    "Quality control self-review {date}: income calculation double-checked "
    "against W-2s, pay stubs, and tax transcript requests. Asset "
    "documentation within the required 90-day freshness window. Credit "
    "report within the 120-day freshness window. AUS findings reconciled "
    "with manual calculations. All documents retained in the loan file "
    "per record-retention policy.",
    "Document imaging log {date}: all pages OCR'd and classified by the "
    "document recognition pipeline. Indexer accuracy 99.3% against the "
    "sampled ground-truth subset. Exceptions queue is empty. Redaction "
    "tools applied to sensitive PII for downstream QC review. Archive "
    "retention set to seven years per policy for residential loan files.",
    "Loan officer note {date}: borrower confirms continued employment via "
    "a written attestation, with no anticipated changes in employment, "
    "compensation, or household composition prior to closing. Borrower "
    "also confirms current housing payment history and that there are "
    "no undisclosed debts arising in the past 30 days. Attestation "
    "retained in the electronic file.",
    "Appraisal review {date}: staff reviewer signed off on the appraisal "
    "for uniform standards compliance, no material comp selection concerns, "
    "adjustments within policy tolerance bands, and reconciliation narrative "
    "adequate. No second appraisal ordered. Collateral underwriter score "
    "within acceptable band.",
    "Automated underwriting system run {date}: DU/LP findings recommendation "
    "returned. Fannie Mae DU case file loaded. Approved-Eligible or "
    "Accept-Eligible per product. All validated data points match "
    "source documents. No red flags or unexpected findings. Findings "
    "report saved to the file.",
    "Communication preferences on file: electronic consent valid; no opt-out "
    "received. Notice of privacy practices acknowledged. Spanish-language "
    "materials offered and declined. TCPA preferences recorded. Do-not-call "
    "status checked.",
]

CLAIM_FILLER = [
    "Claim intake system note {date}: First Notice of Loss captured via IVR "
    "with a 7-minute call duration. Automatic catastrophe flag evaluated; "
    "no declared catastrophe active in the property county at time of loss. "
    "Routed to the standard adjuster queue with priority level normal. "
    "Language preference English confirmed. No previous claims open on this "
    "policy at intake time. Policyholder identity verified via date-of-birth "
    "and last four of SSN.",
    "Policyholder correspondence log {date}: auto-acknowledgement email "
    "delivered with claim number, adjuster name, and estimated next contact "
    "window. Read receipt returned within an hour. Spanish-language option "
    "declined. Accommodations for disability: none requested. SMS opt-in "
    "confirmed for status updates. Adjuster voicemail left same day "
    "confirming the inspection appointment.",
    "Vendor network lookup {date}: preferred mitigation vendors within 25 "
    "miles: 4 in-network, 1 currently unavailable due to capacity. "
    "Assignment offered to policyholder; policyholder indicated they will "
    "use their chosen vendor outside the preferred network. Managed-repair "
    "program participation not elected. Vendor work will be subject to "
    "standard post-loss documentation requirements.",
    "Reserves set {date}: initial indemnity reserve established at estimated "
    "exposure pending inspection. Expense reserve established for adjuster "
    "time and any needed third-party experts. Reserve adequacy to be "
    "reviewed at 30/60/90-day intervals per internal policy. Stop-loss "
    "triggers not met. Management notification not required at this reserve "
    "level.",
    "Subrogation screen {date}: no immediately identified third-party "
    "tortfeasor at intake. File flagged for re-screen post-cause "
    "determination. Standard subrogation preservation language added to "
    "all mitigation authorizations. No tender-of-defense letter required.",
    "SIU referral screen {date}: no indicators meeting the automatic "
    "referral threshold. Keyword detection on FNOL narrative returned a "
    "normal-risk score. Prior-claim pattern analysis returned no flags. "
    "Standard adjuster to remain assigned; SIU re-screen scheduled at "
    "coverage-determination milestone.",
    "Notice to mortgagee {date}: loss notice sent to the lender of record "
    "per the standard mortgage clause. No response required at this stage. "
    "Lender identified as the primary loss payee on any structural "
    "settlement. Mortgagee will be named on any settlement draft exceeding "
    "the policy's single-endorsement threshold.",
    "File documentation checklist {date}: 42 photos uploaded to the claim "
    "file with geo-tags, sketch prepared by the adjuster, contents inventory "
    "pending completion. All uploads indexed by the document management "
    "system. File meets the minimum documentation standard for adjuster "
    "handoff and quality-assurance sampling.",
    "Internal audit sampler excerpt: claim file selected for QA review "
    "cycle {cycle}. File to be reviewed post-closure against state "
    "fair-claims-practices standards and internal handle-time metrics. "
    "Outcome of QA review to be logged for the analyst's annual performance "
    "file.",
    "Policyholder Bill of Rights acknowledgement {date}: electronic delivery "
    "confirmed. 60-day rights summary attached. Includes state-specific "
    "timeline expectations for acknowledgement, inspection, decision, and "
    "payment. No policyholder pushback received.",
    "Catastrophe code not applied: event does not meet the ISO Property "
    "Claim Services catastrophe criteria; claim processed under standard "
    "non-CAT workflow. No special catastrophe deductible applies. "
    "Staffing-level impact none.",
    "Coverage summary auto-populated from the policy system of record "
    "{date}: forms HO-3 with amendatory endorsements for state-specific "
    "language. No optional riders in force at time of loss. Policy "
    "declarations page attached to file.",
    "Adjuster time log {date}: on-site inspection 2.0 hours; desk review "
    "1.25 hours; policyholder contact 0.5 hours; report preparation 1.0 "
    "hour. Mileage 34 round-trip. Time entries submitted via mobile app "
    "and approved by supervisor.",
    "State department of insurance complaint check {date}: no open "
    "complaints associated with this claim. No regulatory inquiries to "
    "date. Complaint register clear.",
]


# ---------------------------------------------------------------------------
# Loan case template: produces 5 variants with parameterized ground truth.
# ---------------------------------------------------------------------------

def _loan_case(idx: int, seed: int, decision: str) -> Case:
    """Build one loan case.

    decision ∈ {"APPROVE", "DENY"}. Ground-truth fields computed from seed
    so each case is deterministic and distinct.
    """
    rng = random.Random(seed)

    # ---- Sample parameters ------------------------------------------------
    applicants = [
        ("Jane Doe", "1988-06-15", "421 Elm St, Portland OR"),
        ("Marcus Liu", "1982-11-03", "1702 Chestnut Pl, Seattle WA"),
        ("Priya Patel", "1990-02-27", "88 Jasmine Ct, San Jose CA"),
        ("Daniel Ortiz", "1985-08-19", "56 Ridge Rd, Denver CO"),
        ("Aisha Hassan", "1992-04-12", "903 Harbor Dr, Boston MA"),
    ]
    applicant, dob, addr = applicants[idx % len(applicants)]

    # Income / credit anchors
    if decision == "APPROVE":
        income = 90_000 + rng.randrange(0, 80_000, 500)
        fico = rng.randrange(720, 790)
        dti_target = rng.uniform(28, 40)
    else:
        income = 50_000 + rng.randrange(0, 30_000, 500)
        fico = rng.randrange(580, 640)
        dti_target = rng.uniform(48, 58)

    purchase_price = 400_000 + rng.randrange(0, 300_000, 5_000)
    appraised = purchase_price + rng.randrange(-10_000, 20_000, 1_000)
    loan_amount = int(purchase_price * 0.8)
    ltv = round(loan_amount / appraised * 100, 1)
    rate = round(rng.uniform(5.75, 7.25), 2)
    monthly_income = round(income / 12)
    housing = round(loan_amount * rate / 100 / 12 + 500)
    other_debt = round(monthly_income * (dti_target / 100) - housing)
    if other_debt < 100:
        other_debt = 100
    dti = round((housing + other_debt) / monthly_income * 100, 1)

    # Reasoning anchor: a specific narrative twist.
    narratives = [
        ("employment_gap_medical",
         "documented FMLA medical leave (surgery) from 2024-01 through 2024-06",
         "The 2024 employment gap is explained by documented FMLA medical leave, "
         "not income instability."),
        ("large_gift",
         "a $14,000 wire on 2026-01-18 documented as a gift from the applicant's "
         "parent, with executed gift letter and donor statement",
         "The large January 2026 deposit is a documented gift from a family "
         "member, sourced per Fannie Mae guidelines."),
        ("co_signer",
         "addition of a non-occupant co-signer (sibling) with $180,000 income "
         "and FICO 780",
         "A non-occupant co-signer with strong income and credit is on the "
         "application, improving the effective qualifying ratios."),
        ("self_employed",
         "self-employment income from a 3-year-old LLC with two years of "
         "tax returns averaging $118,000",
         "Self-employment income is supported by two years of filed returns "
         "with consistent schedule C results."),
        ("recent_raise",
         "a 2026-01 promotion to $142,500 from $118,000, confirmed by "
         "employer VOE and pay stubs",
         "A recent promotion is documented by employer VOE and pay stubs "
         "showing stable income going forward."),
    ]
    narrative_key, narrative_evidence, reasoning_point = narratives[idx % len(narratives)]

    # ---- Build chunks -----------------------------------------------------
    chunks: List[Chunk] = []

    chunks.append(Chunk(
        f"Application received 2026-03-{2 + idx:02d}. Applicant: {applicant}, "
        f"DOB {dob}, primary residence: {addr}.", "fact"))

    chunks.append(Chunk(
        f"W-2 tax year 2025: gross annual wages ${income:,} from "
        f"Pacific Logistics Inc.", "fact",
        key="income_2025", value=f"${income:,}"))

    chunks.append(Chunk(
        f"Pay stub dated 2026-02-14: YTD gross ${income // 6:,}; employer "
        f"Pacific Logistics Inc.", "fact"))

    chunks.append(Chunk(
        f"Tax return 2025 filed 2026-02-10. AGI ${int(income * 0.97):,}; "
        f"filing status single.", "fact"))

    chunks.append(Chunk(
        f"Credit report pulled 2026-03-05. FICO 8: {fico}. Total revolving "
        f"utilization: {rng.randrange(8, 32)}%.", "fact",
        key="fico_score", value=str(fico)))

    chunks.append(Chunk(
        f"Property: {rng.randrange(100, 999)} Oak Ave, {addr.split(',')[-1].strip()}. "
        f"Purchase price per contract: ${purchase_price:,}.", "fact",
        key="purchase_price", value=f"${purchase_price:,}"))

    chunks.append(Chunk(
        f"Appraisal dated 2026-03-10 by Alpine Valuation. Appraised value: "
        f"${appraised:,}.", "fact",
        key="appraised_value", value=f"${appraised:,}"))

    chunks.append(Chunk(
        f"Loan amount requested: ${loan_amount:,}. LTV: {ltv}%.", "fact",
        key="loan_amount", value=f"${loan_amount:,}"))

    chunks.append(Chunk(
        f"DTI calculated 2026-03-15: monthly debt ${other_debt:,} + proposed "
        f"housing ${housing:,} = ${other_debt + housing:,}. Monthly income "
        f"${monthly_income:,}. DTI = {dti}%.",
        "mixed", key="dti", value=f"{dti}%"))

    chunks.append(Chunk(
        f"Rate lock executed 2026-03-14: {rate}% fixed, 30-year term, "
        f"45-day lock.", "fact",
        key="locked_rate", value=f"{rate}%"))

    # Narrative reasoning chain (distinct per case)
    chunks.append(Chunk(
        f"Underwriter note 2026-03-06: file flagged for review because "
        f"{narrative_evidence}. Need resolution before decision.", "reasoning"))
    chunks.append(Chunk(
        f"Applicant correspondence dated 2026-03-08: {narrative_evidence}. "
        f"Supporting documentation attached.", "reasoning"))
    chunks.append(Chunk(
        f"Third-party verification 2026-03-09 confirms: {narrative_evidence}.",
        "reasoning"))
    if decision == "APPROVE":
        chunks.append(Chunk(
            f"Underwriter decision note 2026-03-10: the flagged issue is "
            f"resolved — {narrative_evidence} — and does not indicate "
            f"repayment risk. Current income and credit figures support "
            f"approval at the requested loan amount.", "reasoning"))
    else:
        chunks.append(Chunk(
            f"Underwriter decision note 2026-03-10: even allowing "
            f"{narrative_evidence}, the combined DTI of {dti}% and FICO "
            f"of {fico} fall outside policy tolerance. Adverse action "
            f"notice required.", "reasoning"))

    # Pad with filler (each filler chunk is ~200-400 chars, enough to push
    # trajectory to ~5-8k tokens). Chunk type mark: "mixed" — neither strong
    # fact nor strong reasoning signal by design.
    for i in range(68):
        template = LOAN_FILLER[i % len(LOAN_FILLER)]
        text = template.format(
            cycle=i + 1,
            date=f"2026-03-{(i % 28) + 1:02d}",
            team=f"R-{(i % 5) + 1}",
        )
        chunks.append(Chunk(text, "mixed"))

    rng.shuffle(chunks)

    # ---- Ground truth -----------------------------------------------------
    gt_facts = {
        "income_2025": f"${income:,}",
        "fico_score": str(fico),
        "appraised_value": f"${appraised:,}",
        "loan_amount": f"${loan_amount:,}",
        "dti": f"{dti}%",
        "locked_rate": f"{rate}%",
    }
    reasoning_points = [
        reasoning_point,
        f"DTI of {dti}% and FICO {fico} are the primary quantitative "
        f"factors in the decision.",
    ]
    required_provisions = []
    if decision == "DENY":
        required_provisions = ["ECOA adverse action: DTI exceeds policy"]

    gt = GroundTruth(
        decision=decision,
        facts=gt_facts,
        reasoning_points=reasoning_points,
        required_provisions=required_provisions,
        rationale_exemplar=(
            f"{decision}. Applicant {applicant}, requested loan ${loan_amount:,} "
            f"against appraisal ${appraised:,} (LTV {ltv}%). Income ${income:,}, "
            f"FICO {fico}, DTI {dti}%. {reasoning_point}"
        ),
    )

    return Case(
        case_id=f"loan_L{idx + 1:02d}",
        domain="loan",
        documents=chunks,
        ground_truth=gt,
        task_prompt=(
            "You are a mortgage underwriter. Based on the provided memory, "
            "decide whether to APPROVE or DENY this application. Produce "
            "(a) the decision, (b) a rationale memo that cites specific numeric "
            "facts from the file, and (c) if denied, an adverse action notice "
            "citing specific reasons."
        ),
    )


# ---------------------------------------------------------------------------
# Claim case template
# ---------------------------------------------------------------------------

def _claim_case(idx: int, seed: int, decision: str) -> Case:
    """Build one claim case.

    decision ∈ {"PAY", "PARTIAL PAY", "DENY"}.
    """
    rng = random.Random(seed)

    insureds = [
        ("Maria Chen", "1142 Birch Ln, Austin TX"),
        ("Kenji Tanaka", "28 Bayshore Rd, Miami FL"),
        ("Elena Rossi", "671 Maple Ave, Chicago IL"),
        ("Samuel Okonkwo", "415 Pine Crest, Charlotte NC"),
        ("Farida Al-Sayed", "9 Highland Pl, Phoenix AZ"),
    ]
    insured, address = insureds[idx % len(insureds)]
    policy_num = f"POL-{rng.randrange(700000, 999999)}"
    dwelling_limit = 300_000 + rng.randrange(0, 300_000, 10_000)
    deductible = rng.choice([1_000, 2_500, 5_000])
    policy_year = 2025

    # Scenario
    scenarios = [
        ("water_gradual_partial", "PARTIAL PAY",
         "water damage in basement", "Exclusion 7",
         "wear, tear, deterioration, rust, mold, or gradual seepage",
         "moisture mapping establishes gradual cause rather than a single-event leak"),
        ("water_sudden_full", "PAY",
         "burst pipe flood in kitchen", "none",
         "sudden and accidental discharge is a covered peril",
         "plumber invoice confirms a sudden pipe rupture with documented "
         "active leak at time of loss"),
        ("foundation_settling_deny", "DENY",
         "cracks in foundation wall", "Exclusion 9",
         "settling, cracking, shrinking, or expansion of foundations",
         "engineer report attributes cracking to long-term settlement rather "
         "than a sudden peril"),
        ("fire_kitchen_full", "PAY",
         "fire damage in kitchen", "none",
         "fire is a named peril under Coverage A",
         "fire department report documents an accidental grease fire contained "
         "to one room"),
        ("wind_roof_partial", "PARTIAL PAY",
         "wind-damaged roof and resulting interior water", "Exclusion 12",
         "wear and tear to roofing materials older than 20 years is excluded "
         "from replacement-cost treatment",
         "roof inspection documents 22-year-old shingles; wind loss is covered "
         "on an ACV basis with the interior water damage fully covered"),
    ]
    _, forced_decision, loss_desc, exclusion_label, exclusion_text, reasoning_pt = \
        scenarios[idx % len(scenarios)]

    # Force decision to match requested — generator accepts either the
    # scenario's natural decision or overrides it.
    decision = forced_decision

    # Numeric anchors
    if decision == "PAY":
        repair_estimate = 8_000 + rng.randrange(0, 20_000, 500)
        covered = repair_estimate
        net_payment = max(0, covered - deductible)
    elif decision == "PARTIAL PAY":
        repair_estimate = 14_000 + rng.randrange(0, 15_000, 500)
        covered = int(repair_estimate * rng.uniform(0.35, 0.55))
        net_payment = max(0, covered - deductible)
    else:  # DENY
        repair_estimate = 10_000 + rng.randrange(0, 25_000, 500)
        covered = 0
        net_payment = 0

    chunks: List[Chunk] = []

    chunks.append(Chunk(
        f"Policy HO-3 #{policy_num}, insured: {insured}, property: {address}. "
        f"Effective {policy_year}-04-01 to {policy_year + 1}-04-01.", "fact",
        key="policy_number", value=policy_num))

    chunks.append(Chunk(
        f"Coverage A (dwelling): ${dwelling_limit:,}. Coverage B (other "
        f"structures): ${dwelling_limit // 10:,}. Coverage C (personal "
        f"property): ${dwelling_limit // 2:,}. Deductible: ${deductible:,}.",
        "fact", key="dwelling_limit", value=f"${dwelling_limit:,}"))

    chunks.append(Chunk(
        "Coverage A Exclusion 7: 'We do not cover loss caused by wear, tear, "
        "deterioration, rust, mold, or gradual seepage of water over time.'",
        "fact"))

    chunks.append(Chunk(
        "Coverage A Exclusion 9: 'We do not cover loss caused by settling, "
        "cracking, shrinking, or expansion of foundations.'", "fact"))

    chunks.append(Chunk(
        "Coverage A Exclusion 12: 'Roofing materials older than 20 years "
        "are covered on an actual cash value basis only; full replacement "
        "cost is excluded.'", "fact"))

    chunks.append(Chunk(
        f"First Notice of Loss 2026-02-07: insured reports '{loss_desc} "
        f"discovered 2026-02-06.'",
        "mixed", key="date_of_loss_reported", value="2026-02-06"))

    chunks.append(Chunk(
        f"Inspection by adjuster 2026-02-09. Scope of observed damage "
        f"consistent with reported loss ({loss_desc}).", "mixed"))

    chunks.append(Chunk(
        f"Repair estimate dated 2026-02-14 from CertifiedDry: ${repair_estimate:,} "
        f"total.", "fact", key="repair_estimate", value=f"${repair_estimate:,}"))

    # Scenario-specific technical document (drives causation)
    if "water_gradual" in scenarios[idx % len(scenarios)][0]:
        chunks.append(Chunk(
            "Plumber invoice 2026-02-10 from Austin Plumbing Pro: 'No active "
            "leak found. Evidence of long-term moisture ingress through wall "
            "at grade. No burst pipe.'", "reasoning"))
        chunks.append(Chunk(
            "Moisture mapping report 2026-02-15: wall moisture 22-31%, "
            "consistent with prolonged exposure over weeks to months, "
            "not a single-event leak.", "reasoning"))
    elif "water_sudden" in scenarios[idx % len(scenarios)][0]:
        chunks.append(Chunk(
            "Plumber invoice 2026-02-07 from Miami Plumbing Co: 'Active "
            "leak found at kitchen supply line; 1/2\" copper pipe ruptured. "
            "Shut-off valve seized. Emergency repair performed.'", "reasoning"))
    elif "foundation" in scenarios[idx % len(scenarios)][0]:
        chunks.append(Chunk(
            "Structural engineer report 2026-02-15: 'Cracking is consistent "
            "with differential foundation settlement over multiple years. "
            "No single-event cause observed.'", "reasoning"))
    elif "fire" in scenarios[idx % len(scenarios)][0]:
        chunks.append(Chunk(
            "Fire department report #2026-0412: 'Accidental grease fire "
            "originating at cooktop; contained to kitchen. Cause accidental, "
            "no indication of intentional origin.'", "reasoning"))
    elif "wind" in scenarios[idx % len(scenarios)][0]:
        chunks.append(Chunk(
            "Roof inspection 2026-02-12: 'Shingles dated 2004 installation. "
            "22 years of service. Wind event lifted south-face shingles, "
            "allowing interior water entry.'", "reasoning"))

    # Allocation reasoning
    chunks.append(Chunk(
        f"Adjuster note 2026-02-17: causation analysis per above technical "
        f"report — {reasoning_pt}.", "reasoning"))

    if decision != "PAY":
        chunks.append(Chunk(
            f"Adjuster note 2026-02-18: portions of the loss fall under "
            f"{exclusion_label} ({exclusion_text}); coverage is limited "
            f"accordingly.", "reasoning"))
    else:
        chunks.append(Chunk(
            f"Adjuster note 2026-02-18: loss is a covered peril under "
            f"Coverage A; no applicable exclusions. Proceeding to payment.",
            "reasoning"))

    chunks.append(Chunk(
        f"Claimant correspondence 2026-02-19: {insured} requests clarification "
        f"on coverage decision.", "reasoning"))

    chunks.append(Chunk(
        f"Adjuster response 2026-02-20: coverage turns on cause, not on "
        f"insured awareness. {reasoning_pt.capitalize()}.", "reasoning"))

    # Final numeric allocation
    if decision == "PAY":
        chunks.append(Chunk(
            f"Final allocation 2026-02-22: full repair ${repair_estimate:,} "
            f"paid under Coverage A. Deductible ${deductible:,} applied. "
            f"Net payment: ${net_payment:,}.", "fact",
            key="net_payment", value=f"${net_payment:,}"))
    elif decision == "PARTIAL PAY":
        chunks.append(Chunk(
            f"Final allocation 2026-02-22: covered portion ${covered:,} paid "
            f"under Coverage A. Excluded portion ${repair_estimate - covered:,} "
            f"denied under {exclusion_label}. Deductible ${deductible:,} "
            f"applied. Net payment: ${net_payment:,}.", "fact",
            key="net_payment", value=f"${net_payment:,}"))
    else:  # DENY
        chunks.append(Chunk(
            f"Final allocation 2026-02-22: loss denied under {exclusion_label} "
            f"({exclusion_text}). Net payment: $0.", "fact",
            key="net_payment", value="$0"))

    if decision != "PAY":
        chunks.append(Chunk(
            f"Denial reason citation: '{exclusion_label}: {exclusion_text}.'",
            "fact", key="denial_provision", value=exclusion_label))

    # Filler
    for i in range(80):
        template = CLAIM_FILLER[i % len(CLAIM_FILLER)]
        text = template.format(
            date=f"2026-02-{(i % 27) + 1:02d}",
            cycle=i + 1,
        )
        chunks.append(Chunk(text, "mixed"))

    rng.shuffle(chunks)

    # Ground truth
    gt_facts = {
        "policy_number": policy_num,
        "dwelling_limit": f"${dwelling_limit:,}",
        "repair_estimate": f"${repair_estimate:,}",
        "net_payment": f"${net_payment:,}" if decision != "DENY" else "$0",
    }
    if decision != "PAY":
        gt_facts["denial_provision"] = exclusion_label

    reasoning_points = [
        reasoning_pt.capitalize() + ".",
    ]
    if decision == "PARTIAL PAY":
        reasoning_points.append(
            f"The excluded portion is denied under {exclusion_label} "
            f"({exclusion_text})."
        )
        reasoning_points.append(
            f"The covered portion ${covered:,} is paid under Coverage A "
            f"less the ${deductible:,} deductible."
        )
    elif decision == "DENY":
        reasoning_points.append(
            f"The loss is denied in full under {exclusion_label}."
        )
    else:
        reasoning_points.append(
            "The loss is a covered peril and no exclusions apply."
        )

    required_provisions = [] if decision == "PAY" else [exclusion_label]

    gt = GroundTruth(
        decision=decision,
        facts=gt_facts,
        reasoning_points=reasoning_points,
        required_provisions=required_provisions,
        rationale_exemplar=(
            f"{decision}. Claim {policy_num} at {address}. "
            f"Repair estimate ${repair_estimate:,}; net payment "
            f"${net_payment:,}. {reasoning_pt.capitalize()}."
        ),
    )

    return Case(
        case_id=f"claim_C{idx + 1:02d}",
        domain="claim",
        documents=chunks,
        ground_truth=gt,
        task_prompt=(
            "You are an insurance claim adjudicator. Based on the provided "
            "memory, decide whether to PAY in full, PARTIAL PAY, or DENY "
            "this claim. Produce (a) the decision, (b) a rationale memo "
            "citing specific policy provisions and dollar amounts, and "
            "(c) if denying or partial-paying, a denial/partial notice "
            "citing the specific provisions applied."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level case lists
# ---------------------------------------------------------------------------

def build_large_cases() -> List[Case]:
    loan_plan = [
        (0, 11, "APPROVE"),
        (1, 22, "APPROVE"),
        (2, 33, "DENY"),
        (3, 44, "APPROVE"),
        (4, 55, "DENY"),
    ]
    claim_plan = [
        (0, 101, "PARTIAL PAY"),  # forced by scenario
        (1, 202, "PAY"),
        (2, 303, "DENY"),
        (3, 404, "PAY"),
        (4, 505, "PARTIAL PAY"),
    ]
    return (
        [_loan_case(i, seed, d) for (i, seed, d) in loan_plan]
        + [_claim_case(i, seed, d) for (i, seed, d) in claim_plan]
    )


LARGE_CASES: List[Case] = build_large_cases()


if __name__ == "__main__":
    import json
    for c in LARGE_CASES:
        total_chars = sum(len(d.text) for d in c.documents)
        print(f"{c.case_id}: {len(c.documents)} chunks, ~{total_chars} chars, "
              f"~{total_chars // 4} tokens; decision={c.ground_truth.decision}")
    print(f"\nTotal: {len(LARGE_CASES)} cases.")
    print(f"Avg chars: {sum(sum(len(d.text) for d in c.documents) for c in LARGE_CASES) // len(LARGE_CASES)}")
