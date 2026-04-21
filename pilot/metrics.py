"""Four decomposed metrics.

FRP — Fact Recall Precision: exact-match on ground-truth fact values.
RCS — Reasoning Coherence Score: LLM-judge entailment of ground-truth
      reasoning points from the agent's rationale.
EDA — End-to-end Decision Accuracy: exact match of decision label.
CRR — Compliance Reconstruction Rate: LLM-judge verdict on whether the
      agent's output would meet a compliance-audit standard (for denial/
      partial-pay cases).

All LLM-judge calls use a separate, higher-capacity model than the agent
backend to reduce judge-model bias.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

from cases import Case
from conditions import LLMBackend


@dataclass
class AgentOutput:
    decision: str
    rationale_memo: str
    denial_notice: str = ""


# -----------------------------------------------------------------------------
# FRP — exact-match on ground-truth facts
# -----------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def fact_recall_precision(agent: AgentOutput, gt_facts: Dict[str, str]) -> float:
    """Return fraction of ground-truth facts whose value string appears verbatim in agent output."""
    if not gt_facts:
        return 1.0
    combined = agent.rationale_memo + "\n" + agent.denial_notice
    text = _normalize(combined)
    hits = 0
    for key, value in gt_facts.items():
        v_norm = _normalize(value)
        if v_norm in text:
            hits += 1
    return hits / len(gt_facts)


# -----------------------------------------------------------------------------
# RCS — LLM judge for entailment
# -----------------------------------------------------------------------------

RCS_SYSTEM = (
    "You are an entailment judge. You will be given an agent's rationale memo "
    "and a list of reasoning points. For each reasoning point, decide whether "
    "the rationale entails (supports and is consistent with) that point. "
    "Return a JSON object with one key per point index, mapping to true or false. "
    "Do not include any prose before or after the JSON."
)


def reasoning_coherence_score(
    judge: LLMBackend,
    agent: AgentOutput,
    reasoning_points: List[str],
) -> float:
    if not reasoning_points:
        return 1.0
    points_str = "\n".join(f"{i}: {p}" for i, p in enumerate(reasoning_points))
    user = (
        f"Agent rationale memo:\n---\n{agent.rationale_memo}\n---\n"
        f"Denial/partial notice (if any):\n---\n{agent.denial_notice}\n---\n"
        f"Reasoning points to judge:\n{points_str}\n\n"
        f'Return JSON like {{"0": true, "1": false, ...}} with exactly '
        f"{len(reasoning_points)} keys."
    )
    raw = judge.complete(RCS_SYSTEM, user, max_tokens=400).strip()
    # Extract JSON object from the response robustly
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return 0.0
    import json
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return 0.0
    hits = sum(1 for i in range(len(reasoning_points)) if parsed.get(str(i)) is True)
    return hits / len(reasoning_points)


# -----------------------------------------------------------------------------
# EDA — decision label match
# -----------------------------------------------------------------------------

def end_to_end_decision_accuracy(agent: AgentOutput, gt_decision: str) -> float:
    agent_norm = _normalize(agent.decision)
    gt_norm = _normalize(gt_decision)
    # Loose match: accept exact substring containment
    return 1.0 if gt_norm in agent_norm or agent_norm in gt_norm else 0.0


# -----------------------------------------------------------------------------
# CRR — LLM compliance auditor (denial cases only)
# -----------------------------------------------------------------------------

CRR_SYSTEM = (
    "You are a compliance auditor. Given an agent's decision output (decision + "
    "rationale + denial/partial notice), judge whether the output would meet "
    "a regulatory compliance standard for an adverse action notice (loans, "
    "ECOA/Reg B) or a denial notice (insurance claims, state regs). A compliant "
    "output must cite specific factual anchors (dollar amounts, dates, policy "
    "provisions) and specific reasons tied to those anchors. Approvals without "
    "any denial require no notice and should be judged COMPLIANT by default. "
    'Return only the token COMPLIANT or NONCOMPLIANT.'
)


# -----------------------------------------------------------------------------
# CAR — Calibrated Abstention Rate (Stage 5 metric)
# -----------------------------------------------------------------------------
#
# CAR is a new metric axis that rewards agents for abstaining on cases where
# they would otherwise guess wrong. An architecture that always commits (like
# Summ-only, TMC, SAM, DPM) has commit_rate = 1.0 and conditional_accuracy =
# unconditional_accuracy.
#
# An architecture with a completeness check (VM) may have commit_rate < 1.0,
# with conditional_accuracy potentially > commit_all_accuracy if the abstentions
# concentrate on ambiguous cases where guessing would have been wrong.
#
# The practitioner-relevant scalar: conditional accuracy - commit-all accuracy
# (positive = well-calibrated, negative = over-abstaining).

def car_components(decisions: List[str], gts: List[str]) -> Dict[str, float]:
    """Given paired lists of agent decisions and ground-truth decisions,
    compute commit rate, conditional accuracy (accuracy on committed cases),
    and commit-all accuracy (treating abstentions as zero)."""
    n = len(decisions)
    if n == 0:
        return {"n": 0, "commit_rate": 0.0, "conditional_accuracy": 0.0,
                "commit_all_accuracy": 0.0, "abstain_count": 0}
    committed = [(d, g) for d, g in zip(decisions, gts) if _normalize(d) != "abstain"]
    n_c = len(committed)
    abstain_count = n - n_c
    commit_rate = n_c / n
    cond_acc = (sum(1 for d, g in committed if _normalize(g) in _normalize(d) or _normalize(d) in _normalize(g)) / n_c) if n_c else 0.0
    commit_all_acc = (sum(1 for d, g in zip(decisions, gts)
                          if _normalize(g) in _normalize(d) or _normalize(d) in _normalize(g)) / n)
    return {
        "n": n,
        "commit_rate": commit_rate,
        "conditional_accuracy": cond_acc,
        "commit_all_accuracy": commit_all_acc,
        "abstain_count": abstain_count,
    }


def compliance_reconstruction_rate(
    judge: LLMBackend,
    agent: AgentOutput,
    case: Case,
) -> float:
    # For approve cases, compliance is trivially satisfied (no notice required).
    if "approve" in _normalize(case.ground_truth.decision) and "pay" not in _normalize(case.ground_truth.decision):
        return 1.0 if "approve" in _normalize(agent.decision) else 0.0
    # For deny / partial pay, require the auditor judgment.
    provisions = ", ".join(case.ground_truth.required_provisions) or "(none required)"
    user = (
        f"Case domain: {case.domain}\n"
        f"Agent decision: {agent.decision}\n"
        f"Agent rationale:\n{agent.rationale_memo}\n"
        f"Agent notice:\n{agent.denial_notice}\n"
        f"Required provisions to cite: {provisions}\n"
        "Is this output compliant? Return only COMPLIANT or NONCOMPLIANT."
    )
    raw = judge.complete(CRR_SYSTEM, user, max_tokens=20).strip().upper()
    return 1.0 if "COMPLIANT" in raw and "NON" not in raw else 0.0
