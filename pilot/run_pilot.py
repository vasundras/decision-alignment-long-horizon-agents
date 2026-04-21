"""Stage 1 smoke test runner.

Runs 2 cases x 4 conditions = 8 agent trajectories. For each, computes
FRP, RCS, EDA, CRR. Prints a results table and the approximate API cost.

Intentionally small: verifies the pipeline produces non-degenerate signals
across conditions, not that TMC beats baselines at scale.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from anthropic import Anthropic

from cases import ALL_CASES, Case
from conditions import (
    CONDITION_CLASSES, Condition, LLMBackend, approx_tokens,
)
from metrics import (
    AgentOutput, fact_recall_precision, reasoning_coherence_score,
    end_to_end_decision_accuracy, compliance_reconstruction_rate,
)

# Load .env from project root (one directory up from pilot/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AGENT_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-haiku-4-5-20251001"   # same for cheap Stage 1; swap to Sonnet for rigor

# Cost tracking (Haiku 4.5 approx pricing; adjust if rates differ)
INPUT_COST_PER_MTOK = 1.0
OUTPUT_COST_PER_MTOK = 5.0


class AnthropicBackend:
    """Implements LLMBackend protocol against Anthropic SDK."""
    def __init__(self, model: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        # Concatenate text blocks
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_COST_PER_MTOK / 1_000_000
            + self.output_tokens * OUTPUT_COST_PER_MTOK / 1_000_000
        )


DECISION_SYSTEM = (
    "You are an enterprise decision agent. You have a memory layer that has been "
    "consolidated from a long trajectory of documents. Using ONLY the memory "
    "provided, produce a decision for the task. Your response must be a single "
    "JSON object with three fields: 'decision' (string), 'rationale_memo' (string), "
    "'notice' (string, the adverse action or denial notice if applicable; empty "
    "string otherwise). Cite specific dollar amounts, dates, and provisions from "
    "the memory verbatim. Do not include any prose before or after the JSON."
)


def run_condition(
    agent_backend: AnthropicBackend,
    condition: Condition,
    case: Case,
) -> AgentOutput:
    """Ingest all chunks into the memory, then query for the decision."""
    for chunk in case.documents:
        condition.consolidate(chunk)
    memory_surface = condition.present_at_decision(case.task_prompt)
    user = (
        f"Task:\n{case.task_prompt}\n\n"
        f"Memory:\n{memory_surface}\n\n"
        f"Return the JSON object now."
    )
    raw = agent_backend.complete(DECISION_SYSTEM, user, max_tokens=1200).strip()
    return _parse_agent_output(raw)


def _parse_agent_output(raw: str) -> AgentOutput:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return AgentOutput(decision="PARSE_ERROR", rationale_memo=raw[:500], denial_notice="")
    try:
        d = json.loads(m.group(0))
    except Exception:
        return AgentOutput(decision="PARSE_ERROR", rationale_memo=raw[:500], denial_notice="")
    return AgentOutput(
        decision=str(d.get("decision", "")),
        rationale_memo=str(d.get("rationale_memo", "")),
        denial_notice=str(d.get("notice", "")),
    )


@dataclass
class RunResult:
    case_id: str
    condition: str
    decision: str
    FRP: float
    RCS: float
    EDA: float
    CRR: float
    rationale_preview: str = ""


def run_pilot(budget_chars: int = 800) -> List[RunResult]:
    agent = AnthropicBackend(AGENT_MODEL)
    judge = AnthropicBackend(JUDGE_MODEL)

    results: List[RunResult] = []
    for case in ALL_CASES:
        print(f"\n=== Case: {case.case_id} ({case.domain}) ===")
        for cond_cls in CONDITION_CLASSES:
            cond = cond_cls(agent, budget_chars=budget_chars)
            t0 = time.time()
            output = run_condition(agent, cond, case)
            elapsed = time.time() - t0
            frp = fact_recall_precision(output, case.ground_truth.facts)
            rcs = reasoning_coherence_score(judge, output, case.ground_truth.reasoning_points)
            eda = end_to_end_decision_accuracy(output, case.ground_truth.decision)
            crr = compliance_reconstruction_rate(judge, output, case)
            r = RunResult(
                case_id=case.case_id,
                condition=cond.name,
                decision=output.decision,
                FRP=frp, RCS=rcs, EDA=eda, CRR=crr,
                rationale_preview=output.rationale_memo[:180],
            )
            results.append(r)
            print(f"  [{cond.name:>10}] {elapsed:5.1f}s | "
                  f"dec='{output.decision[:22]:<22}' | "
                  f"FRP={frp:.2f} RCS={rcs:.2f} EDA={eda:.2f} CRR={crr:.2f}")

    print(f"\n=== Cost ===")
    print(f"Agent: in={agent.input_tokens} out={agent.output_tokens} cost=${agent.cost_usd():.4f}")
    print(f"Judge: in={judge.input_tokens} out={judge.output_tokens} cost=${judge.cost_usd():.4f}")
    print(f"TOTAL: ${agent.cost_usd() + judge.cost_usd():.4f}")

    return results


def write_results(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults written to {path}")


def summarize(results: List[RunResult]) -> None:
    from collections import defaultdict
    by_cond = defaultdict(lambda: {"FRP": [], "RCS": [], "EDA": [], "CRR": []})
    for r in results:
        for m in ("FRP", "RCS", "EDA", "CRR"):
            by_cond[r.condition][m].append(getattr(r, m))
    print("\n=== Summary (mean across cases) ===")
    print(f"{'Condition':<12} {'FRP':>6} {'RCS':>6} {'EDA':>6} {'CRR':>6}")
    for cond in ["TMC", "Summ-only", "Retr-only", "Misrouted"]:
        d = by_cond[cond]
        def avg(xs): return sum(xs) / len(xs) if xs else 0.0
        print(f"{cond:<12} {avg(d['FRP']):>6.2f} {avg(d['RCS']):>6.2f} "
              f"{avg(d['EDA']):>6.2f} {avg(d['CRR']):>6.2f}")


def smoke_check(results: List[RunResult]) -> bool:
    """Smoke-test guard: do conditions produce numerically distinct signals?"""
    from collections import defaultdict
    by_cond_metric = defaultdict(list)
    for r in results:
        for m in ("FRP", "RCS", "EDA", "CRR"):
            by_cond_metric[(r.condition, m)].append(getattr(r, m))

    # Signal 1: across conditions, at least one metric should vary by >=0.1
    max_spread = 0.0
    for m in ("FRP", "RCS", "EDA", "CRR"):
        means = [sum(by_cond_metric[(c, m)]) / max(1, len(by_cond_metric[(c, m)]))
                 for c in ["TMC", "Summ-only", "Retr-only", "Misrouted"]]
        spread = max(means) - min(means)
        max_spread = max(max_spread, spread)
        print(f"  spread on {m}: {spread:.2f}")

    # Signal 2: no condition returned PARSE_ERROR for any case
    parse_errors = [r for r in results if r.decision == "PARSE_ERROR"]
    if parse_errors:
        print(f"  WARNING: {len(parse_errors)} parse errors")

    print(f"\n  Max metric spread across conditions: {max_spread:.2f}")
    if max_spread >= 0.1:
        print("  PASS: pipeline produces distinct signals across conditions.")
        return True
    else:
        print("  INVESTIGATE: conditions produce near-identical outputs. Check consolidation logic.")
        return False


if __name__ == "__main__":
    try:
        results = run_pilot(budget_chars=800)
        write_results(results, PROJECT_ROOT / "pilot" / "results" / "stage1.json")
        summarize(results)
        print()
        smoke_check(results)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
