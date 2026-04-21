"""Stage 2 overflow-regime experiment.

Runs 5 conditions × 3 budget ratios × 10 large cases = 150 units, each
producing per-case FRP/RCS/EDA/CRR. Includes:

  - Thread-based concurrency across (case, condition) within a budget.
  - Hard cost cap at $15 (abort w/ error).
  - Cost projection check after each budget tier.
  - Results written to pilot/stage2/results.json.

Permutation tests and figures are produced by analyze_stage2.py.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

from cases_large import LARGE_CASES
from cases import Case
from conditions import (
    CONDITION_CLASSES_STAGE2, Condition, LLMBackend,
)
from metrics import (
    AgentOutput, fact_recall_precision, reasoning_coherence_score,
    end_to_end_decision_accuracy, compliance_reconstruction_rate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

AGENT_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

INPUT_COST_PER_MTOK = 1.0
OUTPUT_COST_PER_MTOK = 5.0
HARD_COST_CAP_USD = 15.0

STAGE2_DIR = PROJECT_ROOT / "pilot" / "stage2"


class AnthropicBackend:
    """Thread-safe-ish LLMBackend with cumulative token/cost tracking."""

    def __init__(self, model: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        # Abort early if we're already over budget (cheap pre-check).
        # Sum across ALL distinct backends (agent + judge) exactly once.
        total = sum(b.cost_usd() for b in _BACKENDS.values())
        if total > HARD_COST_CAP_USD:
            raise RuntimeError(
                f"Hard cost cap ${HARD_COST_CAP_USD:.2f} reached; aborting."
            )
        # Retry on transient errors (rate limits, overloads). Anthropic returns
        # 429 / 529; SDK raises APIStatusError subclasses.
        last_exc = None
        for attempt in range(5):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                with self._lock:
                    self.input_tokens += resp.usage.input_tokens
                    self.output_tokens += resp.usage.output_tokens
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as e:
                msg = str(e).lower()
                last_exc = e
                retryable = any(k in msg for k in (
                    "rate_limit", "overloaded", "429", "529",
                    "timeout", "connection", "apitimeout", "apiconnection",
                ))
                if not retryable or attempt == 4:
                    raise
                # Exponential backoff with jitter
                import random
                time.sleep(1.5 * (2 ** attempt) + random.uniform(0, 1))
        raise last_exc  # unreachable

    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_COST_PER_MTOK / 1_000_000
            + self.output_tokens * OUTPUT_COST_PER_MTOK / 1_000_000
        )


_BACKENDS: Dict[str, AnthropicBackend] = {}


DECISION_SYSTEM = (
    "You are an enterprise decision agent. You have a memory layer that has been "
    "consolidated from a long trajectory of documents. Using ONLY the memory "
    "provided, produce a decision for the task. Your response must be a single "
    "JSON object with three fields: 'decision' (string), 'rationale_memo' (string), "
    "'notice' (string, the adverse action or denial notice if applicable; empty "
    "string otherwise). Cite specific dollar amounts, dates, and provisions from "
    "the memory verbatim. Do not include any prose before or after the JSON."
)


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


def run_condition(
    agent_backend: AnthropicBackend,
    condition: Condition,
    case: Case,
) -> AgentOutput:
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


@dataclass
class RunResult:
    case_id: str
    domain: str
    condition: str
    budget_label: str
    budget_chars: int
    trajectory_chars: int
    decision: str
    FRP: float
    RCS: float
    EDA: float
    CRR: float
    wall_time_s: float
    rationale_preview: str = ""
    aborted: bool = False
    error: str = ""


def run_one(
    agent: AnthropicBackend,
    judge: AnthropicBackend,
    cond_cls,
    case: Case,
    budget_chars: int,
    budget_label: str,
) -> RunResult:
    cond = cond_cls(agent, budget_chars=budget_chars)
    t0 = time.time()
    trajectory_chars = sum(len(d.text) for d in case.documents)
    try:
        output = run_condition(agent, cond, case)
        frp = fact_recall_precision(output, case.ground_truth.facts)
        rcs = reasoning_coherence_score(judge, output, case.ground_truth.reasoning_points)
        eda = end_to_end_decision_accuracy(output, case.ground_truth.decision)
        crr = compliance_reconstruction_rate(judge, output, case)
        return RunResult(
            case_id=case.case_id,
            domain=case.domain,
            condition=cond.name,
            budget_label=budget_label,
            budget_chars=budget_chars,
            trajectory_chars=trajectory_chars,
            decision=output.decision,
            FRP=frp, RCS=rcs, EDA=eda, CRR=crr,
            wall_time_s=round(time.time() - t0, 2),
            rationale_preview=output.rationale_memo[:220],
        )
    except Exception as e:
        return RunResult(
            case_id=case.case_id,
            domain=case.domain,
            condition=cond_cls.name,
            budget_label=budget_label,
            budget_chars=budget_chars,
            trajectory_chars=trajectory_chars,
            decision="ERROR",
            FRP=0.0, RCS=0.0, EDA=0.0, CRR=0.0,
            wall_time_s=round(time.time() - t0, 2),
            aborted="cost cap" in str(e).lower(),
            error=str(e)[:300],
        )


def run_stage2(concurrency: int = 6, cases: List[Case] = None) -> Tuple[List[RunResult], Dict]:
    cases = cases if cases is not None else LARGE_CASES
    agent = AnthropicBackend(AGENT_MODEL)
    judge = AnthropicBackend(JUDGE_MODEL)
    _BACKENDS["agent"] = agent
    _BACKENDS["judge"] = judge

    avg_traj_chars = sum(sum(len(d.text) for d in c.documents) for c in cases) // len(cases)
    avg_traj_tokens = avg_traj_chars // 4
    budgets = [
        ("loose",    int(0.5  * avg_traj_chars)),
        ("moderate", int(0.2  * avg_traj_chars)),
        ("tight",    int(0.05 * avg_traj_chars)),
    ]
    print(f"[stage2] avg trajectory: {avg_traj_chars} chars (~{avg_traj_tokens} tokens)")
    print(f"[stage2] budgets (chars): {dict(budgets)}")
    print(f"[stage2] {len(cases)} cases × {len(CONDITION_CLASSES_STAGE2)} conds × {len(budgets)} budgets "
          f"= {len(cases) * len(CONDITION_CLASSES_STAGE2) * len(budgets)} runs")
    print(f"[stage2] hard cost cap: ${HARD_COST_CAP_USD:.2f}\n")

    all_results: List[RunResult] = []
    start_t = time.time()
    aborted_flag = False

    for budget_label, budget_chars in budgets:
        if aborted_flag:
            break
        cost_before = agent.cost_usd() + judge.cost_usd()
        print(f"\n=== Budget: {budget_label} (B = {budget_chars} chars) ===")

        # Build the 50 work units for this budget
        tasks = []
        for case in cases:
            for cond_cls in CONDITION_CLASSES_STAGE2:
                tasks.append((cond_cls, case, budget_chars, budget_label))

        # Submit with bounded concurrency
        budget_results: List[RunResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(run_one, agent, judge, ccls, case, bc, bl): (ccls.name, case.case_id)
                for (ccls, case, bc, bl) in tasks
            }
            for fut in as_completed(futs):
                name, cid = futs[fut]
                r = fut.result()
                budget_results.append(r)
                all_results.append(r)
                cur_cost = agent.cost_usd() + judge.cost_usd()
                flag = "!" if r.aborted else ("E" if r.error else " ")
                err_suffix = f"  err='{r.error[:60]}'" if r.error else ""
                print(f"  {flag}[{r.condition:>9}] {cid:<10} "
                      f"dec='{r.decision[:18]:<18}' "
                      f"FRP={r.FRP:.2f} RCS={r.RCS:.2f} EDA={r.EDA:.2f} CRR={r.CRR:.2f} "
                      f"| cum=${cur_cost:.3f}{err_suffix}")
                if r.aborted or cur_cost > HARD_COST_CAP_USD:
                    aborted_flag = True

        cost_after = agent.cost_usd() + judge.cost_usd()
        delta = cost_after - cost_before
        print(f"\n[stage2] {budget_label} cost: ${delta:.3f}  cumulative: ${cost_after:.3f}")

        # Project remaining
        remaining_budgets = len(budgets) - (budgets.index((budget_label, budget_chars)) + 1)
        if remaining_budgets > 0 and not aborted_flag:
            projected = cost_after + remaining_budgets * delta * 1.5   # 1.5x safety
            print(f"[stage2] projected at worst (×1.5): ${projected:.3f} "
                  f"(remaining budgets: {remaining_budgets})")
            if projected > HARD_COST_CAP_USD:
                print(f"[stage2] ABORT: projection exceeds cap ${HARD_COST_CAP_USD:.2f}")
                aborted_flag = True

    wall = time.time() - start_t

    meta = {
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_cases": len(cases),
        "conditions": [c.name for c in CONDITION_CLASSES_STAGE2],
        "budgets": dict(budgets),
        "avg_trajectory_chars": avg_traj_chars,
        "avg_trajectory_tokens": avg_traj_tokens,
        "agent_input_tokens": agent.input_tokens,
        "agent_output_tokens": agent.output_tokens,
        "judge_input_tokens": judge.input_tokens,
        "judge_output_tokens": judge.output_tokens,
        "agent_cost_usd": round(agent.cost_usd(), 4),
        "judge_cost_usd": round(judge.cost_usd(), 4),
        "total_cost_usd": round(agent.cost_usd() + judge.cost_usd(), 4),
        "wall_time_s": round(wall, 1),
        "aborted_early": aborted_flag,
    }
    return all_results, meta


def summarize_by_cond(results: List[RunResult]) -> None:
    from collections import defaultdict
    print("\n=== Summary by budget × condition (mean across cases) ===")
    print(f"{'Budget':<10} {'Cond':<10} {'FRP':>6} {'RCS':>6} {'EDA':>6} {'CRR':>6} n")
    agg = defaultdict(lambda: {"FRP": [], "RCS": [], "EDA": [], "CRR": []})
    for r in results:
        for m in ("FRP", "RCS", "EDA", "CRR"):
            agg[(r.budget_label, r.condition)][m].append(getattr(r, m))
    for budget in ("loose", "moderate", "tight"):
        for cond in ("TMC", "TMC-full", "Summ-only", "Retr-only", "Misrouted"):
            d = agg[(budget, cond)]
            if not d["FRP"]:
                continue
            def avg(xs): return sum(xs) / len(xs)
            print(f"{budget:<10} {cond:<10} "
                  f"{avg(d['FRP']):>6.2f} {avg(d['RCS']):>6.2f} "
                  f"{avg(d['EDA']):>6.2f} {avg(d['CRR']):>6.2f} "
                  f"{len(d['FRP']):>2}")


def main():
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    try:
        results, meta = run_stage2(concurrency=3)
    except Exception as e:
        print(f"\nTOP-LEVEL ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    out = {
        "meta": meta,
        "results": [asdict(r) for r in results],
    }
    out_path = STAGE2_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[stage2] wrote {out_path}")

    summarize_by_cond(results)
    print(f"\n[stage2] cost ${meta['total_cost_usd']:.3f}  "
          f"wall {meta['wall_time_s']:.1f}s  "
          f"aborted_early={meta['aborted_early']}")


if __name__ == "__main__":
    main()
