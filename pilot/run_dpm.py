"""Stage 4: Deterministic Projection Memory (DPM) evaluation.

DPM treats memory as a pure function of (event_log, task_spec, budget) computed
at decision time, rather than as a process over time. No consolidation during
the trajectory; events accumulate in an append-only log; a single task-
conditioned LLM call projects the full log into a budget-bounded memory view
at decision time.

Why this Stage exists: Stage 2 and Stage 3 established that stateful memory
architectures (TMC, TMC-full, Retr-only, SAM) do not beat plain summarization
at this scale. The search for an *architectural* improvement over Summ-only
has been unsuccessful. Stage 4 tests a *paradigmatic* alternative motivated by
enterprise requirements (deterministic replay, audit, multi-tenant safety)
that no stateful architecture satisfies.

Tiered protocol with a STRICT pre-registered kill-gate:

  Sprint (target: ~4 min, ~$0.30-0.60)
    DPM x moderate budget x 5 cases, 5-way parallel, temperature=0.
    STRICT gate: DPM must beat Summ-only by >= 5 pp on AT LEAST TWO of
    {EDA, RCS, CRR}. If not, stop immediately; paper does not change.

  Full (target: ~15 min, ~$2-3)
    DPM x 3 budgets x 10 cases, 10-way parallel, temperature=0.
    Paired permutation tests against every Stage 2 baseline.

Hard cost cap: $5 total. If gate fails, do not override.

COMMITMENT: If Stage 4 fails the gate, no further architectures will be
proposed. The paper ships with the enterprise-gap framing as the contribution.

Usage:
    cd pilot/
    python run_dpm.py             # runs sprint; stops at gate if it fails
    python run_dpm.py --sprint-only
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

from cases import ALL_CASES, Case
from cases_large import LARGE_CASES
from conditions import DPM, Condition
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
HARD_COST_CAP_USD = 5.0


# -----------------------------------------------------------------------------
# Backend with temperature control. DPM's projection call must be deterministic
# (temperature=0) so that the same (event_log, task_spec, budget) produces the
# same memory view on replay. The Anthropic API still has some non-determinism
# even at temp=0 due to floating-point/batching effects; we accept that residual
# as bounded and small relative to the variance that stateful compression
# introduces.
# -----------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, model: str, temperature: float = 0.0):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        last_exc = None
        for attempt in range(5):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
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
                sleep_s = min(30.0, 3.0 * (2 ** attempt)) + random.uniform(0, 2.0)
                time.sleep(sleep_s)
        raise last_exc

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


def run_one_case(case: Case, budget_label: str, budget_chars: int, seed: int = 0) -> Dict:
    random.seed(seed)
    # Projection (DPM consolidation + decision-time view) uses temp=0 for
    # determinism. The downstream decision agent also uses temp=0 so that the
    # full pipeline is deterministic on replay.
    agent = AnthropicBackend(AGENT_MODEL, temperature=0.0)
    judge = AnthropicBackend(JUDGE_MODEL, temperature=0.0)
    start = time.time()

    cond = DPM(agent, budget_chars=budget_chars)
    trajectory_chars = sum(len(c.text) for c in case.documents)

    for chunk in case.documents:
        cond.consolidate(chunk)
    memory_surface = cond.present_at_decision(case.task_prompt)

    user = (
        f"Task:\n{case.task_prompt}\n\n"
        f"Memory:\n{memory_surface}\n\n"
        f"Return the JSON object now."
    )
    raw = agent.complete(DECISION_SYSTEM, user, max_tokens=1200).strip()
    output = _parse_agent_output(raw)

    frp = fact_recall_precision(output, case.ground_truth.facts)
    rcs = reasoning_coherence_score(judge, output, case.ground_truth.reasoning_points)
    eda = end_to_end_decision_accuracy(output, case.ground_truth.decision)
    crr = compliance_reconstruction_rate(judge, output, case)

    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "condition": DPM.name,
        "budget_label": budget_label,
        "budget_chars": budget_chars,
        "trajectory_chars": trajectory_chars,
        "decision": output.decision,
        "FRP": frp, "RCS": rcs, "EDA": eda, "CRR": crr,
        "wall_time_s": time.time() - start,
        "agent_input_tokens": agent.input_tokens,
        "agent_output_tokens": agent.output_tokens,
        "judge_input_tokens": judge.input_tokens,
        "judge_output_tokens": judge.output_tokens,
        "cost_usd": agent.cost_usd() + judge.cost_usd(),
        "memory_view_preview": memory_surface[:400],
        "rationale_preview": output.rationale_memo[:240],
        "error": None,
    }


def run_parallel(cases_for_budget: List[Tuple[Case, str, int]], max_workers: int) -> List[Dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one_case, case, label, budget): (case.case_id, label)
            for case, label, budget in cases_for_budget
        }
        for fut in concurrent.futures.as_completed(futures):
            cid, label = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{r['condition']:>4}] {cid} @ {label}: "
                      f"FRP={r['FRP']:.2f} RCS={r['RCS']:.2f} "
                      f"EDA={r['EDA']:.2f} CRR={r['CRR']:.2f} "
                      f"({r['wall_time_s']:.1f}s ${r['cost_usd']:.3f})")
            except Exception as e:
                results.append({
                    "case_id": cid, "budget_label": label, "condition": "DPM",
                    "error": str(e), "FRP": 0.0, "RCS": 0.0, "EDA": 0.0, "CRR": 0.0,
                    "cost_usd": 0.0, "wall_time_s": 0.0,
                })
                print(f"  [ERR] {cid} @ {label}: {e}")
    return results


def load_stage2_baselines() -> List[Dict]:
    path = PROJECT_ROOT / "pilot" / "stage2" / "results.json"
    if not path.exists():
        raise RuntimeError(f"Stage 2 results not found at {path}; "
                          "DPM needs them for paired comparison.")
    with open(path) as f:
        return json.load(f)["results"]


def mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def agg_by(runs, condition, budget, metric, case_ids=None):
    vals = [r[metric] for r in runs
            if r["condition"] == condition and r["budget_label"] == budget
            and (case_ids is None or r["case_id"] in case_ids)]
    return mean(vals)


def derive_budgets_from_baselines(stage2: List[Dict]) -> Dict[str, int]:
    return {
        r["budget_label"]: r["budget_chars"]
        for r in stage2 if r.get("condition") == "Summ-only"
    }


def paired_permutation(a: List[float], b: List[float], n_perm: int = 10_000, seed: int = 0) -> Dict:
    assert len(a) == len(b)
    rnd = random.Random(seed)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    obs = mean(diffs)
    count = 0
    for _ in range(n_perm):
        permuted = [d * (1 if rnd.random() < 0.5 else -1) for d in diffs]
        if abs(mean(permuted)) >= abs(obs):
            count += 1
    return {"obs_diff": obs, "p_value": count / n_perm, "n_pairs": len(a)}


def strict_gate(sprint_results: List[Dict], stage2: List[Dict], budget: str) -> Tuple[bool, str]:
    """Pre-registered STRICT gate: DPM must beat Summ-only by >=5 pp on at least
    TWO of {EDA, RCS, CRR}. This is stricter than Stage 3's "match Summ-only
    minus 5 pp slack" gate. Rationale: we are proposing a novel architecture
    that should, if it is the right architecture, exceed the dominant baseline.
    Matching is not sufficient justification for a positive-result paper.
    """
    required_metrics = ("EDA", "RCS", "CRR")
    min_delta_pp = 0.05
    wins = 0
    lines = []
    case_ids = {r["case_id"] for r in sprint_results}
    for metric in required_metrics:
        dpm_mean = mean([r[metric] for r in sprint_results])
        summ_mean = agg_by(stage2, "Summ-only", budget, metric, case_ids=case_ids)
        delta = dpm_mean - summ_mean
        won = delta >= min_delta_pp
        wins += 1 if won else 0
        lines.append(f"  {metric}: DPM={dpm_mean:.2f} vs Summ-only={summ_mean:.2f} "
                     f"(delta={delta:+.2f}) {'WIN' if won else 'not'}")
    reasons = f"Wins on {wins}/{len(required_metrics)} of {required_metrics} " \
              f"(need >=2, margin >={int(min_delta_pp*100)} pp):\n" + "\n".join(lines)
    return wins >= 2, reasons


def sprint_case_set(cases: List[Case]) -> List[Case]:
    # Use a deterministic 5-case subset aligned with Stage 3's sprint pattern
    loans = [c for c in cases if c.domain == "loan"][:3]
    claims = [c for c in cases if c.domain == "claim"][:2]
    return loans + claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint-only", action="store_true",
                        help="Run only the sprint; do not proceed to full even if gate passes.")
    args = parser.parse_args()

    stage2 = load_stage2_baselines()
    stage2_budgets = derive_budgets_from_baselines(stage2)
    print(f"Stage 2 budgets: {stage2_budgets}")
    print(f"Stage 2 total rows: {len(stage2)}")

    cases = LARGE_CASES
    print(f"LARGE_CASES: {len(cases)} cases "
          f"({sum(1 for c in cases if c.domain=='loan')} loan, "
          f"{sum(1 for c in cases if c.domain=='claim')} claim)")

    # Verify case_id alignment with Stage 2 (so paired permutation is valid).
    stage2_ids = {r["case_id"] for r in stage2}
    large_ids = {c.case_id for c in cases}
    overlap = stage2_ids & large_ids
    if len(overlap) < len(large_ids):
        print(f"WARNING: case-id overlap is {len(overlap)}/{len(large_ids)}. "
              f"Paired comparisons will be limited to the overlap.")

    # ---- Sprint ----
    print("\n===== Sprint: DPM x moderate x 5 cases, 5-way parallel, temp=0 =====")
    sprint_cases = sprint_case_set(cases)
    moderate_budget = stage2_budgets.get("moderate")
    if moderate_budget is None:
        raise RuntimeError("Stage 2 'moderate' budget not found; cannot align.")
    sprint_inputs = [(c, "moderate", moderate_budget) for c in sprint_cases]

    t0 = time.time()
    sprint_results = run_parallel(sprint_inputs, max_workers=5)
    sprint_wall = time.time() - t0
    sprint_cost = sum(r.get("cost_usd", 0.0) for r in sprint_results)
    print(f"\nSprint wall time: {sprint_wall:.0f}s, cost: ${sprint_cost:.3f}")

    gate_pass, gate_reason = strict_gate(sprint_results, stage2, "moderate")
    print("\n--- STRICT kill-gate (pre-registered) ---")
    print(gate_reason)
    print(f"\nGate: {'PASS - proceed to full' if gate_pass else 'FAIL - stop here, paper unchanged'}")

    out_dir = PROJECT_ROOT / "pilot" / "stage4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sprint_results.json", "w") as f:
        json.dump({"meta": {"wall_s": sprint_wall, "cost_usd": sprint_cost,
                            "gate_pass": gate_pass, "gate_reason": gate_reason,
                            "temperature": 0.0},
                   "results": sprint_results}, f, indent=2)

    if args.sprint_only:
        print("\n--sprint-only: stopping here regardless of gate outcome.")
        return
    if not gate_pass:
        print("\nGate failed. COMMITMENT HONORED: no full run, no further architectures.")
        print("Paper ships with the enterprise-gap framing as the contribution.")
        return
    if sprint_cost > HARD_COST_CAP_USD:
        print(f"\nSprint already exceeded hard cost cap (${HARD_COST_CAP_USD}). Stopping.")
        return

    # ---- Full ----
    print("\n===== Full: DPM x 3 budgets x 10 cases, 10-way parallel, temp=0 =====")
    full_inputs = []
    for label, budget in stage2_budgets.items():
        for case in cases[:10]:
            full_inputs.append((case, label, budget))

    t0 = time.time()
    full_results = run_parallel(full_inputs, max_workers=10)
    full_wall = time.time() - t0
    full_cost = sum(r.get("cost_usd", 0.0) for r in full_results)
    total_cost = sprint_cost + full_cost
    print(f"\nFull wall time: {full_wall:.0f}s, full cost: ${full_cost:.3f}")
    print(f"Total cost (sprint + full): ${total_cost:.3f}")

    # ---- Permutation tests ----
    print("\n===== Permutation tests (DPM vs Stage 2 baselines) =====")
    tests = []
    baselines = ["TMC", "TMC-full", "Summ-only", "Retr-only", "Misrouted"]
    for budget in stage2_budgets.keys():
        for metric in ("FRP", "RCS", "EDA", "CRR"):
            dpm_by_case = {r["case_id"]: r[metric]
                           for r in full_results if r["budget_label"] == budget}
            for baseline in baselines:
                base_by_case = {r["case_id"]: r[metric]
                                for r in stage2
                                if r["condition"] == baseline and r["budget_label"] == budget}
                common = sorted(set(dpm_by_case) & set(base_by_case))
                if not common:
                    continue
                a = [dpm_by_case[c] for c in common]
                b = [base_by_case[c] for c in common]
                res = paired_permutation(a, b)
                tests.append({
                    "cond_a": "DPM", "cond_b": baseline, "budget": budget,
                    "metric": metric, **res,
                    "mean_a": mean(a), "mean_b": mean(b),
                })
                sig = "*" if res["p_value"] < 0.05 else " "
                print(f"  {sig} DPM vs {baseline:<10} @ {budget:<8} {metric:<4}: "
                      f"Δ={res['obs_diff']:+.2f} p={res['p_value']:.3f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"meta": {"sprint_wall_s": sprint_wall, "full_wall_s": full_wall,
                            "total_cost_usd": total_cost, "temperature": 0.0},
                   "results": full_results}, f, indent=2)
    with open(out_dir / "permutation_tests.json", "w") as f:
        json.dump({"tests": tests}, f, indent=2)

    # ---- Summary ----
    print("\n===== DPM summary (mean across cases, per budget) =====")
    print(f"{'Budget':<10} {'FRP':>6} {'RCS':>6} {'EDA':>6} {'CRR':>6}")
    for budget in stage2_budgets.keys():
        vals = {m: mean([r[m] for r in full_results if r["budget_label"] == budget])
                for m in ("FRP", "RCS", "EDA", "CRR")}
        print(f"{budget:<10} {vals['FRP']:>6.2f} {vals['RCS']:>6.2f} "
              f"{vals['EDA']:>6.2f} {vals['CRR']:>6.2f}")

    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
