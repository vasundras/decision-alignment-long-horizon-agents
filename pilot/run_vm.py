"""Stage 5: Verified Memory (VM) evaluation.

VM tests a new evaluation axis: Calibrated Abstention Rate (CAR). Rather than
forcing every case to a decision, VM runs a completeness check on the memory
before committing. If the check flags insufficient evidence, the decision is
ABSTAIN rather than a guess.

Three companion metrics accompany the existing four (FRP, RCS, EDA, CRR):
  - commit_rate        — fraction of cases the agent committed to
  - conditional_accuracy — accuracy on committed cases only
  - commit_all_accuracy  — accuracy treating abstentions as wrong (the
                           baseline-equivalent score)

Interpretation:
  - If conditional_accuracy > Summ-only's EDA on the same cases, VM is
    well-calibrated: its abstentions concentrate on cases Summ-only would
    have guessed wrong.
  - If conditional_accuracy == Summ-only's EDA but commit_rate < 1.0, VM is
    over-abstaining: trading coverage for nothing.
  - If commit_rate == 1.0 and metrics match Summ-only, VM is adding no value
    beyond Summ-only (the check never fires).

The contribution is the new metric axis and the demonstrated tradeoff, not
raw-accuracy supremacy. VM does not have a "gate" — its value is
architectural (adds an enterprise-relevant capability: calibrated abstention)
at whatever conditional-accuracy point it lands on.

Usage:
    cd pilot/
    python run_vm.py
"""
from __future__ import annotations
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

from cases import ALL_CASES, Case
from cases_large import LARGE_CASES
from conditions import VM, Condition
from metrics import (
    AgentOutput, fact_recall_precision, reasoning_coherence_score,
    end_to_end_decision_accuracy, compliance_reconstruction_rate, car_components,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

AGENT_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
INPUT_COST_PER_MTOK = 1.0
OUTPUT_COST_PER_MTOK = 5.0
HARD_COST_CAP_USD = 3.0   # tighter than Stage 4 because this is a small add-on


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
                    model=self.model, max_tokens=max_tokens,
                    temperature=self.temperature, system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self.input_tokens += resp.usage.input_tokens
                self.output_tokens += resp.usage.output_tokens
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as e:
                last_exc = e
                if attempt == 4 or not any(k in str(e).lower() for k in
                    ("rate_limit", "overloaded", "429", "529", "timeout", "connection")):
                    raise
                time.sleep(min(30.0, 3.0 * (2 ** attempt)) + random.uniform(0, 2.0))
        raise last_exc

    def cost_usd(self) -> float:
        return (self.input_tokens * INPUT_COST_PER_MTOK / 1_000_000
                + self.output_tokens * OUTPUT_COST_PER_MTOK / 1_000_000)


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


def run_one_case(case: Case, budget_label: str, budget_chars: int) -> Dict:
    agent = AnthropicBackend(AGENT_MODEL, temperature=0.0)
    judge = AnthropicBackend(JUDGE_MODEL, temperature=0.0)
    start = time.time()

    cond = VM(agent, budget_chars=budget_chars)
    trajectory_chars = sum(len(c.text) for c in case.documents)
    for chunk in case.documents:
        cond.consolidate(chunk)

    # Completeness check BEFORE asking for a decision
    check = cond.completeness_check(case.task_prompt)
    gaps = check["gaps"]

    if not check["sufficient"]:
        output = AgentOutput(
            decision="ABSTAIN",
            rationale_memo="Memory insufficient for confident commitment. Gaps: "
                           + "; ".join(gaps),
            denial_notice="",
        )
        # When abstaining, we still compute FRP/RCS on the rationale (largely zero)
        # but the headline is that the agent correctly identified this as a case
        # to flag for human review rather than guessing.
    else:
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
        "condition": VM.name,
        "budget_label": budget_label,
        "budget_chars": budget_chars,
        "trajectory_chars": trajectory_chars,
        "decision": output.decision,
        "committed": check["sufficient"],
        "gaps": gaps,
        "FRP": frp, "RCS": rcs, "EDA": eda, "CRR": crr,
        "wall_time_s": time.time() - start,
        "agent_input_tokens": agent.input_tokens,
        "agent_output_tokens": agent.output_tokens,
        "judge_input_tokens": judge.input_tokens,
        "judge_output_tokens": judge.output_tokens,
        "cost_usd": agent.cost_usd() + judge.cost_usd(),
        "rationale_preview": output.rationale_memo[:240],
        "error": None,
    }


def run_parallel(inputs: List[Tuple[Case, str, int]], workers: int) -> List[Dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one_case, c, l, b): (c.case_id, l) for c, l, b in inputs}
        for fut in concurrent.futures.as_completed(futures):
            cid, label = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                committed = "COMMIT " if r["committed"] else "ABSTAIN"
                print(f"  [VM ] {cid} @ {label}: {committed} | "
                      f"decision='{r['decision'][:18]:<18}' | "
                      f"EDA={r['EDA']:.2f} FRP={r['FRP']:.2f} "
                      f"({r['wall_time_s']:.1f}s ${r['cost_usd']:.3f})")
            except Exception as e:
                results.append({"case_id": cid, "budget_label": label, "condition": "VM",
                                "error": str(e), "FRP": 0.0, "RCS": 0.0, "EDA": 0.0, "CRR": 0.0,
                                "committed": True, "gaps": [],
                                "cost_usd": 0.0, "wall_time_s": 0.0,
                                "decision": "ERROR"})
                print(f"  [ERR] {cid} @ {label}: {e}")
    return results


def mean(xs): return sum(xs) / len(xs) if xs else 0.0


def main():
    with open(PROJECT_ROOT / "pilot" / "stage2" / "results.json") as f:
        stage2 = json.load(f)["results"]
    budgets = {r["budget_label"]: r["budget_chars"]
               for r in stage2 if r["condition"] == "Summ-only"}
    moderate_budget = budgets["moderate"]

    # 10 cases x moderate budget. Stage 5 is small by design.
    cases = LARGE_CASES[:10]
    inputs = [(c, "moderate", moderate_budget) for c in cases]

    print(f"Stage 5: VM x moderate x {len(cases)} cases, 3-way parallel, temp=0 "
          "(reduced from 10 to stay under 10k output-tokens/min rate limit)")
    t0 = time.time()
    results = run_parallel(inputs, workers=3)
    wall = time.time() - t0
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    print(f"\nWall: {wall:.0f}s, cost: ${total_cost:.3f}")

    # Compare to Summ-only on same cases at moderate
    case_ids = [c.case_id for c in cases]
    summ_rows = [r for r in stage2
                 if r["condition"] == "Summ-only" and r["budget_label"] == "moderate"
                 and r["case_id"] in case_ids]
    # IMPORTANT: `results` arrives from concurrent.futures.as_completed in
    # completion order, and `summ_rows` is filtered out of the Stage 2 file
    # in file order. Neither matches `cases` order, so we must pair by
    # case_id. Earlier versions paired positionally, producing nonsense
    # conditional-accuracy numbers.
    vm_by_cid   = {r["case_id"]: r.get("decision", "ERROR") for r in results}
    summ_by_cid = {r["case_id"]: r.get("decision", "ERROR") for r in summ_rows}
    vm_decisions   = [vm_by_cid[c.case_id]   for c in cases]
    summ_decisions = [summ_by_cid[c.case_id] for c in cases]
    gts            = [c.ground_truth.decision for c in cases]

    car_vm   = car_components(vm_decisions, gts)
    car_summ = car_components(summ_decisions, gts)

    print("\n=== CAR components ===")
    print(f"{'Cond':<10} {'n':>3} {'commit':>7} {'cond_acc':>8} {'commit_all':>10} {'abstain':>7}")
    print(f"{'Summ-only':<10} {car_summ['n']:>3} {car_summ['commit_rate']:>7.2f} "
          f"{car_summ['conditional_accuracy']:>8.2f} {car_summ['commit_all_accuracy']:>10.2f} "
          f"{car_summ['abstain_count']:>7}")
    print(f"{'VM':<10} {car_vm['n']:>3} {car_vm['commit_rate']:>7.2f} "
          f"{car_vm['conditional_accuracy']:>8.2f} {car_vm['commit_all_accuracy']:>10.2f} "
          f"{car_vm['abstain_count']:>7}")

    # Per-case reconciliation
    print("\n=== Per-case reconciliation ===")
    print(f"{'case_id':<18} {'Summ-decision':<18} {'VM-decision':<18} {'GT':<15} "
          f"{'Summ✓':>5} {'VM✓':>5}")
    def _norm(s): return s.lower().strip()
    def _match(d, g): return _norm(g) in _norm(d) or _norm(d) in _norm(g)
    for c, vm_r in zip(cases, results):
        summ_r = next((r for r in summ_rows if r["case_id"] == c.case_id), None)
        if not summ_r: continue
        s_ok = "Y" if _match(summ_r["decision"], c.ground_truth.decision) else "N"
        v_ok = "-" if vm_r["decision"] == "ABSTAIN" else (
            "Y" if _match(vm_r["decision"], c.ground_truth.decision) else "N")
        print(f"{c.case_id:<18} {summ_r['decision'][:17]:<18} {vm_r['decision'][:17]:<18} "
              f"{c.ground_truth.decision[:14]:<15} {s_ok:>5} {v_ok:>5}")

    # Headline
    print("\n=== Headline ===")
    delta_cond = car_vm['conditional_accuracy'] - car_summ['commit_all_accuracy']
    delta_cov  = car_vm['commit_rate'] - 1.0
    print(f"VM conditional_accuracy - Summ-only commit_all_accuracy: {delta_cond:+.2f}")
    print(f"VM commit_rate                                         : {car_vm['commit_rate']:.2f}")
    print(f"VM abstentions                                         : {car_vm['abstain_count']}")
    if delta_cond > 0.05 and car_vm['commit_rate'] >= 0.5:
        print("RESULT: well-calibrated. VM trades coverage for accuracy productively.")
    elif car_vm['commit_rate'] == 1.0 and car_vm['conditional_accuracy'] == car_summ['commit_all_accuracy']:
        print("RESULT: check never fires. VM behaves as Summ-only. No added value.")
    elif delta_cond <= 0 and car_vm['commit_rate'] < 1.0:
        print("RESULT: over-abstaining. VM loses coverage without gaining accuracy.")
    else:
        print("RESULT: mixed; inspect per-case table for pattern.")

    out = PROJECT_ROOT / "pilot" / "stage5"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vm_results.json", "w") as f:
        json.dump({"meta": {"wall_s": wall, "cost_usd": total_cost,
                            "car_vm": car_vm, "car_summ": car_summ,
                            "delta_conditional": delta_cond},
                   "results": results}, f, indent=2)
    print(f"\nArtifacts: {out}/vm_results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
