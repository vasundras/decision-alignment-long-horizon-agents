"""Stage 3: Schema-Anchored Memory (SAM) evaluation.

Runs SAM against the same cases and budgets used in Stage 2 and compares to
the existing results.json baselines. Parallelized with a thread pool to cut
wall clock time.

Tiered protocol with a pre-registered kill-gate:

  Sprint (target: ~5 min, ~$0.50)
    SAM × moderate budget × 5 cases, 5-way parallel.
    Gate: proceed iff SAM's mean EDA, FRP, and RCS are each >= Summ-only's
    mean on the same cases and budget minus 5 pp slack.
    If gate fails, stop. Paper does not change.

  Full (target: ~15-20 min, ~$2-3)
    SAM × 3 budgets × 10 cases, 10-way parallel.
    Compute paired permutation tests against every Stage 2 condition.
    If SAM wins, update the paper.

Hard cost cap: $5 total. Abort if projected cost exceeds cap.

Usage:
    cd pilot/
    python run_sam.py             # runs sprint; asks interactively before full
    python run_sam.py --full-auto # runs sprint + full without asking
    python run_sam.py --sprint-only
"""
from __future__ import annotations
import argparse
import asyncio
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

# Reuse existing Stage 1/2 infrastructure.
from cases import ALL_CASES, Case
from cases_large import LARGE_CASES
from conditions import SAM, Condition
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
# Backend (same shape as Stage 2 but each case gets its own client so thread
# concurrency is clean).
# -----------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, model: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        # Retry on transient rate-limit / overload / transport errors.
        last_exc = None
        for attempt in range(5):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
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
                # Cap backoff at 30s; rate-limit window is 60s but we want
                # progress, not pathological chains.
                sleep_s = min(30.0, 3.0 * (2 ** attempt)) + random.uniform(0, 2.0)
                time.sleep(sleep_s)
        raise last_exc  # unreachable

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


# -----------------------------------------------------------------------------
# Per-case runner (synchronous; parallelized via thread pool)
# -----------------------------------------------------------------------------

def run_one_case(
    case: Case,
    budget_label: str,
    budget_chars: int,
    seed: int = 0,
) -> Dict:
    """Run a single (SAM, case, budget) trajectory and compute all four metrics."""
    random.seed(seed)
    agent = AnthropicBackend(AGENT_MODEL)
    judge = AnthropicBackend(JUDGE_MODEL)
    start = time.time()

    cond = SAM(agent, budget_chars=budget_chars)
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
        "condition": SAM.name,
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
        "rationale_preview": output.rationale_memo[:240],
        "error": None,
    }


def run_parallel(
    cases_for_budget: List[Tuple[Case, str, int]],
    max_workers: int,
) -> List[Dict]:
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
                    "case_id": cid, "budget_label": label, "condition": "SAM",
                    "error": str(e), "FRP": 0.0, "RCS": 0.0, "EDA": 0.0, "CRR": 0.0,
                    "cost_usd": 0.0, "wall_time_s": 0.0,
                })
                print(f"  [ERR] {cid} @ {label}: {e}")
    return results


# -----------------------------------------------------------------------------
# Load Stage 2 baselines for comparison
# -----------------------------------------------------------------------------

def load_stage2_baselines() -> Dict:
    path = PROJECT_ROOT / "pilot" / "stage2" / "results.json"
    if not path.exists():
        raise RuntimeError(f"Stage 2 results not found at {path}; "
                          "SAM needs them for comparison.")
    with open(path) as f:
        return json.load(f)["results"]


def mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def agg_by(runs, condition, budget, metric):
    return mean([r[metric] for r in runs
                 if r["condition"] == condition and r["budget_label"] == budget])


# -----------------------------------------------------------------------------
# Budget derivation (match Stage 2)
# -----------------------------------------------------------------------------

def derive_budgets(cases: List[Case]) -> Dict[str, int]:
    avg_chars = sum(sum(len(c.text) for c in case.documents) for case in cases) / len(cases)
    return {
        "loose":    int(avg_chars * 0.50),
        "moderate": int(avg_chars * 0.20),
        "tight":    int(avg_chars * 0.05),
    }


# -----------------------------------------------------------------------------
# Permutation test (paired sign-flip)
# -----------------------------------------------------------------------------

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
    p = count / n_perm
    return {"obs_diff": obs, "p_value": p, "n_pairs": len(a)}


# -----------------------------------------------------------------------------
# Sprint + kill-gate + full
# -----------------------------------------------------------------------------

def sprint_cases(cases: List[Case]) -> List[Case]:
    # Pick 5 cases: interleave domains if possible.
    loans = [c for c in cases if c.domain == "loan"][:3]
    claims = [c for c in cases if c.domain == "claim"][:2]
    return loans + claims


def check_gate(sprint_results: List[Dict], stage2: List[Dict], budget: str) -> Tuple[bool, str]:
    """Decide whether to proceed from sprint to full.

    Criterion (pre-registered): SAM's mean on EDA, FRP, and RCS must each be
    >= Summ-only's mean at this budget minus 5 pp slack. Else stop.
    """
    slack = 0.05
    reasons = []
    pass_all = True
    for metric in ("EDA", "FRP", "RCS"):
        sam_mean = mean([r[metric] for r in sprint_results])
        summ_mean = agg_by(stage2, "Summ-only", budget, metric)
        gap = sam_mean - summ_mean
        reasons.append(f"  {metric}: SAM={sam_mean:.2f} vs Summ-only={summ_mean:.2f} (gap={gap:+.2f})")
        if sam_mean < summ_mean - slack:
            pass_all = False
    return pass_all, "\n".join(reasons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint-only", action="store_true")
    parser.add_argument("--full-auto", action="store_true")
    args = parser.parse_args()

    # Load baselines & build budget map to match Stage 2.
    stage2 = load_stage2_baselines()
    # Use the same budgets Stage 2 derived.
    stage2_budgets = {
        r["budget_label"]: r["budget_chars"]
        for r in stage2 if r.get("condition") == "Summ-only"
    }
    print(f"Stage 2 budgets: {stage2_budgets}")

    # Stage 2 used cases_large.LARGE_CASES (deterministic, template-based).
    # Import it directly — produces identical case_ids as Stage 2's run, which
    # keeps the paired-permutation pairing valid.
    cases = LARGE_CASES
    stage2_case_ids = {r["case_id"] for r in stage2}
    sam_case_ids = {c.case_id for c in cases}
    overlap = stage2_case_ids & sam_case_ids
    print(f"Using {len(cases)} deterministic LARGE_CASES (Stage 2 re-use). "
          f"ID overlap with Stage 2 baselines: {len(overlap)} / {len(stage2_case_ids)}")
    if len(overlap) < len(stage2_case_ids):
        missing = stage2_case_ids - sam_case_ids
        print(f"  WARNING: missing case IDs (pairing will drop them): {sorted(missing)[:5]}...")

    # ---- Sprint ----
    print("\n===== Sprint: SAM × moderate × 5 cases, 5-way parallel =====")
    sprint_cases_list = sprint_cases(cases)
    moderate_budget = stage2_budgets.get("moderate", 5352)
    sprint_inputs = [(c, "moderate", moderate_budget) for c in sprint_cases_list]

    t0 = time.time()
    sprint_results = run_parallel(sprint_inputs, max_workers=1)
    sprint_wall = time.time() - t0
    sprint_cost = sum(r.get("cost_usd", 0.0) for r in sprint_results)
    print(f"\nSprint wall time: {sprint_wall:.0f}s, cost: ${sprint_cost:.3f}")

    gate_pass, gate_reason = check_gate(sprint_results, stage2, "moderate")
    print("\n--- Kill-gate check (pre-registered) ---")
    print(gate_reason)
    print(f"\nGate: {'PASS — proceed to full' if gate_pass else 'FAIL — stopping at sprint'}")

    # Save sprint artifacts regardless of gate outcome.
    out_dir = PROJECT_ROOT / "pilot" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sprint_results.json", "w") as f:
        json.dump({"meta": {"wall_s": sprint_wall, "cost_usd": sprint_cost,
                            "gate_pass": gate_pass, "gate_reason": gate_reason},
                   "results": sprint_results}, f, indent=2)

    if args.sprint_only:
        print("\n--sprint-only: stopping here.")
        return
    if not gate_pass and not args.full_auto:
        print("\nGate failed. Stopping (use --full-auto to override).")
        return
    if sprint_cost > HARD_COST_CAP_USD:
        print(f"\nSprint already exceeded hard cost cap (${HARD_COST_CAP_USD}). Stopping.")
        return

    # ---- Full ----
    print("\n===== Full: SAM × 3 budgets × 10 cases, 10-way parallel =====")
    full_inputs = []
    for label, budget in stage2_budgets.items():
        for case in cases[:10]:
            full_inputs.append((case, label, budget))

    t0 = time.time()
    full_results = run_parallel(full_inputs, max_workers=2)
    full_wall = time.time() - t0
    full_cost = sum(r.get("cost_usd", 0.0) for r in full_results)
    total_cost = sprint_cost + full_cost
    print(f"\nFull wall time: {full_wall:.0f}s, full cost: ${full_cost:.3f}")
    print(f"Total cost (sprint + full): ${total_cost:.3f}")

    # ---- Permutation tests against Stage 2 baselines ----
    print("\n===== Permutation tests =====")
    tests = []
    baselines = ["TMC", "TMC-full", "Summ-only", "Retr-only", "Misrouted"]
    for budget in stage2_budgets.keys():
        for metric in ("FRP", "RCS", "EDA", "CRR"):
            sam_by_case = {r["case_id"]: r[metric]
                           for r in full_results if r["budget_label"] == budget}
            for baseline in baselines:
                base_by_case = {r["case_id"]: r[metric]
                                for r in stage2
                                if r["condition"] == baseline and r["budget_label"] == budget}
                common = sorted(set(sam_by_case) & set(base_by_case))
                if not common:
                    continue
                a = [sam_by_case[c] for c in common]
                b = [base_by_case[c] for c in common]
                res = paired_permutation(a, b)
                tests.append({
                    "cond_a": "SAM", "cond_b": baseline, "budget": budget,
                    "metric": metric, **res,
                    "mean_a": mean(a), "mean_b": mean(b),
                })
                sig = "*" if res["p_value"] < 0.05 else " "
                print(f"  {sig} SAM vs {baseline:<10} @ {budget:<8} {metric:<4}: "
                      f"Δ={res['obs_diff']:+.2f} p={res['p_value']:.3f}")

    # ---- Save ----
    with open(out_dir / "results.json", "w") as f:
        json.dump({"meta": {"sprint_wall_s": sprint_wall, "full_wall_s": full_wall,
                            "total_cost_usd": total_cost},
                   "results": full_results}, f, indent=2)
    with open(out_dir / "permutation_tests.json", "w") as f:
        json.dump({"tests": tests}, f, indent=2)

    # ---- Summary ----
    print("\n===== SAM summary (mean across cases, per budget) =====")
    print(f"{'Budget':<10} {'FRP':>6} {'RCS':>6} {'EDA':>6} {'CRR':>6}")
    for budget in stage2_budgets.keys():
        for metric in ("FRP", "RCS", "EDA", "CRR"):
            pass
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
