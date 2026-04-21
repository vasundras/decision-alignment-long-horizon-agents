"""Stage 5B: VM threshold sweep — three completeness-check prompt strictnesses.

Motivation. VM at its default (strict) prompt over-abstains on the Stage 5 run
(commit_rate=0.20, cond_acc=0.50, commit_all_acc=0.10). A strictness knob is
the natural next experiment: if CAR is a real axis, we should be able to
trade coverage for conditional accuracy along a controllable curve.

This script reuses run_vm.run_one_case but monkey-patches
conditions.VM_COMPLETENESS_SYSTEM with three prompt variants before each
sweep pass. Results written to pilot/stage5/sweep_results.json.
"""
from __future__ import annotations
import concurrent.futures
import json
import time
from pathlib import Path

from cases_large import LARGE_CASES
import conditions as _cond_mod
import run_vm
from metrics import car_components

PROMPTS = {
    "strict": (
        "You are the completeness-check layer for an enterprise decision agent. "
        "Given (a) the consolidated memory for a case and (b) the decision task, "
        "decide whether the memory contains sufficient information to reach a "
        "confident, defensible decision that would survive regulatory audit. "
        "A memory is INSUFFICIENT if ANY of the following hold: any required "
        "factual anchor is missing, ambiguous, or not quoted verbatim; any "
        "reasoning step required for the decision is unresolved; any material "
        "customer correspondence is unanswered; any policy/regulatory provision "
        "is cited without being quoted. Err aggressively toward INSUFFICIENT.\n\n"
        "Return a single JSON object with two fields:\n"
        '  "sufficient": true or false\n'
        '  "gaps": array of short strings describing specific evidence gaps '
        "(empty array if sufficient)\n\n"
        "Return ONLY the JSON. No prose."
    ),
    "moderate": (
        "You are the completeness-check layer for an enterprise decision agent. "
        "Given (a) the consolidated memory for a case and (b) the decision task, "
        "decide whether the memory contains sufficient information to reach a "
        "confident decision. A memory is INSUFFICIENT only if a load-bearing "
        "factual anchor is missing or a material reasoning step is unresolved. "
        "Minor paraphrase of cited provisions, or ordinary inference gaps that "
        "a diligent adjudicator would resolve, are NOT grounds for insufficient. "
        "Abstain when the decision would be a guess.\n\n"
        "Return a single JSON object with two fields:\n"
        '  "sufficient": true or false\n'
        '  "gaps": array of short strings describing specific evidence gaps '
        "(empty array if sufficient)\n\n"
        "Return ONLY the JSON. No prose."
    ),
    "permissive": (
        "You are the completeness-check layer for an enterprise decision agent. "
        "Given (a) the consolidated memory for a case and (b) the decision task, "
        "decide whether the decision is reachable from the memory at all. A "
        "memory is INSUFFICIENT only if the required decision cannot be "
        "determined with the evidence present — i.e. a fundamental anchor is "
        "entirely absent, not merely paraphrased or incomplete. If a diligent "
        "adjudicator could reach the decision given the memory, the memory is "
        "sufficient.\n\n"
        "Return a single JSON object with two fields:\n"
        '  "sufficient": true or false\n'
        '  "gaps": array of short strings describing specific evidence gaps '
        "(empty array if sufficient)\n\n"
        "Return ONLY the JSON. No prose."
    ),
}


def run_at_strictness(level: str, cases, budget_label: str, budget_chars: int):
    _cond_mod.VM_COMPLETENESS_SYSTEM = PROMPTS[level]
    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_vm.run_one_case, c, budget_label, budget_chars): c.case_id
                for c in cases}
        for fut in concurrent.futures.as_completed(futs):
            cid = futs[fut]
            try:
                r = fut.result()
                r["strictness"] = level
                results.append(r)
                committed = "COMMIT " if r["committed"] else "ABSTAIN"
                print(f"  [{level:<10}] {cid}: {committed} | EDA={r['EDA']:.2f} "
                      f"FRP={r['FRP']:.2f} (${r['cost_usd']:.3f})", flush=True)
            except Exception as e:
                results.append({"case_id": cid, "strictness": level, "error": str(e)})
                print(f"  [{level:<10}] {cid}: ERR {e}", flush=True)
    return results, time.time() - t0


def main():
    ROOT = Path(__file__).resolve().parent
    out_path = ROOT / "stage5" / "sweep_results.json"

    # Budget = moderate from Stage 2
    stage2 = json.loads((ROOT / "stage2" / "results.json").read_text())["results"]
    moderate_budget = next(r["budget_chars"] for r in stage2
                           if r["condition"] == "Summ-only" and r["budget_label"] == "moderate")
    cases = LARGE_CASES[:10]

    all_results = []
    meta_by_level = {}
    t_total = time.time()
    for level in ("strict", "moderate", "permissive"):
        print(f"\n=== strictness={level} ===", flush=True)
        results, wall = run_at_strictness(level, cases, "moderate", moderate_budget)
        all_results.extend(results)
        # CAR components at this strictness.
        # IMPORTANT: `results` is in concurrent.futures.as_completed (arrival)
        # order, so we must pair decisions back to cases by case_id. Earlier
        # versions of this script paired positionally, which silently scored
        # each result against the wrong ground truth.
        by_cid = {r["case_id"]: r.get("decision", "ERROR") for r in results}
        decisions = [by_cid[c.case_id] for c in cases]
        gts = [c.ground_truth.decision for c in cases]
        car = car_components(decisions, gts)
        meta_by_level[level] = {
            "wall_s": wall,
            "cost_usd": sum(r.get("cost_usd", 0.0) for r in results),
            "car": car,
        }
        print(f"  n={car['n']} commit={car['commit_rate']:.2f} "
              f"cond_acc={car['conditional_accuracy']:.2f} "
              f"commit_all={car['commit_all_accuracy']:.2f} "
              f"abstain={car['abstain_count']}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {"wall_total_s": time.time() - t_total, "by_level": meta_by_level},
        "results": all_results,
    }, indent=2))
    print(f"\nwrote {out_path}")
    print("\n=== Sweep summary ===")
    print(f"{'level':<12} {'commit':>7} {'cond_acc':>9} {'commit_all':>11} {'abstain':>8}")
    for lev, m in meta_by_level.items():
        car = m["car"]
        print(f"{lev:<12} {car['commit_rate']:>7.2f} {car['conditional_accuracy']:>9.2f} "
              f"{car['commit_all_accuracy']:>11.2f} {car['abstain_count']:>8}")


if __name__ == "__main__":
    main()
