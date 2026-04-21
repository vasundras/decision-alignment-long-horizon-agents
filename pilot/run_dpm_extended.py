"""DPM extension to tight + loose budgets at n=10 (confirmatory follow-up to Stage 4 sprint).

Reuses run_dpm.run_one_case verbatim so the projection pipeline is identical.
"""
from __future__ import annotations
import concurrent.futures
import json
import time
from pathlib import Path

from cases_large import LARGE_CASES
from run_dpm import run_one_case, INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK

BUDGETS = {"tight": 1338, "loose": 13381}


def main():
    out_path = Path(__file__).resolve().parent / "stage4" / "extended_results.json"
    results = []
    t_start = time.time()

    tasks = [(c, lab, b) for lab, b in BUDGETS.items() for c in LARGE_CASES]
    print(f"running {len(tasks)} (case,budget) DPM units", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_one_case, c, lab, b, 0): (c.case_id, lab) for c, lab, b in tasks}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                cid, lab = futs[fut]
                print(f"done {cid}/{lab}: FRP={r.get('FRP')} EDA={r.get('EDA')}", flush=True)
            except Exception as e:
                cid, lab = futs[fut]
                print(f"ERR {cid}/{lab}: {e}", flush=True)
                results.append({"case_id": cid, "budget_label": lab, "error": str(e)})

    in_tok = sum(r.get("agent_input_tokens", 0) + r.get("judge_input_tokens", 0) for r in results)
    out_tok = sum(r.get("agent_output_tokens", 0) + r.get("judge_output_tokens", 0) for r in results)
    cost = in_tok / 1e6 * INPUT_COST_PER_MTOK + out_tok / 1e6 * OUTPUT_COST_PER_MTOK

    out_path.write_text(json.dumps({
        "meta": {"n_cases": len(LARGE_CASES), "budgets": BUDGETS,
                 "wall_s": time.time() - t_start, "cost_usd": cost},
        "results": results,
    }, indent=2))
    print(f"wrote {out_path}; cost ${cost:.2f}; wall {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
