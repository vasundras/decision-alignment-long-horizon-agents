"""Step 5: judge-split validation.

Re-judges every committed Stage 2 row under both a Haiku judge and a Sonnet
judge, using the rationale_preview (220 chars) that was persisted in
pilot/stage2/results.json. For each row we recompute RCS and CRR under each
judge and record the two sets of scores alongside.

Limitation (documented in the paper): the full rationale_memo / denial_notice
were not persisted in Stage 2's results, only the first 240 characters. The
judge-split therefore measures judge-dependence on the stored preview, which
is the opening of the rationale but not the full text. A cleaner replication
would rerun every Stage 2 decision to recover the full text; at $10+ that was
not in Stage 5 scope.

Outputs:
  pilot/stage2/judge_split_results.json  — per-row haiku vs sonnet scores
  pilot/stage2/judge_split_stats.json    — disagreement / correlation /
                                            sign-agreement summary
"""
from __future__ import annotations
import concurrent.futures
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

from cases_large import LARGE_CASES
from metrics import AgentOutput, RCS_SYSTEM, CRR_SYSTEM, _normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"   # per env knowledge: Sonnet 4.6 is current

STAGE2_DIR = PROJECT_ROOT / "pilot" / "stage2"


class JudgeBackend:
    """LLMBackend-compatible wrapper with retry. Separate instance per thread."""

    def __init__(self, model: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        last_exc = None
        for attempt in range(5):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    system=system,
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


def _rcs(judge: JudgeBackend, rationale_preview: str, reasoning_points: List[str]) -> float:
    if not reasoning_points:
        return 1.0
    points_str = "\n".join(f"{i}: {p}" for i, p in enumerate(reasoning_points))
    user = (
        f"Agent rationale memo:\n---\n{rationale_preview}\n---\n"
        f"Denial/partial notice (if any):\n---\n(not separately stored)\n---\n"
        f"Reasoning points to judge:\n{points_str}\n\n"
        f'Return JSON like {{"0": true, "1": false, ...}} with exactly '
        f"{len(reasoning_points)} keys."
    )
    raw = judge.complete(RCS_SYSTEM, user, max_tokens=400).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return 0.0
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return 0.0
    hits = sum(1 for i in range(len(reasoning_points)) if parsed.get(str(i)) is True)
    return hits / len(reasoning_points)


def _crr(judge: JudgeBackend, decision: str, rationale_preview: str,
         domain: str, required_provisions: List[str]) -> float:
    # Approvals (no denial content required) are trivially compliant per the
    # metric definition in metrics.py. We mirror that convention.
    if "approve" in _normalize(decision) and "pay" not in _normalize(decision):
        return 1.0 if "approve" in _normalize(decision) else 0.0
    provisions = ", ".join(required_provisions) or "(none required)"
    user = (
        f"Case domain: {domain}\n"
        f"Agent decision: {decision}\n"
        f"Agent rationale:\n{rationale_preview}\n"
        f"Agent notice:\n(not separately stored)\n"
        f"Required provisions to cite: {provisions}\n"
        "Is this output compliant? Return only COMPLIANT or NONCOMPLIANT."
    )
    raw = judge.complete(CRR_SYSTEM, user, max_tokens=20).strip().upper()
    return 1.0 if "COMPLIANT" in raw and "NON" not in raw else 0.0


def judge_one_row(row: Dict, case) -> Dict:
    """Re-judge a single Stage 2 row under both judges. Returns a dict of
    per-metric, per-judge scores plus the original scores for easy diff."""
    rationale = row.get("rationale_preview", "")
    decision = row["decision"]
    out = {
        "case_id": row["case_id"],
        "condition": row["condition"],
        "budget_label": row["budget_label"],
        "decision": decision,
        "original_RCS": row["RCS"],
        "original_CRR": row["CRR"],
    }
    if decision in ("ERROR", "PARSE_ERROR", "ABSTAIN"):
        out.update({
            "haiku_RCS": 0.0, "haiku_CRR": 0.0,
            "sonnet_RCS": 0.0, "sonnet_CRR": 0.0,
            "skipped": True,
        })
        return out

    h = JudgeBackend(HAIKU)
    s = JudgeBackend(SONNET)
    out["haiku_RCS"] = _rcs(h, rationale, case.ground_truth.reasoning_points)
    out["haiku_CRR"] = _crr(h, decision, rationale, case.domain,
                             case.ground_truth.required_provisions)
    out["sonnet_RCS"] = _rcs(s, rationale, case.ground_truth.reasoning_points)
    out["sonnet_CRR"] = _crr(s, decision, rationale, case.domain,
                              case.ground_truth.required_provisions)
    out["skipped"] = False
    out["haiku_in_tokens"]  = h.input_tokens
    out["haiku_out_tokens"] = h.output_tokens
    out["sonnet_in_tokens"] = s.input_tokens
    out["sonnet_out_tokens"] = s.output_tokens
    return out


def main():
    results = json.loads((STAGE2_DIR / "results.json").read_text())["results"]
    cases_by_id = {c.case_id: c for c in LARGE_CASES}

    # Filter to committed rows only. Abstain / errors are not judge-dependent.
    judge_inputs = [r for r in results
                    if r["decision"] not in ("ERROR", "PARSE_ERROR", "ABSTAIN")
                    and r["case_id"] in cases_by_id]
    print(f"Judge-split: {len(judge_inputs)} committed rows from Stage 2")
    print(f"Judges: Haiku=`{HAIKU}`  Sonnet=`{SONNET}`  temp=0")

    out_rows = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(judge_one_row, r, cases_by_id[r["case_id"]]): i
                for i, r in enumerate(judge_inputs)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                res = fut.result()
                out_rows.append(res)
                done += 1
                if done % 15 == 0 or done == len(judge_inputs):
                    print(f"  ... {done}/{len(judge_inputs)} rows judged "
                          f"(elapsed {time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  ERR on row: {e}", flush=True)

    wall = time.time() - t0
    print(f"\nTotal wall: {wall:.0f}s")

    # Cost sums
    haiku_in  = sum(r.get("haiku_in_tokens", 0) for r in out_rows)
    haiku_out = sum(r.get("haiku_out_tokens", 0) for r in out_rows)
    sonn_in   = sum(r.get("sonnet_in_tokens", 0) for r in out_rows)
    sonn_out  = sum(r.get("sonnet_out_tokens", 0) for r in out_rows)
    cost_haiku = haiku_in * 1.0 / 1e6 + haiku_out * 5.0 / 1e6
    cost_sonn  = sonn_in  * 3.0 / 1e6 + sonn_out  * 15.0 / 1e6
    print(f"Haiku judge tokens: in={haiku_in:,} out={haiku_out:,} cost=${cost_haiku:.2f}")
    print(f"Sonnet judge tokens: in={sonn_in:,} out={sonn_out:,} cost=${cost_sonn:.2f}")

    (STAGE2_DIR / "judge_split_results.json").write_text(json.dumps({
        "meta": {
            "haiku_model": HAIKU, "sonnet_model": SONNET,
            "n_rows": len(out_rows),
            "wall_s": wall,
            "cost_usd": cost_haiku + cost_sonn,
            "note": ("rationale_preview is 220 chars (first segment of rationale_memo). "
                     "Full rationale was not persisted in Stage 2. Judge-split measures "
                     "judge-dependence on the stored preview.")
        },
        "results": out_rows,
    }, indent=2))
    print(f"\nWrote {STAGE2_DIR / 'judge_split_results.json'}")

    # ----- Summary stats -----
    def stats(ax_h: str, ax_s: str) -> Dict:
        pairs = [(r[ax_h], r[ax_s]) for r in out_rows if not r.get("skipped")]
        n = len(pairs)
        if n == 0:
            return {"n": 0}
        hs = [h for h, _ in pairs]; ss = [s for _, s in pairs]
        mad = sum(abs(h - s) for h, s in pairs) / n
        sign_agree = sum(1 for h, s in pairs if (h > 0.5) == (s > 0.5)) / n
        # Pearson
        import math
        mh = sum(hs) / n; ms = sum(ss) / n
        num = sum((h - mh) * (s - ms) for h, s in pairs)
        den_h = math.sqrt(sum((h - mh) ** 2 for h in hs))
        den_s = math.sqrt(sum((s - ms) ** 2 for s in ss))
        r = num / (den_h * den_s) if den_h and den_s else float("nan")
        return {"n": n, "mean_haiku": round(mh, 3), "mean_sonnet": round(ms, 3),
                "mean_abs_diff": round(mad, 3),
                "sign_agreement_at_half": round(sign_agree, 3),
                "pearson_r": round(r, 3) if r == r else None}

    rcs_stats = stats("haiku_RCS", "sonnet_RCS")
    crr_stats = stats("haiku_CRR", "sonnet_CRR")

    # Sign agreement on the Summ-only vs Retr-only pairwise comparison
    # at moderate budget (the headline comparison in Stage 2).
    def pairwise_sign(metric_h: str, metric_s: str, budget: str = "moderate") -> Dict:
        summ = {r["case_id"]: r for r in out_rows
                if r["condition"] == "Summ-only" and r["budget_label"] == budget}
        retr = {r["case_id"]: r for r in out_rows
                if r["condition"] == "Retr-only" and r["budget_label"] == budget}
        common = sorted(set(summ) & set(retr))
        if not common:
            return {"n_pairs": 0}
        h_diffs = [summ[c][metric_h] - retr[c][metric_h] for c in common]
        s_diffs = [summ[c][metric_s] - retr[c][metric_s] for c in common]
        sign_agree = sum(1 for h, s in zip(h_diffs, s_diffs)
                          if (h > 0 and s > 0) or (h < 0 and s < 0) or (h == 0 and s == 0))
        return {
            "n_pairs": len(common),
            "haiku_mean_delta": round(sum(h_diffs) / len(common), 3),
            "sonnet_mean_delta": round(sum(s_diffs) / len(common), 3),
            "sign_agree_fraction": round(sign_agree / len(common), 3),
        }

    stats_out = {
        "meta": {
            "haiku_model": HAIKU, "sonnet_model": SONNET,
            "n_rows": len(out_rows),
            "note": "Re-judged on stored rationale_preview (~220 chars)."
        },
        "rcs": rcs_stats,
        "crr": crr_stats,
        "summ_vs_retr_pairwise_moderate": {
            "RCS": pairwise_sign("haiku_RCS", "sonnet_RCS"),
            "CRR": pairwise_sign("haiku_CRR", "sonnet_CRR"),
        },
    }
    (STAGE2_DIR / "judge_split_stats.json").write_text(json.dumps(stats_out, indent=2))

    print("\n===== Judge-split summary =====")
    print(json.dumps(stats_out, indent=2))
    print(f"\nWrote {STAGE2_DIR / 'judge_split_stats.json'}")


if __name__ == "__main__":
    main()
