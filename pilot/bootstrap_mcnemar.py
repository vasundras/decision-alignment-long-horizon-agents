"""Bootstrap 95% CI and McNemar's exact on Stage 2 pre-existing results.

Run now on pilot/stage2/results.json. Produces:
  - paired mean delta, permutation p, bootstrap 95% CI on delta
  - McNemar's exact p-value on the binary per-case indicator
  - effect size (Cohen's h for proportions)

Outputs: pilot/stage2/bootstrap_mcnemar.json (consumed by paper) + stdout table.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.stats import binomtest

RESULTS = Path(__file__).resolve().parent / "stage2" / "results.json"
OUT = Path(__file__).resolve().parent / "stage2" / "bootstrap_mcnemar.json"
RNG = np.random.default_rng(20260420)


def paired_bootstrap_ci(a: list[float], b: list[float], n_boot: int = 10000, alpha: float = 0.05):
    assert len(a) == len(b)
    deltas = np.array(a) - np.array(b)
    n = len(deltas)
    boot = RNG.choice(deltas, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(deltas.mean()), (lo, hi)


def paired_permutation(a: list[float], b: list[float], n_perm: int = 10000) -> float:
    a = np.array(a); b = np.array(b)
    deltas = a - b
    obs = deltas.mean()
    signs = RNG.choice([-1, 1], size=(n_perm, len(deltas)))
    null = (signs * deltas).mean(axis=1)
    return float((np.abs(null) >= abs(obs)).mean())


def mcnemars_exact(a_binary: list[int], b_binary: list[int]) -> dict:
    """Two-sided McNemar's exact on paired binary outcomes.

    Cells: a=1 b=0 (b_wins=0, a_wins=1) and a=0 b=1 (b_wins=1, a_wins=0).
    Exact binomial test on the discordant pairs.
    """
    assert len(a_binary) == len(b_binary)
    b01 = sum(1 for x, y in zip(a_binary, b_binary) if x == 0 and y == 1)  # b wins
    b10 = sum(1 for x, y in zip(a_binary, b_binary) if x == 1 and y == 0)  # a wins
    n_disc = b01 + b10
    if n_disc == 0:
        return {"p_value": 1.0, "n_discordant": 0, "a_wins": 0, "b_wins": 0}
    res = binomtest(b10, n_disc, p=0.5, alternative="two-sided")
    return {"p_value": float(res.pvalue), "n_discordant": n_disc, "a_wins": b10, "b_wins": b01}


def cohens_h(p1: float, p2: float) -> float:
    # arcsine transform; use small clip to avoid asin domain errors on exactly-0/1
    p1 = min(max(p1, 1e-6), 1 - 1e-6)
    p2 = min(max(p2, 1e-6), 1 - 1e-6)
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def load() -> list[dict]:
    return json.loads(RESULTS.read_text())["results"]


def main():
    rows = load()
    # Index: (case_id, condition, budget) -> metrics
    idx = {(r["case_id"], r["condition"], r["budget_label"]): r for r in rows}
    case_ids = sorted({r["case_id"] for r in rows})
    budgets = ["loose", "moderate", "tight"]
    metrics = ["FRP", "RCS", "EDA", "CRR"]
    conditions = ["TMC", "TMC-full", "Summ-only", "Retr-only", "Misrouted"]

    summary = {"comparisons": []}

    def pair(cond_a: str, cond_b: str, budget: str, metric: str):
        xs_a, xs_b = [], []
        for cid in case_ids:
            ra = idx.get((cid, cond_a, budget))
            rb = idx.get((cid, cond_b, budget))
            if ra is None or rb is None:
                continue
            xs_a.append(float(ra[metric]))
            xs_b.append(float(rb[metric]))
        return xs_a, xs_b

    # Headline comparisons per reviewer request
    headline = [
        ("Summ-only", "Retr-only", "FRP"),
        ("Summ-only", "Retr-only", "RCS"),
        ("Summ-only", "Retr-only", "EDA"),
        ("Summ-only", "TMC", "FRP"),
        ("Summ-only", "TMC-full", "FRP"),
        ("Summ-only", "Misrouted", "FRP"),
        ("Summ-only", "Retr-only", "CRR"),
    ]

    print(f"{'Comp':<40} {'Budget':<10} {'Metric':<6} {'meanDelta':>10} {'CI95':>22} {'perm_p':>8} {'mcnemar_p':>10} {'h':>6}")
    for cond_a, cond_b, metric in headline:
        for budget in budgets:
            xs_a, xs_b = pair(cond_a, cond_b, budget, metric)
            if not xs_a:
                continue
            mean_delta, (lo, hi) = paired_bootstrap_ci(xs_a, xs_b)
            perm_p = paired_permutation(xs_a, xs_b)
            # Binarize: 1 if >= threshold for metric (any positive fact/reasoning point counts)
            # For FRP/RCS (in [0,1]) we binarize at >= 0.5; for EDA/CRR already binary-like.
            if metric in {"FRP", "RCS"}:
                ab = [1 if x >= 0.5 else 0 for x in xs_a]
                bb = [1 if x >= 0.5 else 0 for x in xs_b]
            else:
                ab = [1 if x >= 0.5 else 0 for x in xs_a]
                bb = [1 if x >= 0.5 else 0 for x in xs_b]
            mcn = mcnemars_exact(ab, bb)
            h = cohens_h(mean(xs_a), mean(xs_b))
            comp = f"{cond_a} vs {cond_b}"
            print(
                f"{comp:<40} {budget:<10} {metric:<6} "
                f"{mean_delta:>+10.3f} [{lo:>+6.3f},{hi:>+6.3f}] "
                f"{perm_p:>8.4f} {mcn['p_value']:>10.4f} {h:>+6.2f}"
            )
            summary["comparisons"].append({
                "cond_a": cond_a, "cond_b": cond_b, "budget": budget, "metric": metric,
                "n": len(xs_a), "mean_delta": mean_delta,
                "ci95": [lo, hi], "perm_p": perm_p,
                "mcnemar": mcn, "cohens_h": h,
                "mean_a": mean(xs_a), "mean_b": mean(xs_b),
            })

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {OUT}")


if __name__ == "__main__":
    main()
