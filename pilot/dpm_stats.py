"""Paired stats for DPM(ext) vs Summ-only at tight + loose, case-matched."""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
S2 = ROOT / "stage2" / "results.json"
EXT = ROOT / "stage4" / "extended_results.json"
SPR = ROOT / "stage4" / "sprint_results.json"
OUT = ROOT / "stage4" / "dpm_stats.json"
RNG = np.random.default_rng(20260420)


def paired_bootstrap_ci(a, b, n_boot=10000, alpha=0.05):
    deltas = np.array(a) - np.array(b)
    boot = RNG.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    return float(deltas.mean()), (float(np.percentile(boot, 100*alpha/2)),
                                   float(np.percentile(boot, 100*(1-alpha/2))))


def paired_permutation(a, b, n_perm=10000):
    a = np.array(a); b = np.array(b)
    deltas = a - b
    obs = deltas.mean()
    signs = RNG.choice([-1, 1], size=(n_perm, len(deltas)))
    null = (signs * deltas).mean(axis=1)
    return float((np.abs(null) >= abs(obs)).mean())


def mcnemars_exact(a_b, b_b):
    b01 = sum(1 for x, y in zip(a_b, b_b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a_b, b_b) if x == 1 and y == 0)
    n_d = b01 + b10
    if n_d == 0:
        return {"p_value": 1.0, "n_discordant": 0, "a_wins": 0, "b_wins": 0}
    res = binomtest(b10, n_d, p=0.5, alternative="two-sided")
    return {"p_value": float(res.pvalue), "n_discordant": n_d, "a_wins": b10, "b_wins": b01}


def cohens_h(p1, p2):
    p1 = min(max(p1, 1e-6), 1 - 1e-6)
    p2 = min(max(p2, 1e-6), 1 - 1e-6)
    return float(2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2)))


def main():
    s2 = json.loads(S2.read_text())["results"]
    ext = json.loads(EXT.read_text())["results"]
    spr = json.loads(SPR.read_text())["results"]

    dpm_idx = {}
    for r in ext:
        dpm_idx[(r["case_id"], r["budget_label"])] = r
    # Sprint is moderate-only; keyed similarly
    spr_idx = {(r["case_id"], "moderate"): r for r in spr}

    summ_idx = {(r["case_id"], r["budget_label"]): r for r in s2 if r["condition"] == "Summ-only"}

    comparisons = []
    for bud in ("tight", "moderate", "loose"):
        for metric in ("FRP", "RCS", "EDA", "CRR"):
            xs_dpm, xs_sum = [], []
            idx = dpm_idx if bud in ("tight", "loose") else spr_idx
            for (cid, b), rdpm in idx.items():
                if b != bud: continue
                rsum = summ_idx.get((cid, bud))
                if rsum is None: continue
                xs_dpm.append(float(rdpm[metric]))
                xs_sum.append(float(rsum[metric]))
            if len(xs_dpm) < 3: continue
            md, (lo, hi) = paired_bootstrap_ci(xs_dpm, xs_sum)
            pp = paired_permutation(xs_dpm, xs_sum)
            ab = [1 if x >= 0.5 else 0 for x in xs_dpm]
            bb = [1 if x >= 0.5 else 0 for x in xs_sum]
            mcn = mcnemars_exact(ab, bb)
            h = cohens_h(mean(xs_dpm), mean(xs_sum))
            row = {
                "budget": bud, "metric": metric, "n": len(xs_dpm),
                "mean_dpm": mean(xs_dpm), "mean_summ": mean(xs_sum),
                "mean_delta": md, "ci95": [lo, hi],
                "perm_p": pp, "mcnemar": mcn, "cohens_h": h,
            }
            comparisons.append(row)
            print(f"{bud:<10}{metric:<5} DPM={row['mean_dpm']:.3f} Summ={row['mean_summ']:.3f} "
                  f"delta={md:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] perm_p={pp:.4f} "
                  f"mcn_p={mcn['p_value']:.4f} h={h:+.2f}  (n={len(xs_dpm)})")

    OUT.write_text(json.dumps({"comparisons": comparisons}, indent=2))
    print(f"\nwrote: {OUT}")


if __name__ == "__main__":
    main()
