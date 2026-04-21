# Pre-registration — *When Memory Matters* (formerly "Typed Memory Consolidation")

This document pins the directional hypotheses under which the paper's
experiments were planned and what the measured outcomes were. It is written
after the fact, but the hypotheses themselves were committed in writing
before the runs that measured them, in the paper drafts included in this
repository.

## H1 (precision loss — **reversed by data**)

- **Pre-registered prediction** (committed in `tmc_paper.tex` as of the
  Stage 1 pilot, before Stage 2 measurements):
  > **H1.** Summarization (Summ-only) will fail Fact Recall Precision (FRP)
  > relative to a retrieval buffer (Retr-only), because summarization
  > compresses exact numeric anchors into abstractions.
  > Predicted direction: $\text{FRP}(\text{Summ-only}) < \text{FRP}(\text{Retr-only})$.
  > Predicted effect size: $\ge 0.15$ absolute gap at moderate budget.

- **Pre-registration timestamp.** Committed in the tmc_paper.tex draft
  that predates Stage 2 execution (project path and commit history are in
  this repo; absolute timestamp: on or before 2026-04-18, when Stage 2 was
  first run — see `pilot/stage2/results.json` metadata).

- **Measured outcome.** H1 is **reversed at large magnitude**. On the same
  cases Summ-only's FRP is 0.75 (moderate) versus Retr-only's 0.05
  (moderate), a $+0.70$ gap in the opposite direction, Bonferroni-significant
  at $p < 0.01$ on the 10-case paired comparison.

- **Interpretation.** Capable modern summarizers, when prompted to preserve
  numeric anchors, do so more reliably than a BM25-indexed chunk store when
  the decision-time query has weak lexical overlap with the fact chunks.
  The precision loss we predicted from summarization does not appear at this
  scale with this model family.

## H2 (coherence loss — **directionally supported**)

- **Pre-registered prediction.** Retrieval will fail Reasoning Coherence
  Score (RCS) relative to summarization by at least 0.10 absolute at
  moderate budget, because inferences are not naturally indexable.
- **Measured outcome.** $\text{RCS}(\text{Summ-only}) = 0.62$,
  $\text{RCS}(\text{Retr-only}) = 0.21$ at moderate budget. Direction
  confirmed; effect size ($+0.41$) exceeds the predicted threshold.

## H3 (TMC dominance — **rejected**)

- **Pre-registered prediction.** Typed Memory Consolidation (TMC) will
  achieve higher End-to-end Decision Accuracy (EDA) than both Summ-only
  and Retr-only at the same budget, with $\ge 0.10$ absolute gap versus
  the stronger of the two baselines.
- **Measured outcome.** TMC EDA is dominated by Summ-only EDA at every
  budget, by $-0.20$ to $-0.70$. The strict gate on H3 is rejected.
- **Interpretation.** Routing facts to a BM25 retriever is the load-bearing
  failure of TMC; typed routing does not help when the retrieval leg is
  itself upstream-broken for numeric anchors.

## SAM sprint gate (Stage 3 — **failed**)

- Pre-registered gate: SAM must be within $-0.05$ of Summ-only on EDA, FRP,
  and RCS. SAM fell short by $-0.20$ to $-0.40$. Stopped at sprint per the
  pre-committed stop condition.

## DPM sprint gate (Stage 4 — **failed via tie**)

- Pre-registered gate: DPM must beat Summ-only by $\ge 0.05$ on at least
  two of $\{\text{EDA}, \text{RCS}, \text{CRR}\}$. DPM tied Summ-only
  exactly on all three metrics on the sprint ($n=5$). Pre-committed
  response: stop at sprint, reframe paper around enterprise properties.
- A follow-up extended DPM run ($n=10$ at tight and loose budgets,
  `pilot/stage4/extended_results.json`) does show DPM winning on the tight
  budget (FRP $+0.52$, RCS $+0.53$, EDA $+0.50$, CRR $+0.50$, Cohen's
  $h > 1.1$ on all four). The sprint-gate failure at moderate held; the
  extended tight-regime result is reported as a budget-dependent
  dissociation in §6.5 of the paper.

## VM / CAR (Stage 5 — new axis, over-abstaining)

- No gate on VM; the contribution is the CAR metric axis.
- Single-point VM result (case-matched, after fixing a positional-pairing
  bug in `pilot/run_vm.py` that had originally inflated the commit-rate
  reading): commit_rate $= 0.20$, conditional_accuracy $= 1.00$,
  commit_all_accuracy $= 0.20$. Over-abstaining relative to Summ-only's
  $(1.00, 1.00, 1.00)$ — VM trades coverage but conserves accuracy on
  every commit.
- Follow-up three-point strictness sweep (`pilot/stage5/sweep_results.json`,
  also re-derived after the same pairing fix in `pilot/run_vm_sweep.py`):
  commit rate is monotonic in permissiveness ($0.30 \to 0.60 \to 0.70$);
  conditional accuracy is conserved at $1.00$ at all three points; the
  curve sits on Summ-only's horizontal rather than above it because the
  10-case slice has no hard subset on which VM's check could demonstrate
  selective-abstention value. A denser sweep on a partially-correct base
  condition is named as future work.

## What was not pre-registered

- Sonnet judge-split validation (post-hoc robustness check, Step 5 of the
  Stage 5 hand-off). Reported in Section 9 (Limitations) as a robustness
  diagnostic, not as a hypothesis test.
- The TAMS decision rule is an analytical synthesis over the five stages'
  measurements, derived after the data were in; it is not a pre-registered
  prediction.

## Repository artifacts

- `alignment_paper.tex` / `alignment_paper.pdf` — the published paper.
- `tmc_paper.tex` — the earlier draft with the original typed-routing
  framing, retained for provenance.
- `pilot/stage2/results.json` — 150 baseline runs.
- `pilot/stage3/sprint_results.json` — SAM sprint.
- `pilot/stage4/sprint_results.json`, `extended_results.json`,
  `dpm_stats.json` — DPM sprint, extended, and paired statistics.
- `pilot/stage5/vm_results.json`, `sweep_results.json` — VM single point
  and strictness sweep.
- `pilot/stage2/bootstrap_mcnemar.json` — Stage 2 pairwise statistics.
- `pilot/stage2/judge_split_results.json`, `judge_split_stats.json` — Haiku
  vs Sonnet judge-split over Stage 2's committed outputs.

All statistical artifacts use fixed RNG seed `20260420`.

## Pre-registration vs post-hoc distinction

The three prose hypotheses (H1, H2, H3) are pre-registered; their measured
reversal on H1, confirmation on H2, and rejection on H3 are the paper's
falsifiable content. Every other contribution of the paper (the CRR metric,
the CAR axis, the DPM architecture, the TAMS rule, the enterprise-gap
framing) is post-hoc relative to these hypotheses but pre-registered
relative to the experiments that measure them (e.g., DPM was specified in
`pilot/conditions.py` before `pilot/run_dpm.py` executed it, and the
Stage 4 strict gate was pre-committed in the Stage-4 hand-off prompt).
