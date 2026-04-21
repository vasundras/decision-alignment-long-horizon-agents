"""Four memory conditions for the Stage 1 smoke test.

All conditions share the same LLM backend (the one passed to their __init__),
the same total token budget B, and the same summarizer/retriever primitives.
Only the routing path varies.

Budget enforcement:
  - Summ-only: the summarizer is asked to keep R under budget B.
  - Retr-only: oldest chunks are evicted when total retrieval-buffer size
    exceeds B.
  - TMC: facts live in F (append-only, capped at B/2), reasoning in R
    (summarized, capped at B/2).
  - Misrouted: facts are summarized (capped at B/2), reasoning is appended
    to a retrieval buffer (capped at B/2).

At decision time:
  - Summ-only: policy consumes R.
  - Retr-only: policy consumes top-k retrieved chunks.
  - TMC: policy consumes top-k facts + full R.
  - Misrouted: policy consumes top-k retrieved reasoning + full F-summary.
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import List, Protocol, Optional

from cases import Chunk


# -----------------------------------------------------------------------------
# LLM backend protocol (matches Anthropic SDK shape)
# -----------------------------------------------------------------------------

class LLMBackend(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


# -----------------------------------------------------------------------------
# Token accounting (approximation: 1 token ~= 4 chars)
# -----------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# -----------------------------------------------------------------------------
# Classifier (TMC's phi): regex fast path + fallback
# -----------------------------------------------------------------------------

FACT_PATTERNS = [
    re.compile(r"\$\d[\d,]*(\.\d{2})?"),      # currency
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),     # ISO date
    re.compile(r"\b\d+(\.\d+)?\s*%"),          # percentage
    re.compile(r"\b(POL|ID|W-|FICO)[\w-]*\b"), # identifier-ish
    re.compile(r"\b\d{3,}\b"),                 # any 3+ digit number
]

REASONING_CUES = [
    "because", "therefore", "we accepted", "we concluded", "establishes",
    "indicates", "asserts", "requested", "resolved", "causation", "explained",
    "letter of explanation", "analysis", "rationale", "decision note",
]


def classify_chunk(text: str) -> str:
    """Return 'fact' or 'reasoning'. Mixed content resolves to whichever signal is stronger."""
    fact_hits = sum(1 for p in FACT_PATTERNS if p.search(text))
    reasoning_hits = sum(1 for cue in REASONING_CUES if cue in text.lower())
    # If a chunk has both, use the stronger signal. Ties -> fact (conservative).
    if reasoning_hits > fact_hits:
        return "reasoning"
    return "fact"


# -----------------------------------------------------------------------------
# BM25-lite retriever (stdlib only, deterministic)
# -----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9\$\.\-]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.docs: List[str] = []
        self.tokens: List[List[str]] = []
        self.k1 = k1
        self.b = b

    def add(self, doc: str) -> None:
        self.docs.append(doc)
        self.tokens.append(_tokenize(doc))

    def _avg_dl(self) -> float:
        if not self.tokens:
            return 1.0
        return sum(len(t) for t in self.tokens) / len(self.tokens)

    def _idf(self, term: str) -> float:
        N = max(1, len(self.docs))
        df = sum(1 for toks in self.tokens if term in toks)
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> List[int]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.docs:
            return []
        avgdl = self._avg_dl()
        scores = []
        for i, toks in enumerate(self.tokens):
            dl = len(toks)
            score = 0.0
            tf_cache = {}
            for t in toks:
                tf_cache[t] = tf_cache.get(t, 0) + 1
            for q in q_tokens:
                tf = tf_cache.get(q, 0)
                if tf == 0:
                    continue
                idf = self._idf(q)
                score += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl)))
            scores.append((score, i))
        scores.sort(reverse=True)
        return [i for score, i in scores[:k] if score > 0]


# -----------------------------------------------------------------------------
# Summarizer primitive (LLM-backed)
# -----------------------------------------------------------------------------

SUMMARIZE_SYSTEM = (
    "You are a summarizer that maintains a running record of key information "
    "from a long-horizon agent trajectory. You will be given the current summary "
    "and a new piece of information. Return an updated summary that incorporates "
    "the new information while staying under the specified token budget. Preserve "
    "specific numeric facts (dollar amounts, dates, percentages) exactly as written."
)


def summarize_step(backend: LLMBackend, current_summary: str, new_chunk: str, budget_chars: int) -> str:
    """Integrate new_chunk into current_summary, keeping under budget_chars."""
    # For cheap pilot: if current + new fits, just append. Otherwise, call LLM to compress.
    candidate = (current_summary + "\n" + new_chunk).strip() if current_summary else new_chunk
    if len(candidate) <= budget_chars:
        return candidate
    user = (
        f"Current summary:\n{current_summary}\n\n"
        f"New information:\n{new_chunk}\n\n"
        f"Return an updated summary that stays under ~{budget_chars // 4} tokens "
        f"(~{budget_chars} chars). Preserve any dollar amounts, dates, and percentages exactly. "
        f"Remove reasoning detail before removing facts if forced to choose."
    )
    return backend.complete(SUMMARIZE_SYSTEM, user, max_tokens=budget_chars // 3).strip()


# -----------------------------------------------------------------------------
# Condition base class
# -----------------------------------------------------------------------------

@dataclass
class Memory:
    fact_store: List[Chunk] = field(default_factory=list)   # append-only
    reasoning_summary: str = ""
    retrieval_buffer: List[str] = field(default_factory=list)
    summarized_all: str = ""   # for Summ-only
    token_budget: int = 400     # char budget (chars ~= tokens*4 for rough control)

    @property
    def total_chars(self) -> int:
        return (
            sum(len(c.text) for c in self.fact_store)
            + len(self.reasoning_summary)
            + sum(len(x) for x in self.retrieval_buffer)
            + len(self.summarized_all)
        )


class Condition:
    name: str = "Abstract"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        self.backend = backend
        self.budget = budget_chars
        self.memory = Memory(token_budget=budget_chars)

    def consolidate(self, chunk: Chunk) -> None:
        raise NotImplementedError

    def present_at_decision(self, query: str) -> str:
        """Return the memory surface the policy consumes at decision time."""
        raise NotImplementedError


class SummOnly(Condition):
    name = "Summ-only"

    def consolidate(self, chunk: Chunk) -> None:
        self.memory.summarized_all = summarize_step(
            self.backend, self.memory.summarized_all, chunk.text, self.budget
        )

    def present_at_decision(self, query: str) -> str:
        return f"=== CONSOLIDATED SUMMARY ===\n{self.memory.summarized_all}"


class RetrOnly(Condition):
    name = "Retr-only"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        super().__init__(backend, budget_chars)
        self.index = BM25Index()

    def consolidate(self, chunk: Chunk) -> None:
        self.memory.retrieval_buffer.append(chunk.text)
        self.index.add(chunk.text)
        # Enforce budget: evict oldest until total size under budget
        while sum(len(x) for x in self.memory.retrieval_buffer) > self.budget and len(self.memory.retrieval_buffer) > 1:
            self.memory.retrieval_buffer.pop(0)
            # Rebuild index on eviction (simple + correct for pilot)
            self.index = BM25Index()
            for t in self.memory.retrieval_buffer:
                self.index.add(t)

    def present_at_decision(self, query: str) -> str:
        hits = self.index.search(query, k=8)
        retrieved = [self.memory.retrieval_buffer[i] for i in hits]
        return "=== RETRIEVED (top-8) ===\n" + "\n---\n".join(retrieved)


class TMC(Condition):
    name = "TMC"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        super().__init__(backend, budget_chars)
        self.index = BM25Index()

    def consolidate(self, chunk: Chunk) -> None:
        label = classify_chunk(chunk.text)
        if label == "fact":
            self.memory.fact_store.append(chunk)
            self.index.add(chunk.text)
            # Evict oldest facts if store exceeds half-budget
            half = self.budget // 2
            while sum(len(c.text) for c in self.memory.fact_store) > half and len(self.memory.fact_store) > 1:
                self.memory.fact_store.pop(0)
                self.index = BM25Index()
                for c in self.memory.fact_store:
                    self.index.add(c.text)
        else:
            self.memory.reasoning_summary = summarize_step(
                self.backend, self.memory.reasoning_summary, chunk.text, self.budget // 2
            )

    def present_at_decision(self, query: str) -> str:
        hits = self.index.search(query, k=8)
        facts = [self.memory.fact_store[i].text for i in hits]
        return (
            "=== RETRIEVED FACTS (top-8) ===\n" + "\n---\n".join(facts)
            + "\n\n=== REASONING SUMMARY ===\n" + self.memory.reasoning_summary
        )


class Misrouted(Condition):
    """Ablation: facts -> summarization, reasoning -> retrieval. Inverts TMC."""
    name = "Misrouted"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        super().__init__(backend, budget_chars)
        self.index = BM25Index()
        self.fact_summary = ""

    def consolidate(self, chunk: Chunk) -> None:
        label = classify_chunk(chunk.text)
        if label == "fact":
            # Misroute: summarize facts
            self.fact_summary = summarize_step(
                self.backend, self.fact_summary, chunk.text, self.budget // 2
            )
        else:
            # Misroute: retrieve over reasoning
            self.memory.retrieval_buffer.append(chunk.text)
            self.index.add(chunk.text)
            half = self.budget // 2
            while sum(len(x) for x in self.memory.retrieval_buffer) > half and len(self.memory.retrieval_buffer) > 1:
                self.memory.retrieval_buffer.pop(0)
                self.index = BM25Index()
                for t in self.memory.retrieval_buffer:
                    self.index.add(t)

    def present_at_decision(self, query: str) -> str:
        hits = self.index.search(query, k=8)
        retrieved_reasoning = [self.memory.retrieval_buffer[i] for i in hits]
        return (
            "=== FACT SUMMARY ===\n" + self.fact_summary
            + "\n\n=== RETRIEVED REASONING (top-8) ===\n" + "\n---\n".join(retrieved_reasoning)
        )


class TMCFull(TMC):
    """Variant of TMC: presents the full fact store (not top-8 retrieved) plus
    the full reasoning summary. Diagnoses whether Stage 1 TMC loss was an
    artifact of overly aggressive fact retrieval (top-k) rather than the
    typing-based routing itself.
    """
    name = "TMC-full"

    def present_at_decision(self, query: str) -> str:
        facts = [c.text for c in self.memory.fact_store]
        return (
            "=== FACT STORE (full) ===\n" + "\n---\n".join(facts)
            + "\n\n=== REASONING SUMMARY ===\n" + self.memory.reasoning_summary
        )


# -----------------------------------------------------------------------------
# SAM — Schema-Anchored Memory (Stage 3 proposal)
# -----------------------------------------------------------------------------
#
# SAM keeps a single structured JSON object rather than routing into multiple
# stores. Four slots are enforced by schema: facts (key-value), reasoning_chain
# (inference + basis + source_ref list), pending_questions, and decision_context.
#
# Consolidation is *holistic and lazy*: new events accumulate in a staging
# buffer; when total memory (schema + staging) would exceed budget, the
# summarizer is invoked once to re-synthesize the whole schema from its current
# state plus all staged events. This collapses per-trajectory LLM call count
# from O(N) (incremental summarize) to O(N / batch_size), which is both faster
# in wall clock and arguably better for quality because modern LLMs synthesize
# holistically more reliably than they edit incrementally.

import json

SAM_SCHEMA_SYSTEM = (
    "You maintain a structured JSON memory schema for a long-horizon enterprise agent. "
    "You will be given a current schema (JSON) and a list of new events. Return an "
    "updated JSON schema that integrates the new events while staying under the "
    "specified character budget.\n\n"
    "Schema slots (all four must be present; never omit a slot):\n"
    '  "facts": object mapping keys to exact string values. Preserve dollar amounts, '
    "dates, identifiers, percentages, and named entities verbatim.\n"
    '  "reasoning_chain": array of objects {"inference": str, "basis": str, '
    '"source_ref": str}. Each entry captures one inference drawn over prior events.\n'
    '  "pending_questions": array of strings describing unresolved items.\n'
    '  "decision_context": string describing the current sub-goal.\n\n'
    "When budget is tight, prefer dropping older pending_questions and summarizing "
    "basis text before dropping facts or reasoning_chain entries. Never drop a slot "
    "entirely; if nothing applies, use an empty object/array/string.\n\n"
    "Return ONLY the JSON object. No prose before or after."
)

_INITIAL_SCHEMA_JSON = json.dumps({
    "facts": {},
    "reasoning_chain": [],
    "pending_questions": [],
    "decision_context": "",
})


def synthesize_schema(
    backend: LLMBackend,
    current_schema_json: str,
    staged_events: List[str],
    budget_chars: int,
) -> str:
    """Re-synthesize the schema from its current state plus staged events.

    Returns a JSON string. If parsing fails, falls back to current_schema_json
    so the run does not crash.
    """
    events_block = "\n---\n".join(staged_events) if staged_events else "(none)"
    user = (
        f"Current schema:\n{current_schema_json}\n\n"
        f"New events to integrate:\n{events_block}\n\n"
        f"Return an updated JSON schema under ~{budget_chars} total characters. "
        f"Preserve the four-slot structure exactly."
    )
    # Cap output at budget_chars // 3 (roughly budget_chars/4 tokens worth of
    # slack for whitespace). Also capped at 1800 tokens absolute to respect
    # the organization's 10k output-tokens/min rate limit with a couple of
    # workers in flight. The schema rarely needs more than this.
    out_tokens = max(400, min(1800, budget_chars // 3))
    raw = backend.complete(SAM_SCHEMA_SYSTEM, user, max_tokens=out_tokens).strip()
    # Robust JSON extraction
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return current_schema_json
    candidate = m.group(0)
    try:
        parsed = json.loads(candidate)
        # Guarantee all four slots exist; repair if missing
        parsed.setdefault("facts", {})
        parsed.setdefault("reasoning_chain", [])
        parsed.setdefault("pending_questions", [])
        parsed.setdefault("decision_context", "")
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return current_schema_json


class SAM(Condition):
    """Schema-Anchored Memory: one typed JSON object, holistic lazy consolidation."""

    name = "SAM"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        super().__init__(backend, budget_chars)
        self.schema_json = _INITIAL_SCHEMA_JSON
        self.staging_buffer: List[str] = []

    def _total_size(self) -> int:
        return len(self.schema_json) + sum(len(x) for x in self.staging_buffer)

    def consolidate(self, chunk: Chunk) -> None:
        self.staging_buffer.append(chunk.text)
        # Lazy consolidation: only synthesize when we'd exceed budget.
        if self._total_size() > self.budget:
            self.schema_json = synthesize_schema(
                self.backend, self.schema_json, self.staging_buffer, self.budget
            )
            self.staging_buffer = []
            # Edge case: if schema alone still exceeds budget, synthesize again
            # empty-staged to force another compression pass.
            if len(self.schema_json) > self.budget:
                self.schema_json = synthesize_schema(
                    self.backend, self.schema_json, [], self.budget
                )

    def present_at_decision(self, query: str) -> str:
        # Final flush of any pending staged events before the decision.
        if self.staging_buffer:
            self.schema_json = synthesize_schema(
                self.backend, self.schema_json, self.staging_buffer, self.budget
            )
            self.staging_buffer = []
        return "=== SCHEMA-ANCHORED MEMORY ===\n" + self.schema_json


# -----------------------------------------------------------------------------
# DPM — Deterministic Projection Memory (Stage 4 proposal)
# -----------------------------------------------------------------------------
#
# DPM is architecturally distinct from every other condition in this file: it
# does no consolidation *during* the trajectory at all. Events accumulate in an
# append-only, immutable log. At decision time, a single task-conditioned LLM
# call projects the full log into a budget-bounded memory view. Then the
# decision policy consumes that view.
#
# Motivation (enterprise-framing). Stateful memory architectures (running
# summaries, belief states, graph stores) violate deterministic-replay, audit,
# and multi-tenant-isolation requirements that dominate regulated enterprise
# deployment. Treating memory as a pure function of (event_log, task_spec,
# budget) rather than as a process over time restores these properties:
#   - Deterministic replay: same inputs -> same memory (modulo LLM sampling,
#     mitigated by temperature=0).
#   - Audit: the memory IS the output of a named function; inspectable.
#   - Multi-tenant safety: no shared state between cases.
#   - Incident forensics: event log is a complete immutable record.
#
# Mechanistic prediction (versus Summ-only). Summ-only compresses N times
# across the trajectory; each step is lossy relative to the pre-compression
# state, and the compressor sees only the prior summary plus the new chunk.
# DPM compresses once, with the full event log visible, so information
# preservation is not subject to compounding incremental loss.


DPM_PROJECTION_SYSTEM = (
    "You are the memory projection layer for an enterprise decision agent. "
    "You will be given (a) the complete immutable event log for one case and "
    "(b) the decision task specification. Your job is to produce a budget-"
    "bounded memory view that is optimal for the decision task.\n\n"
    "Your projected memory must include, in this exact order with these exact "
    "section headings:\n\n"
    "== FACTS ==\n"
    "Bullet list of discrete verifiable claims relevant to the decision. "
    "Preserve dollar amounts, dates, percentages, policy provisions, and "
    "identifiers VERBATIM from the event log. Include the source event reference "
    "in parentheses when useful.\n\n"
    "== REASONING ==\n"
    "Short numbered list of inferences needed to reach the decision. Each entry "
    "states the inference and its basis in the event log. Preserve the logical "
    "structure that a compliant denial / adverse action notice would need.\n\n"
    "== COMPLIANCE NOTES ==\n"
    "Brief list of regulatory elements relevant to the decision (policy "
    "provisions cited, exclusion clauses applied, adverse-action rationale "
    "structure). Empty if not applicable.\n\n"
    "Stay under the specified character budget. Do not include JSON metadata, "
    "prefaces, or commentary outside these three sections. Do not summarize or "
    "abstract numeric values - preserve them exactly."
)


class DPM(Condition):
    """Deterministic Projection Memory: append-only log, single-shot projection at decision time."""

    name = "DPM"

    def __init__(self, backend: LLMBackend, budget_chars: int = 1600):
        super().__init__(backend, budget_chars)
        self.event_log: List[str] = []  # append-only, immutable once written

    def consolidate(self, chunk: Chunk) -> None:
        # No compression during the trajectory. Log and move on.
        # This is the single most important property of DPM: consolidation is
        # deferred entirely to decision time, where the projection has a global
        # view of the log.
        self.event_log.append(chunk.text)

    def present_at_decision(self, query: str) -> str:
        events_block = "\n---\n".join(self.event_log) if self.event_log else "(empty log)"
        user = (
            f"=== Event log (immutable, complete) ===\n{events_block}\n\n"
            f"=== Decision task ===\n{query}\n\n"
            f"=== Budget ===\n"
            f"The projected memory view must stay under {self.budget} characters "
            f"(~{self.budget // 4} tokens). Keep FACTS preserved verbatim; "
            f"compress REASONING and COMPLIANCE NOTES only as needed to fit.\n\n"
            f"Produce the projected memory view now."
        )
        # Use a max_tokens that accommodates the full budget. The backend is
        # expected to be run with temperature=0 for determinism; DPM does not
        # enforce this at the condition level because the backend abstraction
        # already models a single-parameter `complete(system, user, max_tokens)`.
        return self.backend.complete(
            DPM_PROJECTION_SYSTEM, user, max_tokens=max(500, self.budget // 3)
        )


# -----------------------------------------------------------------------------
# VM — Verified Memory (Stage 5 proposal)
# -----------------------------------------------------------------------------
#
# Core premise: agents should be able to abstain from deciding when memory is
# insufficient, not guess. VM = Summ-only + a decision-time completeness check.
# The agent self-interrogates before committing: given the memory and the
# required evidence schema for this task, is the evidence sufficient to decide
# confidently, or should I abstain and flag for human review?
#
# This is not a new storage architecture. It is a new READING architecture.
# The storage is Summ-only (empirically dominant). The addition is an
# interrogation step before commitment.
#
# VM introduces a new metric axis the benchmark has not measured: Calibrated
# Abstention Rate (CAR). On cases where VM commits, its accuracy should match
# Summ-only. On cases where VM abstains, we expect the abstentions to
# concentrate on ambiguous cases (those where Summ-only guesses wrong). A
# calibrated VM gains conditional accuracy at the cost of coverage — a tradeoff
# enterprise deployment actively wants (an underwriter who flags-for-review on
# ambiguity is more valuable than one who guesses).

VM_COMPLETENESS_SYSTEM = (
    "You are the completeness-check layer for an enterprise decision agent. "
    "Given (a) the consolidated memory for a case and (b) the decision task, "
    "decide whether the memory contains sufficient information to reach a "
    "confident, defensible decision that would survive regulatory audit. "
    "A memory is insufficient if: any required factual anchor is missing or "
    "ambiguous (e.g., an income figure that could be interpreted two ways, a "
    "date that is not cleanly recoverable, a policy provision cited but not "
    "quoted); or any reasoning step required for the decision is unresolved "
    "(e.g., an employment gap not explained, a causation question unresolved, "
    "a claimant correspondence unanswered).\n\n"
    "Return a single JSON object with two fields:\n"
    '  "sufficient": true or false\n'
    '  "gaps": array of short strings describing specific evidence gaps '
    "(empty array if sufficient)\n\n"
    "Return ONLY the JSON. No prose."
)


class VM(SummOnly):
    """Verified Memory: Summ-only storage plus a decision-time completeness check.

    The condition itself inherits consolidation from SummOnly (the empirical
    winner for memory storage). The only addition is a completeness_check
    method that the runner calls before invoking the decision agent. When the
    check returns insufficient, the runner records the decision as ABSTAIN
    rather than forcing a guess.
    """

    name = "VM"

    def completeness_check(self, task_prompt: str) -> Dict[str, object]:
        """Return {"sufficient": bool, "gaps": List[str]} for the current memory."""
        import json as _json
        memory = self.memory.summarized_all
        user = (
            f"=== Consolidated memory ===\n{memory}\n\n"
            f"=== Decision task ===\n{task_prompt}\n\n"
            f"Return the JSON object now."
        )
        raw = self.backend.complete(
            VM_COMPLETENESS_SYSTEM, user, max_tokens=400
        ).strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            # On parse failure, default to "sufficient" to avoid spurious
            # abstention; the runner can investigate parse errors separately.
            return {"sufficient": True, "gaps": ["(check parse error)"]}
        try:
            parsed = _json.loads(m.group(0))
            return {
                "sufficient": bool(parsed.get("sufficient", True)),
                "gaps": list(parsed.get("gaps", [])),
            }
        except Exception:
            return {"sufficient": True, "gaps": ["(check parse error)"]}


CONDITION_CLASSES = [TMC, SummOnly, RetrOnly, Misrouted]
CONDITION_CLASSES_STAGE2 = [TMC, TMCFull, SummOnly, RetrOnly, Misrouted]
CONDITION_CLASSES_STAGE3 = [SAM]    # compared against Stage 2's existing results.
CONDITION_CLASSES_STAGE4 = [DPM]    # compared against Stage 2's existing results.
CONDITION_CLASSES_STAGE5 = [VM]     # VM = Summ-only storage + completeness check.
