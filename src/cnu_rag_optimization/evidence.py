"""Fail-closed evidence gate for deterministic critical-path routing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣_/·ㆍ-]{1,}")


@dataclass(frozen=True)
class EvidenceRoutingDecision:
    use_deterministic_selector: bool
    reason: str
    confidence: float
    candidate_count: int
    unique_parent_count: int
    scope_match_ratio: float
    keyword_coverage: float
    top_score: float | None


def _first(candidate: Mapping[str, object], keys: Sequence[str]) -> str:
    return next(
        (str(candidate.get(key) or "").strip() for key in keys if candidate.get(key)),
        "",
    )


def _score(candidate: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        raw = candidate.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _terms(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        for token in _TOKEN_RE.findall(str(value or "").lower()):
            if token not in seen:
                seen.add(token)
                output.append(token)
    return tuple(output)


def decide_evidence_route(
    candidates: Sequence[Mapping[str, object]],
    *,
    expected_scope: str = "",
    query_terms: Sequence[str] = (),
    min_top_score: float = 0.5,
    min_keyword_coverage: float = 0.2,
    min_scope_match_ratio: float = 1.0,
    max_unique_parents: int = 1,
    id_keys: tuple[str, ...] = ("id", "item_id", "document_id"),
    parent_keys: tuple[str, ...] = ("parent_id", "document_id"),
    scope_keys: tuple[str, ...] = ("scope", "source", "country", "tenant"),
    score_keys: tuple[str, ...] = ("relevance_score", "score", "_score"),
    text_keys: tuple[str, ...] = ("title", "subject", "content", "text"),
) -> EvidenceRoutingDecision:
    """Skip LLM validation only when observable retrieval evidence converges.

    Missing IDs, split parent documents, scope mismatch, weak score, or weak
    lexical coverage returns ``use_deterministic_selector=False``. Caller then
    keeps its authoritative LLM validation path.
    """
    usable = [candidate for candidate in candidates if _first(candidate, id_keys)]
    parents = {
        _first(candidate, parent_keys) or _first(candidate, id_keys).split("_", 1)[0]
        for candidate in usable
    }
    parents.discard("")
    scopes = [_first(candidate, scope_keys) for candidate in usable]
    scopes = [scope for scope in scopes if scope]
    expected = str(expected_scope or "").strip()
    if expected and scopes:
        scope_ratio = sum(scope == expected for scope in scopes) / len(scopes)
    elif expected:
        scope_ratio = 0.0
    else:
        scope_ratio = 1.0

    terms = _terms(query_terms)
    evidence = " ".join(
        str(candidate.get(key) or "").lower()
        for candidate in usable[:5]
        for key in text_keys
    )
    coverage = sum(term in evidence for term in terms) / len(terms) if terms else 0.0
    scores = [
        value
        for candidate in usable
        if (value := _score(candidate, score_keys)) is not None
    ]
    top_score = max(scores) if scores else None

    blockers: list[str] = []
    if not usable:
        blockers.append("no_candidates")
    if not parents:
        blockers.append("missing_parent_id")
    elif len(parents) > max(1, int(max_unique_parents)):
        blockers.append("parent_divergence")
    if scope_ratio < min_scope_match_ratio:
        blockers.append("scope_mismatch")
    if top_score is None:
        blockers.append("missing_score")
    elif top_score < min_top_score:
        blockers.append("weak_score")
    if coverage < min_keyword_coverage:
        blockers.append("weak_keyword_coverage")

    score_confidence = 0.0 if top_score is None else min(1.0, max(0.0, top_score))
    return EvidenceRoutingDecision(
        use_deterministic_selector=not blockers,
        reason="evidence_converged" if not blockers else ",".join(blockers),
        confidence=min(scope_ratio, coverage, score_confidence),
        candidate_count=len(usable),
        unique_parent_count=len(parents),
        scope_match_ratio=scope_ratio,
        keyword_coverage=coverage,
        top_score=top_score,
    )
