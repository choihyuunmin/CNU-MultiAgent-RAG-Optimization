"""Conservative fast-path and selector-scope decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .config import OptimizationPolicy, SelectorScope


@dataclass(frozen=True)
class QueryFeatures:
    prompt_chars: int
    country_count: int
    direct_lookup: bool
    analysis_requested: bool = False
    has_prior_context: bool = False
    contextual_reference: bool = False
    forced_model: bool = False


@dataclass(frozen=True)
class RoutingDecision:
    complexity: str
    execution_path: str
    fast_path_eligible: bool
    blockers: tuple[str, ...]


def route_query(features: QueryFeatures, policy: OptimizationPolicy) -> RoutingDecision:
    """Route only unambiguous direct lookups to a single-RAG fast path."""
    if features.country_count >= 3 and features.analysis_requested:
        complexity = "L4"
    elif features.country_count >= 2:
        complexity = "L3"
    elif features.direct_lookup and not features.analysis_requested:
        complexity = "L1"
    else:
        complexity = "L2"

    blockers: list[str] = []
    if not features.direct_lookup:
        blockers.append("not_direct_lookup")
    if features.analysis_requested:
        blockers.append("analysis_requested")
    if features.country_count > 1:
        blockers.append("multi_country")
    if features.prompt_chars > policy.fast_path_max_chars:
        blockers.append("prompt_too_long")
    if features.has_prior_context:
        blockers.append("conversation_context")
    if features.contextual_reference:
        blockers.append("contextual_reference")
    if features.forced_model:
        blockers.append("forced_model")

    eligible = complexity == "L1" and not blockers
    path = "single_rag" if policy.fast_path_enabled and eligible else "multi_agent"
    return RoutingDecision(complexity, path, eligible, tuple(blockers))


def should_use_selector(country_count: int, policy: OptimizationPolicy) -> bool:
    if not policy.selector_enabled:
        return False
    if policy.selector_scope is SelectorScope.ALL:
        return True
    return country_count >= 2
