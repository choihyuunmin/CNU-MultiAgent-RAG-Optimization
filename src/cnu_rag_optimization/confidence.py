"""Fail-closed local routing for unambiguous retrieval requests."""

from __future__ import annotations

from dataclasses import dataclass

from .config import OptimizationPolicy


@dataclass(frozen=True)
class ConfidenceRoutingFeatures:
    """Provider- and domain-neutral evidence extracted by an application.

    Integrators decide what counts as an entity, domain term, action term, or
    excluded intent. This package only enforces conservative routing gates.
    """

    candidate_route: str | None
    candidate_count: int
    entity_count: int
    domain_term_matched: bool
    action_term_matched: bool
    excluded_intent: bool = False
    ambiguous: bool = False
    has_prior_context: bool = False
    contextual_reference: bool = False


@dataclass(frozen=True)
class ConfidenceRoutingDecision:
    route: str | None
    use_local_route: bool
    reason: str


def decide_confident_route(
    features: ConfidenceRoutingFeatures,
    policy: OptimizationPolicy = OptimizationPolicy(),
) -> ConfidenceRoutingDecision:
    """Use local route only when every conservative gate passes.

    Any missing or conflicting evidence falls back to integrating system's
    existing LLM router. No query is silently assigned on partial evidence.
    """
    if not policy.confidence_routing_enabled:
        return ConfidenceRoutingDecision(None, False, "feature_disabled")
    if not features.candidate_route or features.candidate_count != 1:
        return ConfidenceRoutingDecision(None, False, "route_not_unique")
    if features.entity_count < 1:
        return ConfidenceRoutingDecision(None, False, "entity_missing")
    if not features.domain_term_matched:
        return ConfidenceRoutingDecision(None, False, "domain_term_missing")
    if not features.action_term_matched:
        return ConfidenceRoutingDecision(None, False, "action_term_missing")
    if features.excluded_intent:
        return ConfidenceRoutingDecision(None, False, "excluded_intent")
    if features.ambiguous:
        return ConfidenceRoutingDecision(None, False, "ambiguous")
    if features.has_prior_context or features.contextual_reference:
        return ConfidenceRoutingDecision(None, False, "context_required")
    return ConfidenceRoutingDecision(
        features.candidate_route,
        True,
        "all_confidence_gates_passed",
    )
