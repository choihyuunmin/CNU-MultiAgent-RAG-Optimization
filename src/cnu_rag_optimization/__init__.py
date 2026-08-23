"""CNU multi-agent RAG inference optimization primitives."""

from .budget import cap_comparison_documents, compact_documents, llm_options
from .confidence import (
    ConfidenceRoutingDecision,
    ConfidenceRoutingFeatures,
    decide_confident_route,
)
from .config import OptimizationPolicy, SelectorScope, TokenBudgets
from .evidence import EvidenceRoutingDecision, decide_evidence_route
from .hedged_stream import HedgeLimiter, stream_with_tail_hedge
from .parallel import parallel_enrich
from .regression import RegressionMetrics, compare_regression_records
from .routing import QueryFeatures, RoutingDecision, route_query, should_use_selector
from .selector import SelectionResult, select_ranked_documents
from .singleflight import AsyncSingleFlight, SingleFlightResult
from .telemetry import LLMCallMetrics, measure_async_call
from .transport import HTTPTransportPolicy, build_async_client
from .typed_dispatch import TypedDispatchResult, try_typed_single_tool_dispatch
from .verified import (
    SpeculativeDraft,
    VerifiedSpeculationResult,
    run_verified_speculation,
)

__all__ = [
    "LLMCallMetrics",
    "HTTPTransportPolicy",
    "ConfidenceRoutingDecision",
    "ConfidenceRoutingFeatures",
    "EvidenceRoutingDecision",
    "HedgeLimiter",
    "OptimizationPolicy",
    "QueryFeatures",
    "RegressionMetrics",
    "RoutingDecision",
    "SelectionResult",
    "SelectorScope",
    "SingleFlightResult",
    "SpeculativeDraft",
    "TokenBudgets",
    "TypedDispatchResult",
    "VerifiedSpeculationResult",
    "AsyncSingleFlight",
    "build_async_client",
    "cap_comparison_documents",
    "compare_regression_records",
    "compact_documents",
    "decide_confident_route",
    "decide_evidence_route",
    "llm_options",
    "measure_async_call",
    "parallel_enrich",
    "route_query",
    "run_verified_speculation",
    "select_ranked_documents",
    "should_use_selector",
    "stream_with_tail_hedge",
    "try_typed_single_tool_dispatch",
]
