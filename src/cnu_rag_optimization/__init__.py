"""CNU multi-agent RAG inference optimization primitives."""

from .budget import cap_comparison_documents, compact_documents, llm_options
from .config import OptimizationPolicy, SelectorScope, TokenBudgets
from .parallel import parallel_enrich
from .regression import RegressionMetrics, compare_regression_records
from .routing import QueryFeatures, RoutingDecision, route_query, should_use_selector
from .selector import SelectionResult, select_ranked_documents
from .singleflight import AsyncSingleFlight, SingleFlightResult
from .telemetry import LLMCallMetrics, measure_async_call
from .transport import HTTPTransportPolicy, build_async_client
from .verified import (
    SpeculativeDraft,
    VerifiedSpeculationResult,
    run_verified_speculation,
)
from .vllm_profiles import VLLM_PROFILES, VLLMProfile

__all__ = [
    "LLMCallMetrics",
    "HTTPTransportPolicy",
    "OptimizationPolicy",
    "QueryFeatures",
    "RegressionMetrics",
    "RoutingDecision",
    "SelectionResult",
    "SelectorScope",
    "SingleFlightResult",
    "SpeculativeDraft",
    "TokenBudgets",
    "VLLMProfile",
    "VLLM_PROFILES",
    "VerifiedSpeculationResult",
    "AsyncSingleFlight",
    "build_async_client",
    "cap_comparison_documents",
    "compare_regression_records",
    "compact_documents",
    "llm_options",
    "measure_async_call",
    "parallel_enrich",
    "route_query",
    "run_verified_speculation",
    "select_ranked_documents",
    "should_use_selector",
]
