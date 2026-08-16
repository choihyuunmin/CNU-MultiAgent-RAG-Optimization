"""CNU multi-agent RAG inference optimization primitives."""

from .budget import cap_comparison_documents, compact_documents, llm_options
from .config import OptimizationPolicy, SelectorScope, TokenBudgets
from .parallel import parallel_enrich
from .routing import QueryFeatures, RoutingDecision, route_query, should_use_selector
from .selector import SelectionResult, select_ranked_documents
from .telemetry import LLMCallMetrics, measure_async_call

__all__ = [
    "LLMCallMetrics",
    "OptimizationPolicy",
    "QueryFeatures",
    "RoutingDecision",
    "SelectionResult",
    "SelectorScope",
    "TokenBudgets",
    "cap_comparison_documents",
    "compact_documents",
    "llm_options",
    "measure_async_call",
    "parallel_enrich",
    "route_query",
    "select_ranked_documents",
    "should_use_selector",
]
