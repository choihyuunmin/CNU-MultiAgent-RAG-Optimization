"""Configuration for optimization-only integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SelectorScope(str, Enum):
    """Queries on which deterministic retrieval selection is enabled."""

    ALL = "all"
    MULTI_COUNTRY = "multi_country"


@dataclass(frozen=True)
class TokenBudgets:
    """Measured output limits used by each LLM stage."""

    preparation: int = 384
    selection: int = 256
    synthesis: int = 1536
    comparison: int = 2048

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def for_role(self, role: str) -> int | None:
        return {
            "preparation": self.preparation,
            "selection": self.selection,
            "synthesis": self.synthesis,
            "comparison": self.comparison,
        }.get(role)


@dataclass(frozen=True)
class OptimizationPolicy:
    """Feature flags kept independent from any production RAG implementation."""

    selector_enabled: bool = True
    selector_scope: SelectorScope = SelectorScope.ALL
    selector_top_k: int = 5
    token_budget_enabled: bool = True
    comparison_input_max_chars: int = 600
    fast_path_enabled: bool = True
    fast_path_max_chars: int = 120
    budgets: TokenBudgets = field(default_factory=TokenBudgets)

    def __post_init__(self) -> None:
        if self.selector_top_k < 1:
            raise ValueError("selector_top_k must be positive")
        if self.comparison_input_max_chars < 1:
            raise ValueError("comparison_input_max_chars must be positive")
        if self.fast_path_max_chars < 1:
            raise ValueError("fast_path_max_chars must be positive")
