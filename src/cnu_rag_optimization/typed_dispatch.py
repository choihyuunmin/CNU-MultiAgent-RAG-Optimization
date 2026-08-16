"""Typed single-tool dispatch that removes redundant LLM tool echo calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from .config import OptimizationPolicy


T = TypeVar("T")
ToolHandler = Callable[[dict[str, object]], Awaitable[T]]
ArgumentValidator = Callable[[Mapping[str, object]], bool]


@dataclass(frozen=True)
class TypedDispatchResult(Generic[T]):
    dispatched: bool
    value: T | None
    reason: str


async def try_typed_single_tool_dispatch(
    *,
    available_tools: Mapping[str, ToolHandler[T]],
    candidate_tool: str | None,
    arguments: Mapping[str, object] | None,
    argument_validator: ArgumentValidator | None = None,
    policy: OptimizationPolicy = OptimizationPolicy(),
) -> TypedDispatchResult[T]:
    """Dispatch only one preselected tool with complete validated arguments.

    Rejected cases return ``dispatched=False`` so caller can invoke existing LLM
    tool router. Tool execution exceptions propagate: retry policy stays with
    integrating application and potentially stateful calls are never duplicated.
    """
    if not policy.typed_dispatch_enabled:
        return TypedDispatchResult(False, None, "feature_disabled")
    if len(available_tools) != 1:
        return TypedDispatchResult(False, None, "tool_not_unique")

    only_tool, handler = next(iter(available_tools.items()))
    if not candidate_tool or candidate_tool != only_tool:
        return TypedDispatchResult(False, None, "candidate_mismatch")
    if arguments is None:
        return TypedDispatchResult(False, None, "arguments_missing")
    if argument_validator is not None:
        try:
            valid = argument_validator(arguments)
        except Exception:
            valid = False
        if not valid:
            return TypedDispatchResult(False, None, "arguments_invalid")

    value = await handler(dict(arguments))
    return TypedDispatchResult(True, value, "typed_dispatch")
