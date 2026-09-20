from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import json
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
    if argument_validator is None:
        return TypedDispatchResult(False, None, "validator_missing")
    # A shallow dict copy shares nested lists with other agents. Reject values
    # whose JSON round trip changes their type/value; validate a separate copy
    # so a validator cannot mutate the object that will actually be executed.
    try:
        snapshot = json.loads(json.dumps(dict(arguments), allow_nan=False))
        if snapshot != dict(arguments):
            return TypedDispatchResult(False, None, "arguments_not_lossless_json")
        validation_copy = json.loads(json.dumps(snapshot, allow_nan=False))
        valid = argument_validator(validation_copy)
        if valid is not True or validation_copy != snapshot:
            return TypedDispatchResult(False, None, "arguments_invalid")
    except Exception:
        return TypedDispatchResult(False, None, "arguments_invalid")

    value = await handler(snapshot)
    return TypedDispatchResult(True, value, "typed_dispatch")
