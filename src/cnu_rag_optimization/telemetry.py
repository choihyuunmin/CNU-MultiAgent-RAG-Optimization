"""Privacy-preserving LLM timing without storing prompts or responses."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LLMCallMetrics(Generic[T]):
    result: T
    role: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


def _usage(result: object) -> tuple[int | None, int | None]:
    usage = getattr(result, "usage", None)
    if usage is None and isinstance(result, dict):
        usage = result.get("usage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


async def measure_async_call(
    role: str,
    call: Callable[[], Awaitable[T]],
) -> LLMCallMetrics[T]:
    """Measure one async LLM call. Prompt and generated text are never recorded."""
    started = time.perf_counter()
    result = await call()
    duration_ms = (time.perf_counter() - started) * 1000.0
    prompt_tokens, completion_tokens = _usage(result)
    return LLMCallMetrics(
        result=result,
        role=role,
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
