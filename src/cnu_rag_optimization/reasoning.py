from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time


class ReasoningStallError(TimeoutError):
    """The model exceeded its answer-free deadline or reasoning allowance."""


class IncompleteGenerationError(RuntimeError):
    """The stream ended without a complete answer or tool call."""


def fields(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return vars(value) if hasattr(value, "__dict__") else {}


@dataclass(frozen=True)
class ReasoningPolicy:
    # Engineering defaults, not a measured optimum or a quality guarantee.
    effort: str = "low"
    first_output_timeout_s: float = 60.0
    max_reasoning_characters: int = 32768

    def __post_init__(self):
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError("unsupported reasoning effort")
        if (isinstance(self.first_output_timeout_s, bool)
                or not math.isfinite(self.first_output_timeout_s)
                or self.first_output_timeout_s <= 0
                or type(self.max_reasoning_characters) is not int
                or self.max_reasoning_characters < 1):
            raise ValueError("reasoning bounds must be finite and positive")

    def payload(self, kwargs):
        """Use the existing proxy's extra_body passthrough; never edit prompts.

        Explicit per-call effort takes precedence over the configured default.
        The older app drops top-level reasoning_effort, so normalize that field.
        """
        result = dict(kwargs)
        extra = dict(result.get("extra_body") or {})
        effort = result.pop("reasoning_effort", None)
        if effort is not None:
            extra["reasoning_effort"] = effort
        elif "reasoning_effort" not in (extra.get("chat_template_kwargs") or {}):
            extra.setdefault("reasoning_effort", self.effort)
        result["extra_body"] = extra
        return result


@dataclass
class StreamProgress:
    reasoning_characters: int = 0
    reasoning_observed: bool = False
    visible_characters: int = 0
    first_chunk_s: float | None = None
    first_reasoning_s: float | None = None
    first_visible_s: float | None = None
    first_tool_s: float | None = None
    finished: bool = False
    truncated: bool = False
    stalled: bool = False

    @property
    def useful(self):
        return self.first_visible_s is not None or self.first_tool_s is not None

    def observe(self, chunk, elapsed):
        if self.first_chunk_s is None:
            self.first_chunk_s = elapsed
        for choice in fields(chunk).get("choices") or ():
            choice = fields(choice)
            delta = fields(choice.get("delta"))
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if isinstance(reasoning, str):
                self.reasoning_observed = True
                self.reasoning_characters += len(reasoning)
                if reasoning and self.first_reasoning_s is None:
                    self.first_reasoning_s = elapsed
            content = delta.get("content")
            if isinstance(content, str) and content:
                self.visible_characters += len(content)
                if self.first_visible_s is None:
                    self.first_visible_s = elapsed
            if delta.get("tool_calls") or delta.get("function_call"):
                if self.first_tool_s is None:
                    self.first_tool_s = elapsed
            finish = choice.get("finish_reason")
            self.finished |= finish in {"stop", "tool_calls", "function_call"}
            self.truncated |= finish in {"length", "content_filter"}

    def metrics(self):
        return {"reasoning_characters": self.reasoning_characters if self.reasoning_observed else None,
                "visible_characters": self.visible_characters,
                "first_chunk_s": self.first_chunk_s, "first_reasoning_s": self.first_reasoning_s,
                "first_visible_s": self.first_visible_s, "first_tool_s": self.first_tool_s,
                "stream_finished": self.finished, "stream_truncated": self.truncated,
                "reasoning_stalled": self.stalled}


async def monitor_stream(source, progress, policy=None):
    """A total first-output deadline, not a timeout renewed by reasoning tokens.

    Keep object identity/order and downstream backpressure. Close the source on
    errors and cancellation. Never retry after output or turn truncation into a
    successful answer. Usage-only terminal chunks are forwarded unchanged.
    """
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + policy.first_output_timeout_s if policy else None
    try:
        while True:
            deadline_scope = None
            try:
                if policy and not progress.useful:
                    if asyncio.get_running_loop().time() >= deadline:
                        progress.stalled = True
                        raise ReasoningStallError("model produced no answer or tool call before deadline")
                    deadline_scope = asyncio.timeout_at(deadline)
                    async with deadline_scope:
                        chunk = await anext(source)
                else:
                    chunk = await anext(source)
            except StopAsyncIteration:
                break
            except TimeoutError as exc:
                if isinstance(exc, ReasoningStallError):
                    raise
                if deadline_scope is None or not deadline_scope.expired():
                    raise
                progress.stalled = True
                raise ReasoningStallError("model produced no answer or tool call before deadline") from exc
            progress.observe(chunk, time.monotonic() - started)
            if policy and not progress.useful and progress.reasoning_characters > policy.max_reasoning_characters:
                progress.stalled = True
                raise ReasoningStallError("answer-free reasoning exceeded character allowance")
            yield chunk
        if policy and (not progress.useful or not progress.finished or progress.truncated):
            raise IncompleteGenerationError("model stream did not complete an answer or tool call")
    finally:
        await source.aclose()
