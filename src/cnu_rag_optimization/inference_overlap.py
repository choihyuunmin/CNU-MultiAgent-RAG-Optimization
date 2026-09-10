"""Unscheduled LLM observation and snapshot-verified read overlap.

No admission, semaphore, routing, prompt edits, LLM retries, or token truncation.
This hides independent read latency; it does not shorten model reasoning.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _count(value):
    return value if type(value) is int and value >= 0 else None


class _Phases:
    def __init__(self, clock):
        self.clock = clock
        self.started = clock()
        self.first_chunk_ms = self.first_reasoning_ms = self.last_reasoning_ms = None
        self.first_visible_ms = None
        self.reasoning_chars = self.chunks = 0
        self.prompt_tokens = self.completion_tokens = self.reasoning_tokens = None

    def observe(self, value, *, stream):
        now = (self.clock() - self.started) * 1000
        usage = _get(value, "usage")
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens"):
                count = _count(_get(usage, key))
                if count is not None:
                    setattr(self, key, count)
            details = _get(usage, "completion_tokens_details")
            count = _count(_get(details, "reasoning_tokens"))
            if count is not None:
                self.reasoning_tokens = count
        if stream:
            self.chunks += 1
            if self.first_chunk_ms is None:
                self.first_chunk_ms = now
        for choice in _get(value, "choices", []) or []:
            delta = _get(choice, "delta" if stream else "message")
            # Count provider-exposed fields transiently; never retain text.
            reasoning = next((text for name in ("reasoning_content", "reasoning", "thinking")
                              if isinstance(text := _get(delta, name), str) and text), "")
            self.reasoning_chars += len(reasoning)
            if reasoning and stream:
                if self.first_reasoning_ms is None:
                    self.first_reasoning_ms = now
                self.last_reasoning_ms = now
            visible = _get(delta, "content")
            if stream and ((isinstance(visible, str) and visible) or _get(delta, "tool_calls")):
                if self.first_visible_ms is None:
                    self.first_visible_ms = now

    def report(self):
        # With multiple choices the span covers all observed choices, not one chain.
        span = None if self.first_reasoning_ms is None else self.last_reasoning_ms - self.first_reasoning_ms
        inconsistent = (self.reasoning_tokens is not None and self.completion_tokens is not None
                        and self.reasoning_tokens > self.completion_tokens)
        return dict(api_ms=(self.clock() - self.started) * 1000,
                    first_chunk_ms=self.first_chunk_ms, first_visible_ms=self.first_visible_ms,
                    first_reasoning_ms=self.first_reasoning_ms, last_reasoning_ms=self.last_reasoning_ms,
                    observed_reasoning_span_ms=span, reasoning_chars=self.reasoning_chars,
                    chunks=self.chunks, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, reasoning_tokens=self.reasoning_tokens,
                    usage_inconsistent=inconsistent, raw_text_logged=False,
                    phase_source="client_observation_not_gpu_compute")


@dataclass(frozen=True)
class ReadContract:
    """Caller attests a pure read of the same pinned snapshot and access scope.

    arguments must encode every effective input exactly (including ordering,
    locale, projection and defaults). No normalization is performed. snapshot
    must identify an enforced immutable revision/transaction, not a timestamp
    guess. authorization is a non-secret tenant/permission-version identifier.
    These declarations cannot prove purity or snapshot enforcement themselves.
    """
    operation: str
    implementation: str
    authorization: str
    snapshot: str
    arguments: bytes
    read_only: bool = False
    deterministic: bool = False

    @property
    def eligible(self):
        return (all(isinstance(x, str) and x for x in
                    (self.operation, self.implementation, self.authorization, self.snapshot))
                and type(self.arguments) is bytes
                and self.read_only is True and self.deterministic is True)


class _PreparedRead:
    def __init__(self, contract, factory, adapter):
        self.contract, self.adapter = contract, adapter
        self.started = adapter.clock()
        self.finished = None
        self.consumed = self.closed = False
        self.task = asyncio.create_task(self._run(factory)) if contract.eligible else None

    async def _run(self, factory):
        try:
            return await factory()
        finally:
            self.finished = self.adapter.clock()

    async def _discard(self):
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def consume(self, actual: ReadContract, fallback: Callable[[], Awaitable[Any]]):
        if self.closed or self.consumed:
            raise RuntimeError("prepared read is request-local and single-use")
        self.consumed = True
        entered = self.adapter.clock()
        reason = "ineligible" if self.task is None else "contract_mismatch"
        if self.task is not None and actual.eligible and actual == self.contract:
            try:
                value = await self.task
            except asyncio.CancelledError:
                # Caller cancellation is never converted to a fallback request.
                raise
            except Exception:
                reason = "read_error"
            else:
                self.adapter.emit({"event": "read_overlap", "reused": True, "reason": "exact_contract",
                    "residual_wait_ms": (self.adapter.clock() - entered) * 1000,
                    "read_ms": (self.finished - self.started) * 1000,
                    "head_start_ms": (entered - self.started) * 1000})
                self.task = None
                return value
        await self._discard()
        self.adapter.emit({"event": "read_overlap", "reused": False, "reason": reason})
        return await fallback()


class InferenceOverlapAdapter:
    """Observe original API calls and explicitly overlap independent pure reads.

    on_event receives metadata only and must be a fast synchronous sink. Existing
    caller timeout/retry behavior remains outside this adapter. stream wraps an
    async generator, not a coroutine returning an SDK stream. No cross-request
    cache or automatic tool invocation exists.
    """
    def __init__(self, *, on_event=None, clock=time.perf_counter):
        self.on_event, self.clock = on_event, clock
        self.observer_errors = 0

    def emit(self, event):
        try:
            if self.on_event:
                self.on_event(event)
        except Exception:
            self.observer_errors += 1

    def _observe(self, phases, value, *, stream):
        try:
            phases.observe(value, stream=stream)
        except Exception:
            self.observer_errors += 1

    def completion(self, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            phases, call_id, status = _Phases(self.clock), uuid.uuid4().hex, "error"
            try:
                response = await original(*args, **kwargs)
                self._observe(phases, response, stream=False)
                status = "ok"
                return response
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            finally:
                self.emit({"event": "llm_phases", "call_id": call_id, "streaming": False,
                           "status": status, **phases.report()})
        return wrapped

    def stream(self, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            phases, call_id, status = _Phases(self.clock), uuid.uuid4().hex, "error"
            try:
                async with aclosing(original(*args, **kwargs)) as upstream:
                    async for chunk in upstream:
                        self._observe(phases, chunk, stream=True)
                        yield chunk
                status = "ok"
            except (asyncio.CancelledError, GeneratorExit):
                status = "cancelled"
                raise
            finally:
                self.emit({"event": "llm_phases", "call_id": call_id, "streaming": True,
                           "status": status, **phases.report()})
        return wrapped

    @asynccontextmanager
    async def prepare_read(self, contract: ReadContract, factory: Callable[[], Awaitable[Any]]):
        """Begin at a dependency-ready point, never from unverified CoT text.

        Factory must perform only the contract's pure I/O. It must propagate
        cancellation and retain existing backend timeouts. No LLM work or shared
        non-concurrent DB session may be passed here.
        """
        prepared = _PreparedRead(contract, factory, self)
        try:
            yield prepared
        finally:
            prepared.closed = True
            await prepared._discard()
