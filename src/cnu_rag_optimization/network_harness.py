"""Network-inspired execution harness; no model, prompt, or answer ownership.

Receiver credits and question-level round-robin live in CompletionRouter.
The bounded relay separates upstream reading from downstream consumption while
retaining every chunk, error and cancellation. It performs no speculative calls,
packet dropping, answer caching, or inference-engine configuration changes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from .completion import CompletionRouter

T = TypeVar("T")


@dataclass
class _Item(Generic[T]):
    value: T


@dataclass
class _Failure:
    error: BaseException


class NetworkHarness:
    """Reuse a router and move stream ownership to a bounded producer task.

    The supplied source supports aclose() and owns its admission slot, as the
    original app does.
    A producer drains that unchanged source; the slot is released on source end,
    not on final downstream consumption. With a full buffer the producer waits:
    bounded memory is preferred to unbounded prefetch. Bounds count chunks, not
    bytes. A consumer and a blocked producer may each additionally hold one item.
    """

    def __init__(self, router: CompletionRouter, *, buffer_chunks: int = 64,
                 on_event: Callable[[dict[str, object]], None] | None = None):
        if type(buffer_chunks) is not int or buffer_chunks < 1:
            raise ValueError("positive integer buffer_chunks required")
        self.router = router
        self.buffer_chunks = buffer_chunks
        self.on_event = on_event
        self.observer_errors = 0

    def slot(self, **metadata):
        return self.router.slot(**metadata)

    def _emit(self, event):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                self.observer_errors += 1

    @asynccontextmanager
    async def relay(self, source: Callable[[], AsyncIterator[T]], *, root_id: str,
                    stage: str) -> AsyncIterator[AsyncIterator[T]]:
        queue: asyncio.Queue = asyncio.Queue(self.buffer_chunks)
        end = object()
        started = time.monotonic()
        peak, chunks, blocked_ms = 0, 0, 0.0
        exhausted = False
        closing = False

        async def put(value):
            nonlocal peak, blocked_ms
            blocked = queue.full()
            before = time.monotonic()
            await queue.put(value)
            if blocked:
                blocked_ms += (time.monotonic() - before) * 1000
            peak = max(peak, queue.qsize())

        async def produce():
            nonlocal chunks
            success = False
            try:
                async with aclosing(source()) as upstream:
                    async for chunk in upstream:
                        chunks += 1
                        await put(_Item(chunk))
                success = True
            except asyncio.CancelledError as exc:
                # Do not enqueue a terminal item into a full abandoned buffer.
                if closing:
                    raise
                # An upstream may cancel its own task. Wake the consumer with
                # that exact error instead of silently leaving it on queue.get().
                await put(_Failure(exc))
            except Exception as exc:
                await put(_Failure(exc))
            else:
                await put(end)
            finally:
                self._emit({"event": "upstream_closed", "root_id": root_id, "stage": stage,
                            "success": success, "chunks_read": chunks, "buffer_capacity": self.buffer_chunks,
                            "buffer_peak_chunks": peak, "buffer_blocked_ms": blocked_ms,
                            "producer_elapsed_ms": (time.monotonic() - started) * 1000})

        async def consume():
            nonlocal exhausted
            while True:
                item = await queue.get()
                if item is end:
                    exhausted = True
                    return
                if isinstance(item, _Failure):
                    raise item.error
                yield item.value

        producer = asyncio.create_task(produce(), name="cnu-network-harness-upstream")
        try:
            async with aclosing(consume()) as output:
                yield output
        finally:
            closing = True
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass
            self._emit({"event": "relay_closed", "root_id": root_id, "stage": stage,
                        "fully_consumed": exhausted, "buffer_peak_chunks": peak,
                        "elapsed_ms": (time.monotonic() - started) * 1000})
