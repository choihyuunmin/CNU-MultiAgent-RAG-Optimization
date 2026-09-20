from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import AsyncIterator, Awaitable, Callable

from .structured_harness import strict_json


@dataclass(frozen=True)
class CallReady:
    index: int
    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class RoundEnd:
    count: int


@dataclass(frozen=True)
class ToolBinding:
    handler: Callable[[dict], Awaitable[str]]
    # Application assertions, never inferred from a tool name. Reads must use
    # the same pinned view; early execution must not bypass approval or guards.
    independent_snapshot_read: bool = False


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str


class StreamProtocolError(ValueError):
    pass


async def execute_tool_stream(
    source: AsyncIterator[CallReady | RoundEnd],
    bindings: dict[str, ToolBinding],
    *,
    mode: str = "stream",
    on_event: Callable[[dict], None] | None = None,
) -> list[ToolResult]:
    """Execute one round in serial, parallel-after-end, or streaming mode.

    Unknown/side-effecting tools form an ordering barrier. Early tasks finish
    before that barrier; subsequent calls run in original order after RoundEnd.
    Missing end, duplicate calls, upstream failure and cancellation fail the
    round and cancel/drain pending tools. No hidden retry or fallback exists.
    Caller supplies per-tool and whole-request deadlines as in its original app.
    Results become visible only after a validated round end, in original order.
    """
    if mode not in {"serial", "parallel", "stream"}:
        raise ValueError("unknown execution mode")
    started = time.perf_counter()
    calls, tasks, ids = [], {}, set()
    ended = barrier = False

    def emit(kind, **fields):
        if on_event is not None:
            try:
                on_event({"event": kind, "elapsed_ms": (time.perf_counter()-started)*1000,
                          **fields})
            except Exception:
                pass  # Observation must never change a tool result.

    async def invoke(call, args):
        emit("tool_start", index=call.index)
        try:
            value = await bindings[call.name].handler(args)
            return ToolResult(call.call_id, call.name, value)
        finally:
            emit("tool_end", index=call.index)

    def start(call, args):
        tasks[call.index] = asyncio.create_task(invoke(call, args))

    try:
        async for event in source:
            if ended:
                raise StreamProtocolError("event after round end")
            if isinstance(event, RoundEnd):
                if type(event.count) is not int or event.count != len(calls):
                    raise StreamProtocolError("round call count mismatch")
                ended = True
                emit("round_end", count=event.count)
                continue
            if not isinstance(event, CallReady):
                raise StreamProtocolError("unsupported event")
            if (type(event.index) is not int or event.index != len(calls)
                    or not event.call_id or event.call_id in ids):
                raise StreamProtocolError("nonsequential index or duplicate/empty call ID")
            if event.name not in bindings:
                raise StreamProtocolError("tool is not registered")
            args = strict_json(event.arguments_json)
            if not isinstance(args, dict):
                raise StreamProtocolError("tool arguments must be a JSON object")
            ids.add(event.call_id)
            binding = bindings[event.name]
            barrier = barrier or not binding.independent_snapshot_read
            eligible = not barrier
            calls.append((event, args, eligible))
            emit("call_ready", index=event.index, early_eligible=eligible)
            if mode == "stream" and eligible:
                start(event, args)
        if not ended:
            raise StreamProtocolError("missing round end")
        if mode == "parallel":
            for call, args, eligible in calls:
                if eligible:
                    start(call, args)
        results = []
        for call, args, _ in calls:
            result = await tasks[call.index] if call.index in tasks else await invoke(call, args)
            results.append(result)
        emit("round_complete", count=len(results))
        return results
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        close = getattr(source, "aclose", None)
        if close is not None:
            await close()
