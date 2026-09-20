from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


def canonical_inputs(value) -> bytes:
    """Exact, order-stable encoding of every effective input (no normalization)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CallContract:
    """Caller attests which call is being overlapped and with which inputs.

    ``inputs`` must encode every effective input exactly (messages, response
    format, temperature, role and any option that changes the output). The
    caller must declare ``deterministic=True`` only for calls whose output is a
    pure function of those inputs, such as temperature-zero structured output.
    """
    operation: str
    implementation: str
    request_key: str
    inputs: bytes
    deterministic: bool = False

    @classmethod
    def build(cls, operation: str, implementation: str, request_key: str, inputs: Any, *,
              deterministic: bool) -> "CallContract":
        return cls(operation, implementation, request_key, canonical_inputs(inputs), deterministic)

    @property
    def eligible(self) -> bool:
        return (all(isinstance(x, str) and x for x in (self.operation, self.implementation, self.request_key))
                and type(self.inputs) is bytes and self.deterministic is True)


class _PreparedCall:
    def __init__(self, contract: CallContract, factory, clock):
        self.contract = contract
        self.started = clock()
        self.finished = None
        self.clock = clock
        self.task = asyncio.create_task(self._run(factory))

    async def _run(self, factory):
        try:
            return await factory()
        finally:
            self.finished = self.clock()

    async def discard(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class CallOverlapAdapter:
    """Request-local registry of prepared calls with exact-contract consumption.

    ``on_event`` receives metadata only (no prompts or outputs). ``max_pending``
    bounds prepared work across requests; ``stale_seconds`` discards prepared
    calls that were never consumed, so a request that took another route cannot
    leak work.
    """

    def __init__(self, *, on_event=None, clock=time.perf_counter, max_pending: int = 64,
                 stale_seconds: float = 300.0):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if not (stale_seconds > 0):
            raise ValueError("stale_seconds must be positive")
        self.on_event = on_event
        self.clock = clock
        self.max_pending = max_pending
        self.stale_seconds = stale_seconds
        self._pending: dict[tuple[str, str], _PreparedCall] = {}

    def emit(self, event):
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            pass  # Tracing must not change application behavior.

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def _purge_stale(self):
        now = self.clock()
        stale = [k for k, p in self._pending.items() if now - p.started > self.stale_seconds]
        for key in stale:
            await self._drop(key, "stale")

    async def _drop(self, key, reason):
        prepared = self._pending.pop(key, None)
        if prepared is None:
            return
        wasted = prepared.task.done() and not prepared.task.cancelled()
        await prepared.discard()
        self.emit({"event": "call_overlap", "operation": key[1], "request_key": key[0],
                   "discarded": True, "reason": reason, "completed_before_discard": wasted,
                   "call_ms": None if prepared.finished is None else (prepared.finished - prepared.started) * 1000})

    def prepare(self, contract: CallContract, factory: Callable[[], Awaitable[Any]]) -> bool:
        """Start the call now. Returns False (and starts nothing) when ineligible,
        already prepared for this request and operation, or at capacity."""
        key = (contract.request_key, contract.operation)
        reason = None
        if not contract.eligible:
            reason = "ineligible"
        elif key in self._pending:
            reason = "already_prepared"
        elif len(self._pending) >= self.max_pending:
            reason = "capacity"
        if reason is not None:
            self.emit({"event": "call_overlap", "operation": contract.operation,
                       "request_key": contract.request_key, "prepared": False, "reason": reason})
            return False
        self._pending[key] = _PreparedCall(contract, factory, self.clock)
        self.emit({"event": "call_overlap", "operation": contract.operation,
                   "request_key": contract.request_key, "prepared": True, "reason": "dependency_ready"})
        return True

    async def consume(self, contract: CallContract, fallback: Callable[[], Awaitable[Any]]):
        """Return the prepared result only for an exactly matching contract; otherwise
        discard any prepared work for this operation and run ``fallback``."""
        await self._purge_stale()
        key = (contract.request_key, contract.operation)
        prepared = self._pending.pop(key, None)
        entered = self.clock()
        reason = "not_prepared"
        if prepared is not None:
            if contract.eligible and contract == prepared.contract:
                try:
                    value = await prepared.task
                except asyncio.CancelledError:
                    raise  # The consumer's own cancellation is never turned into a fallback call.
                except Exception:
                    reason = "call_error"
                else:
                    self.emit({"event": "call_overlap", "operation": contract.operation,
                               "request_key": contract.request_key, "reused": True, "reason": "exact_contract",
                               "head_start_ms": (entered - prepared.started) * 1000,
                               "call_ms": (prepared.finished - prepared.started) * 1000,
                               "residual_wait_ms": (self.clock() - entered) * 1000})
                    return value
            else:
                reason = "contract_mismatch"
                await prepared.discard()
        self.emit({"event": "call_overlap", "operation": contract.operation,
                   "request_key": contract.request_key, "reused": False, "reason": reason})
        return await fallback()

    async def close(self, request_key: str) -> int:
        """Cancel and drop every prepared call of one request. Returns how many were dropped."""
        keys = [k for k in self._pending if k[0] == request_key]
        for key in keys:
            await self._drop(key, "request_closed")
        return len(keys)
