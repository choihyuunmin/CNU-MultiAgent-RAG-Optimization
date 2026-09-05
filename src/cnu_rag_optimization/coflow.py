"""Receiver-credit and visible-coflow admission, independent of LLM providers.

The caller supplies resource identity (aliases sharing a receiver MUST map to
the same identity), root request ID, and a non-content work class. Only dispatch
time changes. No payload is accepted, inspected, rewritten, cached, or retried.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import AsyncIterator, Callable


@dataclass(frozen=True)
class CoflowPolicy:
    order: str = "coflow"
    initial_window: int = 4
    min_window: int = 2
    max_window: int = 8
    adaptive: bool = True
    aging_seconds: float = 10.0
    max_wait_seconds: float = 30.0
    feedback_gain: float = 0.2
    default_service_ms: float = 3000.0

    def __post_init__(self) -> None:
        if self.order not in {"fifo", "coflow"}:
            raise ValueError("order must be fifo or coflow")
        if not 1 <= self.min_window <= self.initial_window <= self.max_window:
            raise ValueError("invalid admission window bounds")
        if any(not math.isfinite(x) or x <= 0 for x in (
            self.aging_seconds, self.max_wait_seconds, self.default_service_ms,
        )):
            raise ValueError("timing values must be finite and positive")
        if not 0 < self.feedback_gain <= 1:
            raise ValueError("feedback_gain must be in (0, 1]")


@dataclass
class AdmissionTicket:
    sequence: int
    resource: str
    root_id: str
    work_class: str
    queued_at: float
    estimate_ms: float
    future: asyncio.Future[None]
    started_at: float | None = None
    released: bool = False
    window_at_start: int = 0
    pending_at_start: int = 0

    @property
    def wait_ms(self) -> float:
        end = self.started_at if self.started_at is not None else self.queued_at
        return max(0.0, (end - self.queued_at) * 1000.0)


@dataclass
class _Receiver:
    window: float
    active: dict[int, AdmissionTicket] = field(default_factory=dict)
    pending: list[AdmissionTicket] = field(default_factory=list)
    congestion: float = 0.0
    quiet_samples: int = 0
    feedback_samples: int = 0


class CoflowAdmission:
    """Single-event-loop admission with receiver-wide credits.

    Visible coflow cost is the bottleneck outstanding work across receivers.
    This is an online heuristic, not an oracle for future agent branches. Old
    queued calls get FIFO priority after max_wait_seconds; active calls remain
    non-preemptive. Loss of telemetry freezes the last bounded credit window.
    """

    def __init__(self, policy: CoflowPolicy = CoflowPolicy(), *, clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self.clock = clock
        self._sequence = count()
        self._receivers: dict[str, _Receiver] = {}
        self._samples: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=32))

    def _receiver(self, resource: str) -> _Receiver:
        return self._receivers.setdefault(resource, _Receiver(float(self.policy.initial_window)))

    def _limit(self, receiver: _Receiver) -> int:
        return max(self.policy.min_window, min(self.policy.max_window, math.floor(receiver.window)))

    def _estimate(self, resource: str, work_class: str) -> float:
        samples = sorted(self._samples[(resource, work_class)])
        return samples[len(samples) // 2] if samples else self.policy.default_service_ms

    def _next(self, receiver: _Receiver) -> AdmissionTicket:
        now = self.clock()
        overdue = [x for x in receiver.pending if now - x.queued_at >= self.policy.max_wait_seconds]
        if self.policy.order == "fifo" or overdue:
            return min(overdue or receiver.pending, key=lambda x: x.sequence)
        costs: dict[str, float] = defaultdict(float)
        for state in self._receivers.values():
            demand: dict[str, float] = defaultdict(float)
            for item in state.pending:
                demand[item.root_id] += item.estimate_ms
            for item in state.active.values():
                elapsed = (now - item.started_at) * 1000.0
                demand[item.root_id] += max(item.estimate_ms * 0.1, item.estimate_ms - elapsed)
            for root, work in demand.items():
                costs[root] = max(costs[root], work / self._limit(state))
        # Aging gradually decreases the cost before the overdue FIFO escape.
        return min(receiver.pending, key=lambda x: (
            costs[x.root_id] / (1.0 + (now - x.queued_at) / self.policy.aging_seconds),
            x.sequence,
        ))

    def _dispatch(self, receiver: _Receiver) -> None:
        receiver.pending[:] = [x for x in receiver.pending if not x.future.cancelled()]
        while receiver.pending and len(receiver.active) < self._limit(receiver):
            ticket = self._next(receiver)
            receiver.pending.remove(ticket)
            ticket.started_at = self.clock()
            ticket.window_at_start = self._limit(receiver)
            ticket.pending_at_start = len(receiver.pending)
            receiver.active[ticket.sequence] = ticket
            ticket.future.set_result(None)

    async def acquire(self, resource: str, root_id: str, work_class: str = "default") -> AdmissionTicket:
        if not resource or not root_id:
            raise ValueError("resource and root_id are required")
        ticket = AdmissionTicket(
            next(self._sequence), resource, root_id, work_class, self.clock(),
            self._estimate(resource, work_class), asyncio.get_running_loop().create_future(),
        )
        state = self._receiver(resource)
        state.pending.append(ticket)
        self._dispatch(state)
        try:
            await ticket.future
            return ticket
        except asyncio.CancelledError:
            if ticket in state.pending:
                state.pending.remove(ticket)
            if ticket.sequence in state.active:
                del state.active[ticket.sequence]
            ticket.released = True
            self._dispatch(state)
            raise

    def release(self, ticket: AdmissionTicket, *, success: bool = True) -> None:
        if ticket.released:
            return
        state = self._receiver(ticket.resource)
        if state.active.pop(ticket.sequence, None) is None:
            raise ValueError("ticket is not active in this scheduler")
        ticket.released = True
        if success:
            elapsed = max(0.001, (self.clock() - ticket.started_at) * 1000.0)
            self._samples[(ticket.resource, ticket.work_class)].append(elapsed)
        self._dispatch(state)

    def feedback(self, resource: str, *, waiting: float, running: float) -> dict[str, object]:
        if any(not math.isfinite(x) or x < 0 for x in (waiting, running)):
            raise ValueError("queue feedback must be finite and non-negative")
        state = self._receiver(resource)
        old = self._limit(state)
        state.feedback_samples += 1
        fraction = waiting / max(1.0, waiting + running)
        gain = self.policy.feedback_gain
        state.congestion = (1 - gain) * state.congestion + gain * fraction
        if self.policy.adaptive:
            if waiting > 0:
                state.quiet_samples = 0
                state.window = max(self.policy.min_window, state.window * (1 - state.congestion / 2))
            else:
                state.quiet_samples += 1
                if state.quiet_samples >= 2 and state.pending:
                    state.window = min(self.policy.max_window, state.window + 1)
                    state.quiet_samples = 0
        self._dispatch(state)
        return {"old_window": old, **self.snapshot(resource)}

    def snapshot(self, resource: str) -> dict[str, object]:
        state = self._receiver(resource)
        return {"window": self._limit(state), "active": len(state.active),
                "pending": len(state.pending), "congestion": state.congestion,
                "feedback_samples": state.feedback_samples}

    @asynccontextmanager
    async def slot(self, resource: str, root_id: str, work_class: str = "default") -> AsyncIterator[AdmissionTicket]:
        ticket = await self.acquire(resource, root_id, work_class)
        success = False
        try:
            yield ticket
            success = True
        finally:
            self.release(ticket, success=success)
