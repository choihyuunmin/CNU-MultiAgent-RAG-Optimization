"""Non-preemptive, completion-aware dispatch outside the inference engine.

Only caller-declared metadata enters this scheduler. Requests, model options,
responses, and credentials remain inside the caller's original send functions.
One instance owns one event loop and must see all traffic it is meant to balance.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import TypeVar

T = TypeVar("T")


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReceiverSpec:
    """All endpoint aliases sharing capacity MUST use the same resource_id."""

    resource_id: str
    capacity: int = 1

    def __post_init__(self) -> None:
        if not self.resource_id or type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("resource_id and a positive integer capacity are required")


@dataclass(frozen=True)
class ReplicaSpec:
    replica_id: str
    resource_id: str
    contract_id: str
    initial_service_ms: float

    def __post_init__(self) -> None:
        if not self.replica_id or not self.resource_id or not self.contract_id:
            raise ValueError("replica_id, resource_id, and contract_id are required")
        _positive(self.initial_service_ms, "initial_service_ms")


@dataclass(frozen=True)
class CompletionPolicy:
    routing: str = "completion"
    ordering: str = "coflow"
    ewma_alpha: float = 0.2
    aging_seconds: float = 10.0
    starvation_seconds: float = 30.0
    residual_floor: float = 0.1
    recheck_seconds: float = 0.25
    history_size: int = 512

    def __post_init__(self) -> None:
        if self.routing not in {"completion", "recent", "fastest"}:
            raise ValueError("routing must be completion, recent, or fastest")
        if self.ordering not in {"coflow", "fifo", "fair"}:
            raise ValueError("ordering must be coflow or fifo")
        if not 0 < self.ewma_alpha <= 1 or not 0 < self.residual_floor <= 1:
            raise ValueError("ewma_alpha and residual_floor must be in (0, 1]")
        _positive(self.aging_seconds, "aging_seconds")
        _positive(self.starvation_seconds, "starvation_seconds")
        _positive(self.recheck_seconds, "recheck_seconds")
        if type(self.history_size) is not int or self.history_size < 1:
            raise ValueError("history_size must be a positive integer")


@dataclass(eq=False)
class CompletionTicket:
    sequence: int
    root_id: str
    stage: str
    contract_id: str
    work_units: float
    eligible: tuple[str, ...]
    queued_at: float
    future: asyncio.Future[None] = field(repr=False)
    replica_id: str | None = None
    resource_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    estimated_service_ms: float = 0.0
    first_chunk_at: float | None = None
    released: bool = False

    @property
    def wait_ms(self) -> float:
        end = self.started_at if self.started_at is not None else self.finished_at
        return max(0.0, ((end if end is not None else self.queued_at) - self.queued_at) * 1000)


class CompletionRouter:
    """Route by predicted queue + service time; reserve capacity atomically.

    Pending calls remain unbound until dispatch. A small virtual list schedule
    accounts for earlier pending calls and all active calls sharing a receiver.
    A slower idle replica is used only when it predicts an earlier completion.
    Coflow ordering estimates *visible* remaining work, not unknown future DAG
    branches. Aging gives overdue work FIFO precedence at its next feasible slot.
    """

    def __init__(
        self,
        receivers: Iterable[ReceiverSpec],
        replicas: Iterable[ReplicaSpec],
        policy: CompletionPolicy = CompletionPolicy(),
        *,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        receiver_list, replica_list = tuple(receivers), tuple(replicas)
        self.receivers = {item.resource_id: item for item in receiver_list}
        self.replicas = {item.replica_id: item for item in replica_list}
        if not self.receivers or len(self.receivers) != len(receiver_list):
            raise ValueError("unique, nonempty receivers required")
        if not self.replicas or len(self.replicas) != len(replica_list):
            raise ValueError("unique, nonempty replicas required")
        if any(item.resource_id not in self.receivers for item in replica_list):
            raise ValueError("replica references an unknown receiver")
        self.policy, self.clock, self.on_event = policy, clock, on_event
        self._enabled = set(self.replicas)
        self._sequence = count()
        self._dispatch_sequence = count()
        self._last_dispatch: dict[str, int] = {}
        self._pending: list[CompletionTicket] = []
        self._active: dict[int, CompletionTicket] = {}
        self._history: OrderedDict[tuple[str, str, int], float] = OrderedDict()
        self._recent = {item.replica_id: item.initial_service_ms for item in replica_list}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recheck: asyncio.TimerHandle | None = None
        self.observer_errors = 0

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("a CompletionRouter cannot be shared across event loops")
        self._loop = loop

    @staticmethod
    def _key(replica_id: str, ticket: CompletionTicket) -> tuple[str, str, int]:
        return replica_id, ticket.stage, math.floor(math.log2(ticket.work_units))

    def _estimate(self, replica_id: str, ticket: CompletionTicket) -> float:
        key = self._key(replica_id, ticket)
        rate = self._history.get(key, self.replicas[replica_id].initial_service_ms)
        if key in self._history:
            self._history.move_to_end(key)
        return max(0.001, rate * ticket.work_units)

    def _remaining(self, ticket: CompletionTicket, now: float) -> float:
        elapsed = max(0.0, (now - ticket.started_at) * 1000)
        return max(ticket.estimated_service_ms * self.policy.residual_floor,
                   ticket.estimated_service_ms - elapsed,
                   elapsed - ticket.estimated_service_ms)

    def _ordered(self, now: float) -> list[CompletionTicket]:
        if self.policy.ordering == "fair":
            # One turn per backlogged question, FIFO within each question.
            # Fair in admission opportunities, NOT GPU seconds or output tokens.
            flows: dict[str, deque[CompletionTicket]] = defaultdict(deque)
            for ticket in sorted(self._pending, key=lambda t: t.sequence):
                flows[ticket.root_id].append(ticket)
            roots = sorted(flows, key=lambda root: (
                self._last_dispatch.get(root, -1), flows[root][0].sequence))
            ordered = []
            while roots:
                for root in roots:
                    ordered.append(flows[root].popleft())
                roots = [root for root in roots if flows[root]]
            return ordered
        costs: dict[str, float] = defaultdict(float)
        for ticket in self._active.values():
            costs[ticket.root_id] += self._remaining(ticket, now)
        for ticket in self._pending:
            choices = [self._estimate(r, ticket) for r in ticket.eligible if r in self._enabled]
            costs[ticket.root_id] += min(choices, default=math.inf)

        def priority(ticket: CompletionTicket):
            age = max(0.0, now - ticket.queued_at)
            if self.policy.ordering == "fifo" or age >= self.policy.starvation_seconds:
                return 0, ticket.sequence, ticket.sequence
            return 1, costs[ticket.root_id] / (1 + age / self.policy.aging_seconds), ticket.sequence

        return sorted(self._pending, key=priority)

    def _dispatch(self) -> None:
        now = self.clock()
        # Cancellation can race with release before acquire's except block runs.
        for ticket in tuple(self._pending):
            if ticket.future.cancelled():
                self._pending.remove(ticket)
                ticket.finished_at, ticket.released = now, True
                self._emit("cancelled", ticket, wait_ms=ticket.wait_ms)
        live_roots = {t.root_id for t in self._pending} | {t.root_id for t in self._active.values()}
        self._last_dispatch = {root: turn for root, turn in self._last_dispatch.items() if root in live_roots}
        lanes: dict[str, list[float]] = {}
        for resource, spec in self.receivers.items():
            active = [self._remaining(t, now) for t in self._active.values() if t.resource_id == resource]
            lanes[resource] = sorted(active + [0.0] * (spec.capacity - len(active)))
        for ticket in self._ordered(now):
            choices = []
            for replica_id in ticket.eligible:
                if replica_id not in self._enabled:
                    continue
                replica = self.replicas[replica_id]
                queue = min(lanes[replica.resource_id])
                service = self._estimate(replica_id, ticket)
                if self.policy.routing == "completion":
                    score = queue + service
                elif self.policy.routing == "recent":
                    score = self._recent[replica_id]
                else:
                    score = replica.initial_service_ms
                choices.append((score, replica_id, queue, service))
            if not choices:
                continue  # Explicitly disabled replicas wait for enable or caller cancellation.
            _, replica_id, queue, service = min(choices)
            resource = self.replicas[replica_id].resource_id
            lane = lanes[resource].index(queue)
            lanes[resource][lane] = queue + service
            if queue > 0:
                continue
            self._pending.remove(ticket)
            ticket.replica_id, ticket.resource_id = replica_id, resource
            ticket.started_at, ticket.estimated_service_ms = now, service
            self._active[ticket.sequence] = ticket
            if self.policy.ordering == "fair":
                self._last_dispatch[ticket.root_id] = next(self._dispatch_sequence)
            ticket.future.set_result(None)
            self._emit("dispatch", ticket, wait_ms=ticket.wait_ms, estimated_service_ms=service)
        if self._recheck is not None:
            self._recheck.cancel()
            self._recheck = None
        if self._pending and self._active:
            # Reconsider unsent calls when estimates overrun, even without arrivals
            # or completions. This never cancels, duplicates, or migrates a call.
            self._recheck = self._loop.call_later(self.policy.recheck_seconds, self._dispatch)

    def _emit(self, event: str, ticket: CompletionTicket, **values: object) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"event": event, "sequence": ticket.sequence, "root_id": ticket.root_id,
                           "stage": ticket.stage, "replica_id": ticket.replica_id,
                           "resource_id": ticket.resource_id, **values})
        except Exception:
            # A broken logger must not change responses or leak admission credits.
            self.observer_errors += 1

    async def acquire(
        self, *, root_id: str, stage: str, contract_id: str, work_units: float = 1.0,
        allowed_replicas: Iterable[str] | None = None,
    ) -> CompletionTicket:
        self._bind_loop()
        if not root_id or not stage or not contract_id:
            raise ValueError("root_id, stage, and contract_id are required")
        _positive(work_units, "work_units")
        allowed = set(self.replicas if allowed_replicas is None else allowed_replicas)
        if not allowed.issubset(self.replicas):
            raise ValueError("unknown allowed replica")
        eligible = tuple(sorted(r for r in allowed if self.replicas[r].contract_id == contract_id))
        if not eligible:
            raise ValueError("no replica satisfies the declared execution contract")
        for replica_id in eligible:
            _positive(self.replicas[replica_id].initial_service_ms * work_units, "estimated service")
        ticket = CompletionTicket(next(self._sequence), root_id, stage, contract_id, work_units,
                                  eligible, self.clock(), self._loop.create_future())
        self._pending.append(ticket)
        self._dispatch()
        try:
            await ticket.future
            return ticket
        except asyncio.CancelledError:
            if ticket in self._pending:
                self._pending.remove(ticket)
                ticket.finished_at, ticket.released = self.clock(), True
                self._emit("cancelled", ticket, wait_ms=ticket.wait_ms)
                self._dispatch()
            elif self._active.get(ticket.sequence) is ticket:
                self.release(ticket, success=False)
            raise

    def release(self, ticket: CompletionTicket, *, success: bool = True) -> None:
        self._bind_loop()
        if ticket.released:
            return
        if self._active.get(ticket.sequence) is not ticket:
            raise ValueError("ticket is not active in this router")
        del self._active[ticket.sequence]
        ticket.finished_at, ticket.released = self.clock(), True
        service = max(0.001, (ticket.finished_at - ticket.started_at) * 1000)
        if success:
            key, alpha = self._key(ticket.replica_id, ticket), self.policy.ewma_alpha
            rate = service / ticket.work_units
            old = self._history.get(key)
            self._history[key] = rate if old is None else alpha * rate + (1 - alpha) * old
            self._history.move_to_end(key)
            while len(self._history) > self.policy.history_size:
                self._history.popitem(last=False)
            self._recent[ticket.replica_id] = alpha * service + (1 - alpha) * self._recent[ticket.replica_id]
        self._emit("finished", ticket, success=success, wait_ms=ticket.wait_ms,
                   service_ms=service, total_ms=ticket.wait_ms + service,
                   estimated_service_ms=ticket.estimated_service_ms,
                   prediction_error_ms=service - ticket.estimated_service_ms,
                   first_chunk_ms=None if ticket.first_chunk_at is None else
                   (ticket.first_chunk_at - ticket.queued_at) * 1000)
        self._dispatch()

    def set_enabled(self, replica_id: str, enabled: bool) -> None:
        """External health checks may disable future dispatch; active calls drain."""
        self._bind_loop()
        if replica_id not in self.replicas:
            raise ValueError("unknown replica")
        if enabled:
            self._enabled.add(replica_id)
        else:
            self._enabled.discard(replica_id)
        self._dispatch()

    def snapshot(self) -> dict[str, object]:
        return {"pending": len(self._pending), "active": len(self._active),
                "active_by_receiver": {r: sum(t.resource_id == r for t in self._active.values())
                                       for r in self.receivers},
                "history_entries": len(self._history), "observer_errors": self.observer_errors}

    @asynccontextmanager
    async def slot(self, **metadata) -> AsyncIterator[CompletionTicket]:
        ticket = await self.acquire(**metadata)
        success = False
        try:
            yield ticket
            success = True
        finally:
            self.release(ticket, success=success)

    async def call(self, senders: Mapping[str, Callable[[], Awaitable[T]]], **metadata) -> T:
        """Call exactly one original non-streaming sender; no retry or rewrite."""
        async with self.slot(allowed_replicas=senders, **metadata) as ticket:
            return await senders[ticket.replica_id]()

    @asynccontextmanager
    async def stream(
        self,
        senders: Mapping[str, Callable[[], AbstractAsyncContextManager[AsyncIterator[T]]]],
        **metadata,
    ) -> AsyncIterator[AsyncIterator[T]]:
        """Keep a credit until stream close; incomplete streams do not train estimates."""
        ticket = await self.acquire(allowed_replicas=senders, **metadata)
        exhausted = success = False

        async def chunks(upstream: AsyncIterator[T]) -> AsyncIterator[T]:
            nonlocal exhausted
            async for chunk in upstream:
                if ticket.first_chunk_at is None:
                    ticket.first_chunk_at = self.clock()
                yield chunk
            exhausted = True

        try:
            async with senders[ticket.replica_id]() as upstream:
                async with aclosing(chunks(upstream)) as output:
                    yield output
            success = exhausted
        finally:
            self.release(ticket, success=success)
