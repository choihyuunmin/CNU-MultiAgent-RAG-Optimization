"""Observable workflow adapter; never reads, shortens, or rewrites reasoning text.

Budgets are per-model scheduling estimates, NOT model output limits or GPU bytes.
All authoritative work is retained. Missing estimates run exclusively, and work
larger than a configured budget runs alone rather than being silently truncated.
Instances belong to one asyncio event loop / application process.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import json
import math
import time
from typing import TypeVar
from uuid import uuid4

T = TypeVar("T")


@dataclass(frozen=True)
class TokenReservation:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in asdict(self).values()):
            raise ValueError("token estimates must be nonnegative integers")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")

    @property
    def kv_tokens(self):
        # Cached input still occupies logical KV capacity. Do not subtract it.
        return self.input_tokens + self.output_tokens

    @property
    def prefill_tokens(self):
        return self.input_tokens - self.cached_tokens


@dataclass(frozen=True)
class ModelBudget:
    max_calls: int
    max_kv_tokens: int
    max_prefill_tokens: int

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()):
            raise ValueError("model budgets must be positive integers")


@dataclass(frozen=True)
class ServingPressure:
    observed_at: float  # time.monotonic(), not wall clock
    kv_usage: float
    waiting: int
    preemptions_delta: int

    def healthy(self, now: float, max_age_s: float, kv_ceiling: float) -> bool:
        values = tuple(asdict(self).values())
        return (
            all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)
            and 0 <= now - self.observed_at <= max_age_s
            and 0 <= self.kv_usage < kv_ceiling
            and self.waiting == 0 and self.preemptions_delta == 0
        )


class ModelAdmission:
    """FIFO token budget with cancellation-safe accounting and no short-job bypass.

    Oversized/unknown reservations consume the entire budget exclusively. This
    guarantees eventual admission, not that such a request fits physical VRAM.
    """
    def __init__(self, budget: ModelBudget):
        self.budget = budget
        self.active = self.kv_tokens = self.prefill_tokens = 0
        self.pending = deque()
        self.condition = asyncio.Condition()
        self.exclusive = False

    def _cost(self, reservation):
        exclusive = reservation is None or (
            reservation.kv_tokens > self.budget.max_kv_tokens
            or reservation.prefill_tokens > self.budget.max_prefill_tokens)
        if exclusive:
            return self.budget.max_kv_tokens, self.budget.max_prefill_tokens, True
        return reservation.kv_tokens, reservation.prefill_tokens, False

    def _fits(self, cost):
        kv, prefill, exclusive = cost
        return (not self.exclusive and self.active < self.budget.max_calls
                and (not exclusive or self.active == 0)
                and self.kv_tokens + kv <= self.budget.max_kv_tokens
                and self.prefill_tokens + prefill <= self.budget.max_prefill_tokens)

    @asynccontextmanager
    async def slot(self, reservation: TokenReservation | None, *, optional=False):
        cost = self._cost(reservation)
        ticket = object()
        acquired = False
        try:
            async with self.condition:
                if optional:
                    # Never queue speculative work ahead of authoritative work.
                    if not self.pending and reservation is not None and not cost[2] and self._fits(cost):
                        acquired = True
                else:
                    self.pending.append(ticket)
                    try:
                        await self.condition.wait_for(
                            lambda: self.pending[0] is ticket and self._fits(cost))
                        self.pending.popleft()
                        acquired = True
                    except BaseException:
                        self.pending.remove(ticket)
                        self.condition.notify_all()
                        raise
                if acquired:
                    self.active += 1
                    self.kv_tokens += cost[0]
                    self.prefill_tokens += cost[1]
                    self.exclusive = cost[2]
                    self.condition.notify_all()
            yield acquired
        finally:
            if acquired:
                async with self.condition:
                    self.active -= 1
                    self.kv_tokens -= cost[0]
                    self.prefill_tokens -= cost[1]
                    self.exclusive = False
                    self.condition.notify_all()

    def snapshot(self):
        return {"active": self.active, "queued": len(self.pending),
                "reserved_kv_tokens": self.kv_tokens,
                "reserved_prefill_tokens": self.prefill_tokens}


@dataclass
class WorkflowTrace:
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time.monotonic)
    spans: list[dict] = field(default_factory=list)
    dropped_spans: int = 0
    elapsed_s: float | None = None


_trace: ContextVar[WorkflowTrace | None] = ContextVar("adapter_trace", default=None)
_stage: ContextVar[str] = ContextVar("adapter_stage", default="unclassified")


def current_workflow_trace_id() -> str | None:
    """Opaque server-generated correlation ID; never a client/session identifier."""
    trace = _trace.get()
    return trace.trace_id if trace is not None else None


class WorkflowAdapter:
    """Coordinate measured stages without altering model/tool request payloads.

    `observe` only records. `budget` applies configured model gates. No policy
    automatically raises HTTP/pipeline capacity or authorizes a quality change.
    """
    def __init__(self, *, mode="observe", budgets: Mapping[str, ModelBudget] | None = None,
                 speculation=False, pressure_max_age_s=5.0, speculation_kv_ceiling=0.80,
                 max_spans=2048):
        if mode not in {"observe", "budget"}:
            raise ValueError("mode must be observe or budget")
        if mode == "budget" and not budgets:
            raise ValueError("budget mode requires explicit per-model budgets")
        if (not math.isfinite(pressure_max_age_s) or pressure_max_age_s <= 0
                or not math.isfinite(speculation_kv_ceiling)
                or not 0 < speculation_kv_ceiling < 1
                or type(max_spans) is not int or max_spans < 1):
            raise ValueError("invalid telemetry bounds")
        self.mode, self.speculation = mode, speculation
        self.pressure_max_age_s, self.speculation_kv_ceiling = pressure_max_age_s, speculation_kv_ceiling
        self.max_spans = max_spans
        self.gates = {name: ModelAdmission(b) for name, b in (budgets or {}).items()}
        self.pressure: dict[str, ServingPressure] = {}

    @contextmanager
    def request(self):
        trace = WorkflowTrace()
        token = _trace.set(trace)
        try:
            yield trace
        finally:
            trace.elapsed_s = time.monotonic() - trace.started_at
            _trace.reset(token)

    @contextmanager
    def stage(self, name: str):
        # Integration supplies static stage names, never query/reasoning text.
        token = _stage.set(name)
        try:
            yield
        finally:
            _stage.reset(token)

    def record(self, kind: str, started: float, *, model="", **numeric):
        trace = _trace.get()
        if trace is None:
            return
        if len(trace.spans) >= self.max_spans:
            trace.dropped_spans += 1
            return
        # Explicitly reject free text in extra fields. No payloads/hashes stored.
        fields = {k: v for k, v in numeric.items()
                  if v is None or isinstance(v, bool)
                  or isinstance(v, (int, float)) and math.isfinite(v)}
        trace.spans.append({"kind": kind, "stage": _stage.get(), "model": model,
                            "start_s": started - trace.started_at,
                            "end_s": time.monotonic() - trace.started_at, **fields})

    @asynccontextmanager
    async def execution(self, model: str, reservation: TokenReservation | None = None):
        """Hold only a model call, not its enclosing agent/HTTP lifetime.

        Streaming holds credit until source exhaustion/close; downstream
        backpressure is included. There is no unbounded prefetch buffer.
        """
        queued = time.monotonic()
        gate = self.gates.get(model) if self.mode == "budget" else None
        async with self._slot(gate, reservation):
            started = time.monotonic()
            self.record("adapter_wait", queued, model=model, gated=gate is not None)
            success = False
            try:
                yield
                success = True
            finally:
                self.record("model_rpc_lifetime", started, model=model, success=success,
                            estimated_input_tokens=reservation.input_tokens if reservation else None,
                            estimated_output_tokens=reservation.output_tokens if reservation else None)

    @asynccontextmanager
    async def _slot(self, gate, reservation):
        if gate is None:
            yield
        else:
            async with gate.slot(reservation):
                yield

    async def call(self, model: str, call: Callable[[], Awaitable[T]], *, reservation=None) -> T:
        async with self.execution(model, reservation):
            return await call()

    async def stream(self, model: str, factory: Callable[[], AsyncIterator[T]], *, reservation=None):
        async with self.execution(model, reservation):
            source = factory()
            try:
                async for chunk in source:
                    yield chunk
            finally:
                if hasattr(source, "aclose"):
                    await source.aclose()

    def speculation_allowed(self, model: str) -> bool:
        pressure = self.pressure.get(model)
        return bool(self.mode == "budget" and self.speculation and model in self.gates
                    and pressure and pressure.healthy(time.monotonic(), self.pressure_max_age_s,
                                                     self.speculation_kv_ceiling))

    async def verified_overlap(self, *, model: str, predicted_input: Mapping,
                               authoritative_input: Callable[[], Awaitable[Mapping]],
                               invoke: Callable[[Mapping], Awaitable[T]],
                               reservation: TokenReservation | None,
                               read_only: bool = False) -> tuple[T, bool]:
        """Reuse only an exact FULL invocation match within this request.

        Inputs must include model/revision, prompt/messages/history, generation
        options, tools, evidence/context and tenant/security scope where relevant.
        No document-ID-only reuse. Read-only/pure calls only, explicit opt-in.
        The speculative callback is a raw invocation: do not nest the same gate.
        """
        predicted = json.dumps(predicted_input, sort_keys=True, ensure_ascii=False, allow_nan=False)
        sentinel = object()
        task = None
        decision_started = time.monotonic()
        reused = False

        async def draft():
            if not self.speculation_allowed(model):
                return sentinel
            async with self.gates[model].slot(reservation, optional=True) as admitted:
                if not admitted:
                    return sentinel
                started = time.monotonic()
                try:
                    return await invoke(json.loads(predicted))
                finally:
                    self.record("speculative_rpc_lifetime", started, model=model)

        if read_only and self.speculation_allowed(model):
            task = asyncio.create_task(draft())
        try:
            authoritative = await authoritative_input()
            signature = json.dumps(authoritative, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if task is not None and signature == predicted:
                try:
                    value = await task
                    if value is not sentinel:
                        reused = True
                        return value, True
                except Exception:
                    pass  # authoritative invocation is the only error fallback
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # A mismatched input cannot borrow the predicted token estimate.
            estimate = reservation if signature == predicted else None
            return await self.call(model, lambda: invoke(authoritative), reservation=estimate), False
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.record("speculation_decision", decision_started, model=model,
                        launched=task is not None, reused=reused)

    def snapshot(self):
        return {"mode": self.mode, "models": {name: gate.snapshot() for name, gate in self.gates.items()},
                "speculation_enabled": self.speculation,
                "speculation_allowed": {name: self.speculation_allowed(name) for name in self.gates},
                "quality_equivalence_established": False}


def diagnose_trace(trace: WorkflowTrace) -> dict:
    """Observed interval coverage, NOT a hidden CoT or causal GPU decomposition.

    Overlapping intervals are unioned per category; categories can still overlap
    each other and must not be added to obtain end-to-end latency.
    """
    totals = {}
    for kind in {s["kind"] for s in trace.spans}:
        intervals = sorted((s["start_s"], s["end_s"]) for s in trace.spans if s["kind"] == kind)
        end = -math.inf
        duration = 0.0
        for start, stop in intervals:
            duration += max(0.0, stop - max(start, end))
            end = max(end, stop)
        totals[kind] = duration
    elapsed = trace.elapsed_s
    waiting = totals.get("adapter_wait", 0)
    starts = [s["start_s"] for s in trace.spans
              if s["kind"] in {"agent_lifetime", "adapter_wait", "model_rpc_lifetime"}]
    stage_work = {}
    for span in trace.spans:
        if span["kind"] == "model_rpc_lifetime":
            key = (span["stage"], span["model"])
            stage_work[key] = stage_work.get(key, 0) + span["end_s"] - span["start_s"]
    return {"elapsed_s": elapsed, "interval_union_s": totals,
            "adapter_queue_dominant": bool(elapsed and waiting > elapsed / 2),
            "before_first_instrumented_stage_s": min(starts) if starts else None,
            "stage_model_work_s": [{"stage": stage, "model": model, "sum_s": work}
                                    for (stage, model), work in sorted(stage_work.items())],
            "observed_reasoning_tokens": sum(s.get("reasoning_tokens") or 0 for s in trace.spans),
            "reasoning_tokens_observed": any(s.get("reasoning_tokens") is not None for s in trace.spans),
            "dropped_spans": trace.dropped_spans,
            "scope": "observable workflow intervals; categories overlap; not a causal critical path"}
