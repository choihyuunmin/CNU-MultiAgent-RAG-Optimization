from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time


@dataclass(eq=False)
class Flow:
    completed: int = 0


@dataclass(eq=False)
class Ticket:
    flow: Flow
    queued: float
    sequence: int
    ready: asyncio.Future
    acquired: bool = False
    released: bool = False


class ContinuationWindow:
    """One resource, one event loop; bounded, non-preemptive dispatch.

    Completed calls are observed progress, not a prediction of future DAG size.
    With no progress difference this is FIFO. Aged pending work takes FIFO
    precedence, preventing fresh workflows from waiting behind an endless loop.
    Each workflow uses a caller-scoped Flow, including all its parallel branches.
    """

    def __init__(self, capacity: int, *, ordering='continuation', aging_s=10.0,
                 clock=time.monotonic, on_event=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        if ordering not in {'continuation', 'fifo'}:
            raise ValueError('unknown dispatch ordering')
        if not math.isfinite(aging_s) or aging_s <= 0:
            raise ValueError('aging_s must be finite and positive')
        self.capacity, self.ordering, self.aging_s = capacity, ordering, aging_s
        self.clock, self.on_event = clock, on_event
        self.active = 0
        self.pending = []
        self.sequence = 0
        self.observer_errors = 0
        self._loop = None
        self._flow = ContextVar(f'continuation_flow_{id(self)}', default=None)
        self._owned = ContextVar(f'continuation_owned_{id(self)}', default=None)

    @contextmanager
    def flow(self):
        token = self._flow.set(Flow())
        try:
            yield
        finally:
            self._flow.reset(token)

    def _emit(self, **event):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                self.observer_errors += 1

    def _dispatch(self):
        now = self.clock()
        self.pending[:] = [t for t in self.pending if not t.ready.cancelled()]
        while self.pending and self.active < self.capacity:
            def priority(t):
                if self.ordering == 'fifo' or now-t.queued >= self.aging_s:
                    return (0, t.sequence, t.sequence)
                return (1, -min(t.flow.completed, 3), t.sequence)
            ticket = min(self.pending, key=priority)
            self.pending.remove(ticket)
            self.active += 1
            ticket.acquired = True
            ticket.ready.set_result(None)

    def _release(self, ticket, success):
        if ticket.released:
            return
        ticket.released = True
        if ticket in self.pending:
            self.pending.remove(ticket)
        if ticket.acquired:
            self.active -= 1
            if success:
                ticket.flow.completed += 1
        self._dispatch()

    @asynccontextmanager
    async def slot(self):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError('a dispatch window belongs to one event loop')
        self._loop = loop
        task = asyncio.current_task()
        # A nested transport wrapper in the SAME task must not acquire twice.
        # Child tasks inherit ContextVars but must acquire their own credits.
        if self._owned.get() is task:
            yield
            return
        ticket = Ticket(self._flow.get() or Flow(), self.clock(), self.sequence, loop.create_future())
        self.sequence += 1
        self.pending.append(ticket)
        self._dispatch()
        success = False
        started = None
        token = None
        try:
            await ticket.ready
            started = self.clock()
            token = self._owned.set(task)
            self._emit(event='dispatch', wait_s=started-ticket.queued,
                       completed_before=ticket.flow.completed, active=self.active,
                       queued=len(self.pending))
            yield
            success = True
        finally:
            if token is not None:
                self._owned.reset(token)
            self._release(ticket, success)
            self._emit(event='release', success=success, acquired=ticket.acquired,
                       wait_s=(started if started is not None else self.clock())-ticket.queued,
                       service_s=self.clock()-started if started is not None else None,
                       active=self.active, queued=len(self.pending))

    def snapshot(self):
        return {'capacity': self.capacity, 'ordering': self.ordering,
                'active': self.active, 'queued': len(self.pending),
                'observer_errors': self.observer_errors}
