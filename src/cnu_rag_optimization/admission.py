from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import math


class PressureGate:
    def __init__(self, limit=8, maximum=32, minimum=1, adaptive=False):
        if not 1 <= minimum <= limit <= maximum:
            raise ValueError("require 1 <= minimum <= limit <= maximum")
        self.limit, self.maximum, self.minimum = limit, maximum, minimum
        self.adaptive = adaptive
        self.active = 0
        self.pending = deque()
        self.condition = asyncio.Condition()
        self.healthy_samples = 0
        self.changes = 0

    async def observe(self, *, waiting=None, kv_usage=None, preemptions=0):
        """AIMD with hysteresis; absent telemetry never implies spare capacity.

        Preemptions must be a counter *delta*, not the lifetime counter. Existing
        work drains normally when the limit decreases. No request is cancelled.
        """
        if not self.adaptive:
            return
        if waiting is None or kv_usage is None or not all(
            math.isfinite(x) for x in (waiting, kv_usage, preemptions)
        ):
            self.healthy_samples = 0
            return
        async with self.condition:
            old = self.limit
            if waiting >= 2 or kv_usage >= 0.85 or preemptions > 0:
                self.limit = max(self.minimum, self.limit // 2)
                self.healthy_samples = 0
            elif waiting == 0 and kv_usage < 0.65 and self.pending:
                self.healthy_samples += 1
                if self.healthy_samples >= 3:
                    self.limit = min(self.maximum, self.limit + 1)
                    self.healthy_samples = 0
            else:
                self.healthy_samples = 0
            self.changes += self.limit != old
            self.condition.notify_all()

    @asynccontextmanager
    async def slot(self):
        ticket = object()
        acquired = False
        try:
            async with self.condition:
                self.pending.append(ticket)
                try:
                    await self.condition.wait_for(
                        lambda: self.pending[0] is ticket and self.active < self.limit
                    )
                except BaseException:
                    self.pending.remove(ticket)
                    self.condition.notify_all()
                    raise
                self.pending.popleft()
                self.active += 1
                acquired = True
                self.condition.notify_all()
            yield
        finally:
            if acquired:
                async with self.condition:
                    self.active -= 1
                    self.condition.notify_all()
