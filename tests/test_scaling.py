import asyncio

import pytest

from cnu_rag_optimization.admission import PressureGate


def test_gate_cancellation_and_pressure_drain():
    async def run():
        gate = PressureGate(limit=2, maximum=4, adaptive=True)
        order = []
        release = asyncio.Event()
        async def hold(i):
            async with gate.slot():
                order.append(i)
                await release.wait()
        first = asyncio.create_task(hold(0))
        second = asyncio.create_task(hold(1))
        await asyncio.sleep(0)
        cancelled = asyncio.create_task(hold(2))
        fourth = asyncio.create_task(hold(3))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await gate.observe(waiting=3, kv_usage=.9)
        assert gate.limit == 1 and gate.active == 2
        release.set()
        await asyncio.gather(first, second, fourth)
        assert order == [0, 1, 3]
        assert gate.active == 0 and not gate.pending
        await gate.observe(waiting=None, kv_usage=None)
        assert gate.limit == 1
    asyncio.run(run())


def test_adaptive_growth_requires_pending_demand_and_three_healthy_samples():
    async def run():
        gate = PressureGate(limit=1, maximum=2, adaptive=True)
        for _ in range(5):
            await gate.observe(waiting=0, kv_usage=.1)
        assert gate.limit == 1
        release = asyncio.Event()
        started = asyncio.Event()
        async def waiting():
            async with gate.slot():
                started.set()
                await release.wait()
        async with gate.slot():
            task = asyncio.create_task(waiting())
            await asyncio.sleep(0)
            for _ in range(2):
                await gate.observe(waiting=0, kv_usage=.1)
            assert gate.limit == 1 and not started.is_set()
            await gate.observe(waiting=0, kv_usage=.1)
            await asyncio.wait_for(started.wait(), 1)
            assert gate.limit == 2 and gate.active == 2
            release.set()
        await task
        assert gate.active == 0
    asyncio.run(run())
