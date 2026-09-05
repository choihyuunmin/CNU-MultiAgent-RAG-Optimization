import asyncio

import pytest

from cnu_rag_optimization.coflow import CoflowAdmission, CoflowPolicy


def policy(**kwargs):
    return CoflowPolicy(initial_window=1, min_window=1, max_window=3, adaptive=False, **kwargs)


def test_coflow_finishes_smaller_visible_wave_before_large_wave():
    async def run():
        scheduler = CoflowAdmission(policy())
        occupied = await scheduler.acquire("shared-engine", "occupied")
        large = [asyncio.create_task(scheduler.acquire("shared-engine", "large")) for _ in range(2)]
        small = asyncio.create_task(scheduler.acquire("shared-engine", "small"))
        await asyncio.sleep(0)
        scheduler.release(occupied)
        ticket = await small
        assert not any(x.done() for x in large)
        scheduler.release(ticket)
        for task in large:
            scheduler.release(await task)
        assert scheduler.snapshot("shared-engine")["active"] == 0
    asyncio.run(run())


def test_fifo_ablation_uses_same_credit_limit():
    async def run():
        scheduler = CoflowAdmission(policy(order="fifo"))
        occupied = await scheduler.acquire("engine", "first")
        tasks = [asyncio.create_task(scheduler.acquire("engine", root)) for root in ("large", "large", "small")]
        await asyncio.sleep(0)
        scheduler.release(occupied)
        for task in tasks:
            ticket = await task
            scheduler.release(ticket)
        assert scheduler.snapshot("engine")["active"] == 0
    asyncio.run(run())


def test_cancellation_before_and_after_credit_grant_does_not_leak():
    async def run():
        scheduler = CoflowAdmission(policy())
        occupied = await scheduler.acquire("engine", "a")
        queued = asyncio.create_task(scheduler.acquire("engine", "b"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        successor = asyncio.create_task(scheduler.acquire("engine", "c"))
        await asyncio.sleep(0)
        scheduler.release(occupied)
        successor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await successor
        assert scheduler.snapshot("engine")["active"] == 0
        assert scheduler.snapshot("engine")["pending"] == 0
    asyncio.run(run())


def test_receiver_isolation_and_exception_release():
    async def run():
        scheduler = CoflowAdmission(policy())
        first = await scheduler.acquire("engine-a", "r")
        with pytest.raises(RuntimeError):
            async with scheduler.slot("engine-b", "r"):
                raise RuntimeError("application error")
        assert scheduler.snapshot("engine-b")["active"] == 0
        scheduler.release(first)
    asyncio.run(run())


def test_congestion_feedback_stays_bounded_without_preempting_active_work():
    async def run():
        scheduler = CoflowAdmission(CoflowPolicy(initial_window=4, min_window=2, max_window=8))
        tickets = [await scheduler.acquire("engine", str(i)) for i in range(4)]
        for _ in range(20):
            scheduler.feedback("engine", waiting=10, running=4)
        snapshot = scheduler.snapshot("engine")
        assert snapshot["window"] == 2
        assert snapshot["active"] == 4
        for ticket in tickets:
            scheduler.release(ticket)
        with pytest.raises(ValueError):
            scheduler.feedback("engine", waiting=float("nan"), running=0)
    asyncio.run(run())


def test_aged_large_coflow_gets_next_available_credit():
    async def run():
        now = [0.0]
        scheduler = CoflowAdmission(policy(), clock=lambda: now[0])
        occupied = await scheduler.acquire("engine", "busy")
        old = [asyncio.create_task(scheduler.acquire("engine", "old")) for _ in range(2)]
        await asyncio.sleep(0)
        now[0] = 31.0
        new = asyncio.create_task(scheduler.acquire("engine", "new"))
        await asyncio.sleep(0)
        scheduler.release(occupied)
        for task in old:
            scheduler.release(await task)
        scheduler.release(await new)
    asyncio.run(run())
