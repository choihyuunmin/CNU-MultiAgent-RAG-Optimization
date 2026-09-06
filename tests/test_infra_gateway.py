"""CPU-only admission ordering, fairness, and cancellation tests."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import time
import pytest

path = Path(__file__).parents[1]/'scripts'/'moleg_gpu_gateway.py'
spec = importlib.util.spec_from_file_location('moleg_gpu_gateway_test', path)
gateway = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gateway
spec.loader.exec_module(gateway)


def run_order(policy, stages, quota=3, max_wait=6):
    async def scenario():
        admission = gateway.Admission(1, policy, short_quota=quota, max_wait=max_wait)
        acquired = []
        async def job(index, stage):
            async with admission.slot(stage):
                acquired.append(index)
                await asyncio.sleep(0)
        async with admission.slot('hold'):
            tasks = [asyncio.create_task(job(i, stage)) for i, stage in enumerate(stages)]
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
        assert admission.active == 0 and not admission.queue
        return acquired
    return asyncio.run(scenario())


def test_fifo_keeps_arrival_order():
    assert run_order('fifo', ['prepare','classify','classify']) == [0,1,2]


def test_short_requests_can_pass_a_long_waiting_request():
    assert run_order('fair_short', ['prepare','classify','classify']) == [1,2,0]


def test_quota_protects_long_requests():
    assert run_order('fair_short', ['prepare','classify','classify','classify','classify'], quota=2) == [1,2,0,3,4]


def test_aging_restores_oldest_request():
    assert run_order('fair_short', ['prepare','classify','classify'], max_wait=0) == [0,1,2]


def test_cancel_waiting_request_does_not_leak_slot():
    async def scenario():
        admission = gateway.Admission(1)
        async def job():
            async with admission.slot('prepare'):
                pass
        async with admission.slot('hold'):
            task = asyncio.create_task(job())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert admission.active == 0 and not admission.queue
        await job()
        assert admission.active == 0
    asyncio.run(scenario())


def test_cancel_running_request_releases_slot():
    async def scenario():
        admission = gateway.Admission(1)
        started = asyncio.Event()
        async def job():
            async with admission.slot('prepare'):
                started.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(job())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission.active == 0 and not admission.queue
    asyncio.run(scenario())


def test_concurrency_never_exceeds_limit():
    async def scenario():
        admission = gateway.Admission(3, 'fair_short')
        counts = []
        async def job(i):
            async with admission.slot('classify' if i%2 else 'prepare'):
                counts.append(admission.active)
                await asyncio.sleep(.001)
        await asyncio.gather(*(job(i) for i in range(30)))
        assert max(counts) == 3
        assert admission.active == 0 and not admission.queue
    asyncio.run(scenario())


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        gateway.Admission(0)
    with pytest.raises(ValueError):
        gateway.Admission(4, 'unknown')
