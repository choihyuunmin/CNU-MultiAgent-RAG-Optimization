import asyncio
from contextlib import asynccontextmanager

import pytest

from cnu_rag_optimization import CompletionPolicy, CompletionRouter, NetworkHarness, ReceiverSpec, ReplicaSpec


def make_router(capacity=1, *, ordering="fair"):
    return CompletionRouter([ReceiverSpec("receiver", capacity)],
                            [ReplicaSpec("alias", "receiver", "unchanged", 1)],
                            CompletionPolicy(ordering=ordering))


def meta(root):
    return dict(root_id=root, stage="stage", contract_id="unchanged")


def test_question_round_robin_preserves_fifo_within_each_flow():
    async def run():
        router = make_router()
        first = await router.acquire(**meta("a"))
        tasks = [asyncio.create_task(router.acquire(**meta(root))) for root in ("a", "a", "b", "b")]
        await asyncio.sleep(0)
        router.release(first)
        order = []
        for task in (tasks[2], tasks[0], tasks[3], tasks[1]):
            ticket = await asyncio.wait_for(task, 1)
            order.append(ticket.root_id)
            router.release(ticket)
        assert order == ["b", "a", "b", "a"]
        assert router.snapshot()["active"] == router.snapshot()["pending"] == 0
        assert router._last_dispatch == {}
    asyncio.run(run())


def test_relay_preserves_chunks_and_releases_credit_before_consumer_drains():
    async def run():
        router = make_router()
        events, values, closed = [], [object(), object(), object()], asyncio.Event()
        harness = NetworkHarness(router, buffer_chunks=4, on_event=events.append)
        async def source():
            try:
                async with harness.slot(**meta("first")):
                    for value in values:
                        yield value
            finally:
                closed.set()
        async with harness.relay(source, root_id="first", stage="answer") as output:
            first = await anext(output)
            await closed.wait()
            assert router.snapshot()["active"] == 0
            other = await router.acquire(**meta("other"))
            router.release(other)
            remaining = [value async for value in output]
        assert [first, *remaining] == values
        assert events[-1]["fully_consumed"] is True
        assert all(e["buffer_peak_chunks"] <= 4 for e in events)
    asyncio.run(run())


def test_relay_bounds_buffer_and_cancels_full_producer_on_early_close():
    async def run():
        router = make_router()
        produced, events, closed = [], [], asyncio.Event()
        harness = NetworkHarness(router, buffer_chunks=2, on_event=events.append)
        async def source():
            try:
                async with harness.slot(**meta("q")):
                    for i in range(1000):
                        produced.append(i)
                        yield i
            finally:
                closed.set()
        async with harness.relay(source, root_id="q", stage="answer") as output:
            assert await anext(output) == 0
            await asyncio.sleep(0)
            assert len(produced) <= 4  # two queued, one yielded, one producer-held
        assert closed.is_set()
        assert router.snapshot()["active"] == 0
        assert events[-1]["fully_consumed"] is False
    asyncio.run(run())


@pytest.mark.parametrize("failure", [ValueError("original upstream failure"), asyncio.CancelledError("original cancellation")])
def test_relay_propagates_original_failure_after_prior_chunk(failure):
    async def run():
        harness = NetworkHarness(make_router(), buffer_chunks=1)
        async def source():
            yield "before-error"
            raise failure
        async with harness.relay(source, root_id="q", stage="answer") as output:
            assert await anext(output) == "before-error"
            with pytest.raises(type(failure)) as caught:
                await asyncio.wait_for(anext(output), 1)
            assert caught.value is failure
    asyncio.run(run())


def test_upstream_self_cancellation_wakes_consumer_and_releases_credit():
    async def run():
        router = make_router()
        harness = NetworkHarness(router, buffer_chunks=1)
        async def source():
            async with harness.slot(**meta("q")):
                yield "before-cancellation"
                asyncio.current_task().cancel("upstream cancellation")
                await asyncio.sleep(0)
        async with harness.relay(source, root_id="q", stage="s") as output:
            assert await anext(output) == "before-cancellation"
            with pytest.raises(asyncio.CancelledError, match="upstream cancellation"):
                await asyncio.wait_for(anext(output), 1)
        assert router.snapshot()["active"] == 0
    asyncio.run(run())


def test_relay_observer_failure_does_not_change_response():
    async def run():
        def broken(event):
            raise RuntimeError("observer")
        harness = NetworkHarness(make_router(), on_event=broken)
        async def source():
            yield None  # None is data, not a terminal marker.
        async with harness.relay(source, root_id="q", stage="s") as output:
            assert [x async for x in output] == [None]
        assert harness.observer_errors == 2
    asyncio.run(run())


def test_harness_rejects_unbounded_buffer():
    with pytest.raises(ValueError):
        NetworkHarness(make_router(), buffer_chunks=0)
