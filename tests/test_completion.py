import asyncio
from contextlib import asynccontextmanager

import pytest

from cnu_rag_optimization import (
    CompletionPolicy, CompletionRouter, ReceiverSpec, ReplicaSpec,
)


def setup(*, same_receiver=False, policy=None, on_event=None):
    now = [0.0]
    receivers = [ReceiverSpec("gpu-a")]
    if not same_receiver:
        receivers.append(ReceiverSpec("gpu-b"))
    router = CompletionRouter(receivers, [
        ReplicaSpec("fast", "gpu-a", "model-v1", 1000),
        ReplicaSpec("slow", "gpu-a" if same_receiver else "gpu-b", "model-v1", 1500),
    ], policy or CompletionPolicy(), clock=lambda: now[0], on_event=on_event)
    return router, now


def metadata(root="q", stage="prepare", **kwargs):
    return dict(root_id=root, stage=stage, contract_id="model-v1", **kwargs)


def test_prefers_fast_idle_but_overflows_when_busy():
    async def run():
        router, _ = setup()
        first = await router.acquire(**metadata("first"))
        second = await router.acquire(**metadata("second"))
        assert (first.replica_id, second.replica_id) == ("fast", "slow")
        assert router.snapshot()["active_by_receiver"] == {"gpu-a": 1, "gpu-b": 1}
        router.release(first, success=False)
        router.release(second, success=False)
    asyncio.run(run())


def test_waits_for_nearly_free_fast_replica_instead_of_slow_idle():
    async def run():
        router, now = setup()
        first = await router.acquire(**metadata("first"))
        now[0] = 0.9
        waiting = asyncio.create_task(router.acquire(**metadata("waiting")))
        await asyncio.sleep(0)
        assert not waiting.done()
        assert router.snapshot()["active_by_receiver"]["gpu-b"] == 0
        now[0] = 1.0
        router.release(first)
        second = await waiting
        assert second.replica_id == "fast"
        assert second.wait_ms == pytest.approx(100)
        router.release(second, success=False)
    asyncio.run(run())


def test_virtual_queue_accounts_for_earlier_pending_work():
    async def run():
        router, now = setup()
        first = await router.acquire(**metadata("first"))
        now[0] = 0.9
        older = asyncio.create_task(router.acquire(**metadata("older")))
        await asyncio.sleep(0)
        newer = await router.acquire(**metadata("newer"))
        assert newer.replica_id == "slow"  # Fast lane includes older's reservation.
        assert not older.done()
        now[0] = 1.0
        router.release(first)
        router.release(await older, success=False)
        router.release(newer, success=False)
    asyncio.run(run())


@pytest.mark.parametrize("routing", ["fastest", "recent"])
def test_baselines_use_same_capacity_but_ignore_wait_in_route_score(routing):
    async def run():
        router, _ = setup(policy=CompletionPolicy(routing=routing, ordering="fifo"))
        first = await router.acquire(**metadata("first"))
        task = asyncio.create_task(router.acquire(**metadata("second")))
        await asyncio.sleep(0)
        assert not task.done()
        router.release(first, success=False)
        second = await task
        assert second.replica_id == "fast"
        router.release(second, success=False)
    asyncio.run(run())


def test_aliases_share_receiver_capacity_and_contracts_do_not_mix():
    async def run():
        router, _ = setup(same_receiver=True)
        first = await router.acquire(**metadata("first"))
        task = asyncio.create_task(router.acquire(**metadata("second")))
        await asyncio.sleep(0)
        assert router.snapshot()["active"] == 1
        assert not task.done()
        with pytest.raises(ValueError, match="execution contract"):
            await router.acquire(root_id="x", stage="s", contract_id="different-model")
        with pytest.raises(ValueError, match="unknown allowed"):
            await router.acquire(**metadata(allowed_replicas=["missing"]))
        router.release(first, success=False)
        router.release(await task, success=False)
    asyncio.run(run())


def test_stage_and_size_history_are_separate_and_bounded():
    async def run():
        router, now = setup(policy=CompletionPolicy(history_size=2))
        ticket = await router.acquire(**metadata(allowed_replicas=["fast"]))
        now[0] = 4.0
        router.release(ticket)
        # Slow measured prepare calls must not make a different stage avoid fast.
        answer = await router.acquire(**metadata(stage="answer"))
        assert answer.replica_id == "fast"
        now[0] += 1
        router.release(answer)
        prepare = await router.acquire(**metadata())
        assert prepare.replica_id == "slow"
        now[0] += 1.5
        router.release(prepare)
        assert router.snapshot()["history_entries"] == 2
        large = await router.acquire(**metadata(work_units=8))
        assert large.estimated_service_ms == 8000
        router.release(large, success=False)
    asyncio.run(run())


@pytest.mark.parametrize("ordering, expected", [("coflow", "small"), ("fifo", "large")])
def test_visible_question_completion_order(ordering, expected):
    async def run():
        events = []
        router, _ = setup(same_receiver=True, policy=CompletionPolicy(ordering=ordering), on_event=events.append)
        first = await router.acquire(**metadata("occupied"))
        tasks = [asyncio.create_task(router.acquire(**metadata(root))) for root in ("large", "large", "small")]
        await asyncio.sleep(0)
        router.release(first, success=False)
        assert [e for e in events if e["event"] == "dispatch"][-1]["root_id"] == expected
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                router.release(task.result(), success=False)
                tasks.remove(task)
    asyncio.run(run())


def test_overdue_large_question_gets_fifo_precedence():
    async def run():
        router, now = setup(same_receiver=True)
        first = await router.acquire(**metadata("busy"))
        old = [asyncio.create_task(router.acquire(**metadata("old"))) for _ in range(2)]
        await asyncio.sleep(0)
        now[0] = 31
        new = asyncio.create_task(router.acquire(**metadata("new")))
        await asyncio.sleep(0)
        router.release(first, success=False)
        for task in old:
            router.release(await task, success=False)
        assert not new.done() or new.result().root_id == "new"
        router.release(await new, success=False)
    asyncio.run(run())


def test_cancel_before_or_after_grant_and_release_race_leave_no_credits():
    async def run():
        router, _ = setup(same_receiver=True)
        first = await router.acquire(**metadata("busy"))
        waiting = asyncio.create_task(router.acquire(**metadata("cancel-before")))
        await asyncio.sleep(0)
        waiting.cancel()
        router.release(first, success=False)  # Happens before cancellation handler runs.
        with pytest.raises(asyncio.CancelledError):
            await waiting
        first = await router.acquire(**metadata("busy-again"))
        waiting = asyncio.create_task(router.acquire(**metadata("cancel-after")))
        await asyncio.sleep(0)
        router.release(first, success=False)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert router.snapshot()["active"] == router.snapshot()["pending"] == 0
        assert router.snapshot()["history_entries"] == 0
    asyncio.run(run())


def test_pending_call_is_rerouted_when_replica_disabled():
    async def run():
        router, now = setup()
        first = await router.acquire(**metadata("busy"))
        now[0] = 0.9
        waiting = asyncio.create_task(router.acquire(**metadata("waiting")))
        await asyncio.sleep(0)
        router.set_enabled("fast", False)
        second = await waiting
        assert second.replica_id == "slow"
        assert router.snapshot()["active"] == 2  # Disabled active call was not preempted.
        router.release(first, success=False)
        router.release(second, success=False)
    asyncio.run(run())


def test_overrun_recheck_can_use_idle_replica_without_new_arrivals():
    async def run():
        router, now = setup(policy=CompletionPolicy(recheck_seconds=0.005))
        first = await router.acquire(**metadata("busy"))
        now[0] = 0.9
        waiting = asyncio.create_task(router.acquire(**metadata("waiting")))
        await asyncio.sleep(0)
        assert not waiting.done()
        now[0] = 3.0  # Busy replica missed its expected completion.
        second = await asyncio.wait_for(waiting, timeout=0.5)
        assert second.replica_id == "slow"
        router.release(first, success=False)
        router.release(second, success=False)
        assert router._recheck is None
    asyncio.run(run())


def test_no_enabled_replica_waits_until_reenabled():
    async def run():
        router, _ = setup()
        router.set_enabled("fast", False)
        router.set_enabled("slow", False)
        waiting = asyncio.create_task(router.acquire(**metadata()))
        await asyncio.sleep(0)
        assert not waiting.done()
        router.set_enabled("slow", True)
        router.release(await waiting, success=False)
    asyncio.run(run())


def test_call_preserves_payload_result_and_failure_without_retry():
    async def run():
        router, _ = setup()
        request = {"messages": [{"content": "unchanged"}], "temperature": 0, "stream": False}
        original = repr(request)
        result, calls = object(), []

        async def send():
            calls.append(request)
            return result

        assert await router.call({"fast": send}, **metadata()) is result
        assert len(calls) == 1 and calls[0] is request and repr(request) == original

        async def fail():
            calls.append(request)
            raise RuntimeError("original failure")

        with pytest.raises(RuntimeError, match="original failure"):
            await router.call({"fast": fail, "slow": fail}, **metadata())
        assert len(calls) == 2  # Failed request is not retried at another replica.
        assert router.snapshot()["active"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("early_close", [False, True])
def test_stream_holds_credit_until_close_and_preserves_chunks(early_close):
    async def run():
        events, closed = [], []
        router, now = setup(on_event=events.append)
        pieces = [b'data: {"x":1}\n\n', b"data: [DONE]\n\n"]

        @asynccontextmanager
        async def send():
            async def generate():
                for piece in pieces:
                    now[0] += 0.25
                    yield piece
            try:
                yield generate()
            finally:
                closed.append(True)

        actual = []
        async with router.stream({"fast": send}, **metadata()) as output:
            async for piece in output:
                assert router.snapshot()["active"] == 1
                actual.append(piece)
                if early_close:
                    break
        assert actual == (pieces[:1] if early_close else pieces)
        assert closed == [True]
        assert router.snapshot()["active"] == 0
        event = events[-1]
        assert event["event"] == "finished" and event["success"] is not early_close
        assert event["first_chunk_ms"] == 250
        assert router.snapshot()["history_entries"] == (0 if early_close else 1)
    asyncio.run(run())


def test_stream_cancellation_closes_sender_and_does_not_learn():
    async def run():
        router, _ = setup()
        started, closed = asyncio.Event(), []

        @asynccontextmanager
        async def send():
            async def generate():
                started.set()
                await asyncio.Event().wait()
                yield b"never"
            try:
                yield generate()
            finally:
                closed.append(True)

        async def consume():
            async with router.stream({"fast": send}, **metadata()) as stream:
                async for _ in stream:
                    pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]
        assert router.snapshot()["active"] == router.snapshot()["history_entries"] == 0
    asyncio.run(run())


def test_wait_included_in_total_time_and_observer_failure_is_isolated():
    async def run():
        events = []
        router, now = setup(same_receiver=True, on_event=events.append)
        first = await router.acquire(**metadata("busy"))
        waiting = asyncio.create_task(router.acquire(**metadata("waiting")))
        await asyncio.sleep(0)
        now[0] = 2
        router.release(first)
        ticket = await waiting
        now[0] = 5
        router.release(ticket)
        assert events[-1]["wait_ms"] == 2000
        assert events[-1]["service_ms"] == 3000
        assert events[-1]["total_ms"] == 5000

        def fail(_):
            raise RuntimeError("logger unavailable")

        router.on_event = fail
        router.release(await router.acquire(**metadata()))
        assert router.snapshot()["active"] == 0
        assert router.snapshot()["observer_errors"] == 2
    asyncio.run(run())


def test_foreign_release_and_cross_loop_use_are_rejected():
    router, _ = setup()

    async def first():
        other, _ = setup()
        ticket = await router.acquire(**metadata())
        with pytest.raises(ValueError, match="not active"):
            other.release(ticket)
        router.release(ticket, success=False)
        router.release(ticket)  # Idempotent cleanup.
    asyncio.run(first())
    with pytest.raises(RuntimeError, match="event loops"):
        asyncio.run(router.acquire(**metadata()))


def test_400_mock_calls_preserve_results_and_shared_capacity():
    async def run():
        router = CompletionRouter(
            [ReceiverSpec("a", 2), ReceiverSpec("b", 3)],
            [ReplicaSpec("a1", "a", "model-v1", 10),
             ReplicaSpec("a2", "a", "model-v1", 10),
             ReplicaSpec("b1", "b", "model-v1", 15)],
        )
        active = {"a": 0, "b": 0}
        peaks = active.copy()
        actual_calls = []

        async def one(index):
            result = object()

            async def send(replica):
                resource = router.replicas[replica].resource_id
                active[resource] += 1
                peaks[resource] = max(peaks[resource], active[resource])
                try:
                    actual_calls.append(index)
                    await asyncio.sleep(0)
                    return result
                finally:
                    active[resource] -= 1

            senders = {r: (lambda r=r: send(r)) for r in router.replicas}
            assert await router.call(senders, **metadata(str(index), stage="answer")) is result

        await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(400))), timeout=5)
        assert sorted(actual_calls) == list(range(400))
        assert peaks == {"a": 2, "b": 3}
        assert router.snapshot()["active"] == router.snapshot()["pending"] == 0
        assert router._recheck is None
    asyncio.run(run())


@pytest.mark.parametrize("failure_point", ["open", "read", "close"])
def test_stream_failure_is_not_retried_or_learned(failure_point):
    async def run():
        router, _ = setup()
        calls = []

        @asynccontextmanager
        async def send():
            calls.append(1)
            if failure_point == "open":
                raise RuntimeError("open")

            async def chunks():
                yield b"first"
                if failure_point == "read":
                    raise RuntimeError("read")

            yield chunks()
            if failure_point == "close":
                raise RuntimeError("close")

        with pytest.raises(RuntimeError, match=failure_point):
            async with router.stream({"fast": send, "slow": send}, **metadata()) as output:
                async for _ in output:
                    pass
        assert calls == [1]
        assert router.snapshot()["active"] == router.snapshot()["history_entries"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("work_units", [0, -1, float("nan"), float("inf"), 1e308])
def test_invalid_work_estimate_fails_before_enqueue(work_units):
    async def run():
        router, _ = setup()
        with pytest.raises(ValueError):
            await router.acquire(**metadata(work_units=work_units))
        assert router.snapshot()["pending"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("kwargs", [dict(routing="bad"), dict(ordering="bad"),
    dict(ewma_alpha=float("nan")), dict(residual_floor=0), dict(history_size=0),
    dict(aging_seconds=float("inf")), dict(starvation_seconds=-1)])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        CompletionPolicy(**kwargs)
