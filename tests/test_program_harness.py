import asyncio

import pytest

from cnu_rag_optimization.program_harness import ProgramHarness


def test_ready_successor_overlaps_unrelated_work_and_reports():
    async def run():
        events = []
        harness = ProgramHarness(events.append)
        dependency = asyncio.Event()
        unrelated = asyncio.Event()
        successor_done = asyncio.Event()

        async def root():
            async def predecessor():
                await dependency.wait()
                return 4
            async def successor(value):
                assert value == 4
                successor_done.set()
                return 5
            first = asyncio.create_task(predecessor())
            followup = harness.after("next", first, successor)
            dependency.set()
            await successor_done.wait()
            assert not unrelated.is_set()
            unrelated.set()
            return await harness.join(followup)

        assert await harness.wrap(root)() == 5
        row = events[0]["nodes"][0]
        assert row["name"] == "next" and row["status"] == "ok"
        assert row["overlap_before_join_ms"] is not None
    asyncio.run(run())


def test_scope_isolation_and_cleanup():
    async def run():
        events = []
        harness = ProgramHarness(events.append)
        never = asyncio.Event()
        async def root(name):
            predecessor = asyncio.create_task(never.wait())
            harness.after(name, predecessor, lambda _: asyncio.sleep(0))
            return name
        assert await asyncio.gather(harness.wrap(root)("a"), harness.wrap(root)("b")) == ["a", "b"]
        assert {event["nodes"][0]["name"] for event in events} == {"a", "b"}
        assert all(event["cancelled"] == 1 for event in events)
        assert harness.scope.get() is None
    asyncio.run(run())


def test_errors_propagate_and_observer_errors_do_not():
    async def run():
        harness = ProgramHarness(lambda event: 1 / 0)
        async def root():
            async def bad(_):
                raise RuntimeError("same error")
            predecessor = asyncio.create_task(asyncio.sleep(0, result=1))
            return await harness.join(harness.after("bad", predecessor, bad))
        with pytest.raises(RuntimeError, match="same error"):
            await harness.wrap(root)()
    asyncio.run(run())


def test_duplicate_and_foreign_nodes_rejected():
    async def run():
        harness = ProgramHarness()
        async def root():
            one = asyncio.create_task(asyncio.sleep(0, result=1))
            task = harness.after("x", one, lambda value: asyncio.sleep(0, result=value))
            with pytest.raises(ValueError):
                harness.after("x", one, lambda value: asyncio.sleep(0, result=value))
            with pytest.raises(ValueError):
                await harness.join(asyncio.create_task(asyncio.sleep(0)))
            return await harness.join(task)
        assert await harness.wrap(root)() == 1
    asyncio.run(run())


def test_barrier_control_waits_until_join_and_keeps_original_value():
    async def run():
        events, calls = [], []
        harness = ProgramHarness(events.append, start_when_ready=False)
        value = object()

        async def root(state):
            async def operation(item):
                calls.append(item)
                return item
            predecessor = asyncio.create_task(asyncio.sleep(0, result=value))
            task = harness.after("read", predecessor, operation)
            await predecessor
            await asyncio.sleep(0)  # Let the dependency observer record readiness.
            assert calls == [] and not task.done()
            return await harness.join(task)

        assert await harness.wrap(root, request_key=lambda state: state["id"])(
            {"id": "q1"}) is value
        assert calls == [value]
        event = events[0]
        node = event["nodes"][0]
        assert event["schedule"] == "barrier" and event["request_id"] == "q1"
        assert node["dependency_ready_ms"] <= node["joined_ms"] <= node["started_ms"]
        assert node["ready_wait_ms"] >= 0
    asyncio.run(run())


def test_control_does_not_run_unjoined_work():
    async def run():
        calls = []
        harness = ProgramHarness(start_when_ready=False)

        async def root():
            async def operation(_):
                calls.append("unexpected")
            first = asyncio.create_task(asyncio.sleep(0, result=1))
            harness.after("unused", first, operation)
            await first
            await asyncio.sleep(0)
        await harness.wrap(root)()
        assert not calls
        assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("cnu-program-")]
    asyncio.run(run())
