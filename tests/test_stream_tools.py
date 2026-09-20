import asyncio

import pytest

from cnu_rag_optimization.stream_tools import (
    CallReady, RoundEnd, StreamProtocolError, ToolBinding, execute_tool_stream,
)


def call(i, name="read", args="{}"):
    return CallReady(i, f"call-{i}", name, args)


def test_execution_overlaps_generation_but_results_keep_model_order():
    async def run():
        began, release = asyncio.Event(), asyncio.Event()
        invoked = []
        async def handler(args):
            invoked.append(args["i"])
            if args["i"] == 0:
                began.set()
                await release.wait()
            return str(args["i"])
        async def source():
            yield call(0, args='{"i":0}')
            await asyncio.wait_for(began.wait(), 1)  # no round end yet
            yield call(1, args='{"i":1}')
            release.set()
            yield RoundEnd(2)
        out = await execute_tool_stream(source(), {"read": ToolBinding(handler, True)})
        assert [x.content for x in out] == ["0", "1"]
        assert sorted(invoked) == [0, 1]
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["serial", "parallel", "stream"])
def test_side_effect_is_ordering_barrier_and_calls_are_not_removed(mode):
    async def run():
        order = []
        async def handler(args):
            await asyncio.sleep(0)
            order.append(args["i"])
            return str(args["i"])
        async def source():
            yield call(0, args='{"i":0}')
            yield call(1, "write", '{"i":1}')
            yield call(2, args='{"i":2}')
            yield RoundEnd(3)
        out = await execute_tool_stream(source(), {
            "read": ToolBinding(handler, True), "write": ToolBinding(handler)}, mode=mode)
        assert order == [0, 1, 2]
        assert len(out) == 3
    asyncio.run(run())


@pytest.mark.parametrize("tail", [[], [RoundEnd(2)], [call(0)], [RoundEnd(1), call(1)]])
def test_invalid_stream_cancels_and_drains_tools(tail):
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()
        async def handler(args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        async def source():
            yield call(0)
            await started.wait()
            for item in tail:
                yield item
        with pytest.raises(StreamProtocolError):
            await execute_tool_stream(source(), {"read": ToolBinding(handler, True)})
        assert closed.is_set()
    asyncio.run(run())


def test_cancellation_closes_source_and_running_tool():
    async def run():
        started, tool_closed, source_closed = (asyncio.Event() for _ in range(3))
        async def handler(args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                tool_closed.set()
        async def source():
            try:
                yield call(0)
                await asyncio.Event().wait()
            finally:
                source_closed.set()
        task = asyncio.create_task(execute_tool_stream(source(), {"read": ToolBinding(handler, True)}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tool_closed.is_set() and source_closed.is_set()
    asyncio.run(run())


@pytest.mark.parametrize("args", ['{"x":1,"x":2}', '{"x":NaN}', '[1]', '{'])
def test_malformed_arguments_never_invoke_tool(args):
    async def run():
        async def handler(value):
            pytest.fail("invalid args invoked tool")
        async def source():
            yield call(0, args=args)
            yield RoundEnd(1)
        with pytest.raises(ValueError):
            await execute_tool_stream(source(), {"read": ToolBinding(handler, True)})
    asyncio.run(run())


def test_tool_failure_is_not_retried_and_observer_failure_is_isolated():
    async def run():
        calls = []
        async def handler(args):
            calls.append(1)
            raise RuntimeError("tool failed")
        def observer(event):
            raise ValueError("sink failed")
        async def source():
            yield call(0)
            yield RoundEnd(1)
        with pytest.raises(RuntimeError, match="tool failed"):
            await execute_tool_stream(source(), {"read": ToolBinding(handler, True)}, on_event=observer)
        assert calls == [1]
    asyncio.run(run())
