import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter, ReadContract


def contract(**changes):
    return replace(ReadContract("metadata", "v1", "tenant-a:acl-v3", "snapshot-1",
                                b'{"ids":[1,2]}', True, True), **changes)


def test_original_calls_have_no_admission_and_preserve_identity():
    async def run():
        adapter = InferenceOverlapAdapter()
        started, release = [], asyncio.Event()
        payload, response = object(), object()
        async def original(value, **kwargs):
            assert value is payload and kwargs["messages"] is payload
            started.append(1)
            await release.wait()
            return response
        tasks = [asyncio.create_task(adapter.completion(original)(payload, messages=payload)) for _ in range(32)]
        await asyncio.sleep(0)
        assert len(started) == 32
        release.set()
        assert all(item is response for item in await asyncio.gather(*tasks))
    asyncio.run(run())


def test_stream_reports_observed_phases_without_text_and_closes():
    async def run():
        events, now, closed = [], [0.0], []
        adapter = InferenceOverlapAdapter(on_event=events.append, clock=lambda: now[0])
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {"reasoning_content": "private reasoning 2"}}]},
            {"choices": [{"delta": {"content": "private answer"}}]},
            {"usage": {"completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 5}}},
        ]
        async def original(**kwargs):
            try:
                for i, chunk in enumerate(chunks):
                    now[0] = (i + 1) / 10
                    yield chunk
            finally:
                closed.append(True)
        result = [x async for x in adapter.stream(original)(model="unchanged")]
        assert all(a is b for a, b in zip(result, chunks)) and closed
        event = events[0]
        assert event["first_chunk_ms"] == 100
        assert event["first_visible_ms"] == 400
        assert event["observed_reasoning_span_ms"] == pytest.approx(100)
        assert event["usage_inconsistent"] is True
        assert "private" not in repr(events)
    asyncio.run(run())


def test_nonstream_cannot_infer_reasoning_duration():
    async def run():
        events = []
        adapter = InferenceOverlapAdapter(on_event=events.append)
        response = SimpleNamespace(usage=SimpleNamespace(completion_tokens=20,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12)))
        async def original():
            return response
        assert await adapter.completion(original)() is response
        assert events[0]["reasoning_tokens"] == 12
        assert events[0]["observed_reasoning_span_ms"] is None
    asyncio.run(run())


def test_early_close_and_original_errors_propagate():
    async def run():
        closed = []
        adapter = InferenceOverlapAdapter()
        async def source():
            try:
                yield object()
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        wrapped = adapter.stream(source)()
        await anext(wrapped)
        await wrapped.aclose()
        assert closed
        error = ValueError("original")
        async def fail():
            raise error
        with pytest.raises(ValueError) as caught:
            await adapter.completion(fail)()
        assert caught.value is error
    asyncio.run(run())


def test_verified_read_overlaps_and_is_consumed_once():
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        adapter = InferenceOverlapAdapter()
        value = object()
        async def read():
            started.set()
            await finish.wait()
            return value
        async def fallback():
            pytest.fail("matching read must be reused")
        async with adapter.prepare_read(contract(), read) as prepared:
            await started.wait()  # I/O is active before the LLM completes.
            finish.set()
            assert await prepared.consume(contract(), fallback) is value
            with pytest.raises(RuntimeError):
                await prepared.consume(contract(), fallback)
        with pytest.raises(RuntimeError):
            await prepared.consume(contract(), fallback)
    asyncio.run(run())


@pytest.mark.parametrize("changes", [
    {"snapshot": "snapshot-2"}, {"authorization": "tenant-b:acl-v3"},
    {"arguments": b'{"ids":[2,1]}'}, {"implementation": "v2"},
    {"operation": "other"}, {"read_only": False}, {"deterministic": False},
])
def test_contract_mismatch_cancels_read_and_runs_original(changes):
    async def run():
        started, closed, fallback_calls = asyncio.Event(), [], []
        async def read():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        async def fallback():
            fallback_calls.append(True)
            return 42
        async with InferenceOverlapAdapter().prepare_read(contract(), read) as prepared:
            await started.wait()
            assert await prepared.consume(contract(**changes), fallback) == 42
        assert closed and fallback_calls == [True]
    asyncio.run(run())


def test_no_snapshot_means_no_speculative_io():
    async def run():
        calls = []
        async def read():
            calls.append(1)
            return 8
        missing = contract(snapshot="")
        async with InferenceOverlapAdapter().prepare_read(missing, read) as prepared:
            await asyncio.sleep(0)
            assert not calls
            assert await prepared.consume(missing, read) == 8
        assert calls == [1]
    asyncio.run(run())


def test_failed_read_falls_back_but_fallback_error_is_not_hidden():
    async def run():
        error = LookupError("original")
        async def read():
            raise OSError("speculative")
        async def fallback():
            raise error
        async with InferenceOverlapAdapter().prepare_read(contract(), read) as prepared:
            with pytest.raises(LookupError) as caught:
                await prepared.consume(contract(), fallback)
            assert caught.value is error
    asyncio.run(run())


def test_request_cancellation_cleans_read_without_fallback():
    async def run():
        started, closed, calls = asyncio.Event(), [], []
        async def read():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        async def fallback():
            calls.append(1)
        async def request():
            async with InferenceOverlapAdapter().prepare_read(contract(), read) as prepared:
                await prepared.consume(contract(), fallback)
        task = asyncio.create_task(request())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed and not calls
    asyncio.run(run())


def test_bad_telemetry_does_not_break_calls():
    async def run():
        def sink(event):
            raise OSError("sink failed")
        adapter = InferenceOverlapAdapter(on_event=sink)
        async def original():
            return {"choices": 42}
        assert await adapter.completion(original)() == {"choices": 42}
        assert adapter.observer_errors == 2
    asyncio.run(run())
