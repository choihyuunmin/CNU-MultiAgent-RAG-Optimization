import asyncio
from types import SimpleNamespace

import pytest

from cnu_rag_optimization import CompletionPolicy, CompletionRouter, NetworkHarness, ReceiverSpec, ReplicaSpec
from cnu_rag_optimization.application_adapter import ApplicationAdapter


def build(*, controlled=True, callback=None):
    router = CompletionRouter([ReceiverSpec("r", 1)], [ReplicaSpec("model", "r", "model", 1)],
                              CompletionPolicy(ordering="fair"))
    adapter = ApplicationAdapter(default_model="model", aliases=["model"],
        harness=NetworkHarness(router) if controlled else None, on_event=callback)
    return adapter, router


@pytest.mark.parametrize("controlled", [False, True])
def test_original_kwargs_response_identity_and_usage(controlled):
    async def run():
        events, calls = [], []
        adapter, router = build(controlled=controlled, callback=events.append)
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3))
        parameters = {"model": "model", "messages": [{"role": "user", "content": "unchanged"}],
                      "temperature": 0.3, "max_tokens": 42, "extra_body": {"unchanged": True}}
        async def original(**kwargs):
            calls.append(kwargs)
            return response
        assert await adapter.completion(original)(**parameters) is response
        assert calls == [parameters]
        assert calls[0]["messages"] is parameters["messages"]
        assert events[-1]["completion_tokens"] == 3
        assert events[-1]["status"] == "ok"
        assert "messages" not in events[-1]
        assert router.snapshot()["active"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("controlled", [False, True])
def test_stream_identity_and_original_error(controlled):
    async def run():
        adapter, router = build(controlled=controlled)
        chunk, error = object(), ValueError("original")
        async def original(**kwargs):
            yield chunk
            raise error
        stream = adapter.stream(original)(model="model")
        assert await anext(stream) is chunk
        with pytest.raises(ValueError) as caught:
            await anext(stream)
        assert caught.value is error
        assert router.snapshot()["active"] == 0
    asyncio.run(run())


def test_unknown_model_passes_through_and_is_explicitly_unmanaged():
    async def run():
        events = []
        adapter, _ = build(callback=events.append)
        async def original(**kwargs):
            return kwargs["model"]
        assert await adapter.completion(original)(model="new-model") == "new-model"
        assert events[-1]["managed"] is False
    asyncio.run(run())


def test_queued_cancellation_does_not_call_original_or_leak_ticket():
    async def run():
        events, calls = [], []
        adapter, router = build(callback=events.append)
        held = await router.acquire(root_id="held", stage="s", contract_id="model")
        async def original(**kwargs):
            calls.append(kwargs)
        task = asyncio.create_task(adapter.completion(original)(model="model"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        router.release(held)
        assert not calls
        assert router.snapshot()["pending"] == router.snapshot()["active"] == 0
        assert events[-1]["status"] == "cancelled"
        assert events[-1]["api_ms"] is None
    asyncio.run(run())


def test_asgi_isolates_concurrent_requests_and_preserves_messages():
    async def run():
        events, sent = [], []
        adapter, _ = build(callback=events.append)
        response = {"type": "http.response.body", "body": b"unchanged", "more_body": False}
        async def llm(**kwargs):
            await asyncio.sleep(0)
            return "result"
        wrapped_llm = adapter.completion(llm)
        async def app(scope, receive, send):
            await wrapped_llm(model="model")
            await send(response)
        async def send(message):
            sent.append(message)
        scopes = [{"type": "http", "path": "/api/generate/stream",
                   "headers": [(b"x-experiment-case-id", q)]} for q in (b"q1", b"q2")]
        await asyncio.gather(*(adapter.asgi(app)(scope, None, send) for scope in scopes))
        finished = [e for e in events if e["event"] == "request_finished"]
        assert {e["case_id"] for e in finished} == {"q1", "q2"}
        assert len({e["root_id"] for e in finished}) == 2
        assert all(len(e["calls"]) == 1 and e["calls"][0]["root_id"] == e["root_id"] for e in finished)
        assert all(item is response for item in sent)
    asyncio.run(run())


def test_observer_failure_does_not_break_completion():
    async def run():
        def broken(event):
            raise RuntimeError("observer only")
        adapter, _ = build(callback=broken)
        async def original(**kwargs):
            return 7
        assert await adapter.completion(original)() == 7
        assert adapter.observer_errors == 1
    asyncio.run(run())
