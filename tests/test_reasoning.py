import asyncio
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cnu_rag_optimization import WorkflowAdapter, TokenReservation
from cnu_rag_optimization.reasoning import (
    IncompleteGenerationError, ReasoningPolicy, ReasoningStallError,
    StreamProgress, monitor_stream,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from moleg_model_transport import chat_payload, install_standard_transport
from moleg_unrestricted import ActiveCalls, remove_admission_limits
from moleg_workflow_adapter import install, load_adapter


def chunk(delta=None, finish=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}]}


def test_all_64_calls_enter_without_waiting_for_another_call_to_finish():
    async def run():
        adapter, accounting = WorkflowAdapter(), ActiveCalls()
        entered, all_entered, release = [], asyncio.Event(), asyncio.Event()

        async def invoke(i):
            async with accounting.limit():
                entered.append(i)
                if len(entered) == 64:
                    all_entered.set()
                await release.wait()
                return i

        # Unknown and huge token estimates cannot serialize these requests.
        tasks = [asyncio.create_task(adapter.call("same-model", lambda i=i: invoke(i),
                 reservation=None if i % 2 else TokenReservation(1000000, 1000000))) for i in range(64)]
        try:
            await asyncio.wait_for(all_entered.wait(), 1)
            assert accounting.active == 64 and accounting.queued == 0
            release.set()
            assert await asyncio.gather(*tasks) == list(range(64))
            assert accounting.active == 0
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(run())


def test_remove_global_admission_keeps_session_state_and_other_middleware():
    middleware = lambda name: SimpleNamespace(cls=type(name, (), {}))
    auth, queue = middleware("AuthMiddleware"), middleware("GlobalWaitQueueMiddleware")
    app = SimpleNamespace(user_middleware=[auth, queue], middleware_stack=None)
    sessions = object()
    generator = SimpleNamespace(_global=SimpleNamespace(active=0, queued=0), _session_refs=sessions)
    remove_admission_limits(app, generator)
    assert app.user_middleware == [auth]
    assert generator._session_refs is sessions
    assert generator._global.max_concurrency is None
    app.middleware_stack = object()
    with pytest.raises(RuntimeError):
        remove_admission_limits(app, generator)


@pytest.mark.parametrize("delta,finish", [({"content": "정상 답변"}, "stop"),
                                         ({"tool_calls": [{"function": {"name": "search"}}]}, "tool_calls")])
def test_reasoning_then_normal_answer_or_tool_call_preserves_every_chunk(delta, finish):
    async def run():
        chunks = [chunk({"reasoning_content": "private thought"}), chunk(delta), chunk(finish=finish),
                  {"choices": [], "usage": {"completion_tokens": 12}}]
        closed = []
        async def source():
            try:
                for item in chunks:
                    yield item
            finally:
                closed.append(True)
        progress = StreamProgress()
        output = [x async for x in monitor_stream(source(), progress, ReasoningPolicy())]
        assert all(a is b for a, b in zip(output, chunks, strict=True))
        assert progress.useful and progress.finished and closed == [True]
        assert progress.reasoning_characters == len("private thought")
        assert "private thought" not in json.dumps(progress.metrics())
    asyncio.run(run())


def test_reasoning_tokens_do_not_renew_first_answer_deadline():
    async def run():
        closed = []
        async def source():
            try:
                while True:
                    await asyncio.sleep(.002)
                    yield chunk({"reasoning": "x"})
            finally:
                closed.append(True)
        progress = StreamProgress()
        with pytest.raises(ReasoningStallError):
            async with asyncio.timeout(1):
                async for _ in monitor_stream(source(), progress, ReasoningPolicy(first_output_timeout_s=.025)):
                    pass
        assert progress.stalled and closed == [True]
        assert progress.first_visible_s is None
    asyncio.run(run())


def test_first_answer_deadline_covers_silent_provider():
    async def run():
        closed = asyncio.Event()
        async def source():
            try:
                await asyncio.Event().wait()
                yield chunk()
            finally:
                closed.set()
        with pytest.raises(ReasoningStallError):
            async for _ in monitor_stream(source(), StreamProgress(), ReasoningPolicy(first_output_timeout_s=.01)):
                pass
        assert closed.is_set()
    asyncio.run(run())


@pytest.mark.parametrize("items", [[chunk(finish="stop")],
                                  [chunk({"reasoning": "x"}, "length")],
                                  [chunk({"content": "partial"}, "length")],
                                  [chunk({"content": "partial"})]])
def test_empty_truncated_or_unfinished_output_is_not_success(items):
    async def run():
        async def source():
            for item in items:
                yield item
        with pytest.raises(IncompleteGenerationError):
            async for _ in monitor_stream(source(), StreamProgress(), ReasoningPolicy()):
                pass
    asyncio.run(run())


def test_early_close_preserves_backpressure():
    async def run():
        closed = []
        async def source():
            try:
                yield chunk({"content": "first"})
                raise AssertionError("unwanted prefetch")
            finally:
                closed.append(True)
        stream = monitor_stream(source(), StreamProgress(), ReasoningPolicy())
        await anext(stream)
        await stream.aclose()
        assert closed == [True]
    asyncio.run(run())


def test_explicit_effort_and_other_generation_fields_are_preserved():
    payload = {"model": "worker agent", "messages": [{"content": "private input"}],
               "max_tokens": 2048, "reasoning_effort": "high", "tools": [{"type": "function"}],
               "extra_body": {"chat_template_kwargs": {"custom": True}}}
    before = copy.deepcopy(payload)
    changed = ReasoningPolicy().payload(payload)
    assert changed["extra_body"]["reasoning_effort"] == "high"
    assert changed["max_tokens"] == 2048 and changed["tools"] == payload["tools"]
    assert payload == before


def test_proxy_allowlist_does_not_mutate_input_or_leak_into_direct_route():
    async def create(*, model, messages, temperature, stream, extra_body=None):
        pass
    kwargs = {"extra_body": {"reasoning_effort": "low", "allowed_openai_params": ["seed"]}}
    before = copy.deepcopy(kwargs)
    proxy = chat_payload(kwargs, model="worker agent", streaming=True, create=create, proxy=True)
    assert proxy["extra_body"]["allowed_openai_params"] == ["seed", "reasoning_effort"]
    assert kwargs == before
    direct = chat_payload({"extra_body": {"reasoning_effort": "low"}}, model="actual-model",
                          streaming=True, create=create)
    assert direct["extra_body"] == {"reasoning_effort": "low"}


def fake_llm(monkeypatch, create):
    infra, package, llm = ModuleType("infra"), ModuleType("infra.llm"), ModuleType("infra.llm.client")
    infra.llm, package.client = package, llm
    llm.MASTER_MODEL = "orchestrator"
    llm._get_openai_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.acompletion_via_proxy = llm.acompletion_stream_via_proxy = lambda: None
    for module in [infra, package, llm]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return llm


@pytest.mark.parametrize("cancel", [False, True])
def test_integrated_guard_closes_sdk_stream_and_never_retries(monkeypatch, cancel):
    async def run():
        calls, closed, received = [], [], asyncio.Event()
        class Source:
            def __aiter__(self):
                return self
            async def __anext__(self):
                received.set()
                if cancel:
                    await asyncio.Event().wait()
                return chunk({"reasoning": "private thought too long"})
            async def close(self):
                closed.append(True)
        async def create(*, model, messages, temperature, stream, extra_body=None, max_tokens=None,
                         seed=None, stop=None, reasoning_effort=None):
            calls.append(dict(model=model, messages=messages, extra_body=extra_body,
                              max_tokens=max_tokens, seed=seed, stop=stop, reasoning_effort=reasoning_effort))
            return Source()
        llm = fake_llm(monkeypatch, create)
        install_standard_transport()
        adapter = WorkflowAdapter()
        adapter.reasoning_policies = {"worker agent": ReasoningPolicy(max_reasoning_characters=8)}
        install(adapter, {}, [])
        async def consume():
            with adapter.request() as trace:
                try:
                    async for _ in llm.acompletion_stream_via_proxy(model="worker agent", messages=[],
                                                                   max_tokens=2048, seed=7, stop=["END"]):
                        pass
                finally:
                    assert "private thought" not in json.dumps(trace.spans)
                    assert any(s.get("success") is False for s in trace.spans)
        task = asyncio.create_task(consume())
        await received.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ReasoningStallError):
            await task
        assert len(calls) == 1 and closed == [True]
        assert calls[0]["extra_body"] == {"reasoning_effort": "low", "allowed_openai_params": ["reasoning_effort"]}
        assert calls[0]["max_tokens"] == 2048 and calls[0]["seed"] == 7 and calls[0]["stop"] == ["END"]
    asyncio.run(run())


def test_new_config_and_removed_budget_configs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    adapter, _, _ = load_adapter(root / "integrations/2025-moleg-search/workflow-reasoning.json")
    assert adapter.reasoning_policies["worker agent"].effort == "low"
    path = tmp_path / "old.json"
    for config in [{"mode": "budget"}, {"mode": "observe", "budgets": {}}]:
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError):
            load_adapter(path)
