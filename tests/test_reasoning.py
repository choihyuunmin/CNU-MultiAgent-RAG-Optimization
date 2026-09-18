import asyncio
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from cnu_rag_optimization.reasoning import (
    IncompleteGenerationError, ReasoningPolicy, ReasoningStallError,
    StreamProgress, monitor_stream,
)


def chunk(delta=None, finish=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}]}


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


def fake_llm(monkeypatch, create):
    infra, package, llm = ModuleType("infra"), ModuleType("infra.llm"), ModuleType("infra.llm.client")
    infra.llm, package.client = package, llm
    llm.MASTER_MODEL = "orchestrator"
    llm._get_openai_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.acompletion_via_proxy = llm.acompletion_stream_via_proxy = lambda: None
    for module in [infra, package, llm]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return llm
