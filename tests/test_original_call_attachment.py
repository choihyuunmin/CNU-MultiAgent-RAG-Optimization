import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter

spec = importlib.util.spec_from_file_location("original_runner",
    Path(__file__).parents[1] / "scripts" / "serve_embedded_adapter.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_attach_bypasses_added_admission_and_keeps_inputs():
    async def run():
        payload, answer, chunk = object(), object(), object()
        async def limited(**kwargs):
            pytest.fail("old admission wrapper executed")
        async def original(**kwargs):
            assert kwargs["messages"] is payload
            return answer
        async def stream(**kwargs):
            assert kwargs["messages"] is payload
            yield chunk
        client = SimpleNamespace(acompletion_via_proxy=limited,
            acompletion_stream_via_proxy=limited, raw=original, raw_stream=stream)
        runner.attach_original_calls(client, InferenceOverlapAdapter(),
            completion_source="raw", stream_source="raw_stream")
        assert await client.acompletion_via_proxy(messages=payload) is answer
        assert [x async for x in client.acompletion_stream_via_proxy(messages=payload)] == [chunk]
    asyncio.run(run())


def test_missing_original_function_does_not_partially_attach():
    original = object()
    client = SimpleNamespace(acompletion_via_proxy=original, raw=lambda: None)
    with pytest.raises(AttributeError):
        runner.attach_original_calls(client, InferenceOverlapAdapter(),
            completion_source="raw", stream_source="missing")
    assert client.acompletion_via_proxy is original
