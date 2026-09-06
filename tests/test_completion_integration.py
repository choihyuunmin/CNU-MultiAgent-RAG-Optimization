"""Exercise the public example with existing-client stubs, never a live model."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "completion_example", Path(__file__).resolve().parents[1] / "examples" / "completion_routing.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_original_non_stream_client_receives_unchanged_options():
    async def run():
        router = example.build_router()
        request = {"model": "original-alias", "messages": [{"role": "user", "content": "test"}],
                   "stream": False, "temperature": 0.7, "max_tokens": 123,
                   "extra_body": {"original_option": True}}
        received, result = [], object()

        async def create(**kwargs):
            received.append(kwargs)
            return result

        actual = await example.call_existing_clients(router, {"fast": client(create)}, request,
                                                       root_id="q", stage="classify")
        assert actual is result
        assert received == [request]
        assert received[0]["messages"] is request["messages"]
        assert received[0]["extra_body"] is request["extra_body"]
        with pytest.raises(ValueError, match="stream"):
            await example.call_existing_clients(router, {}, {**request, "stream": True},
                                                root_id="q", stage="classify")
    asyncio.run(run())


def test_original_stream_client_options_and_chunk_identity_preserved():
    async def run():
        router = example.build_router()
        pieces, closed, received = [object(), object()], [], []
        request = {"model": "original-alias", "messages": [], "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0}

        class Stream:
            def __aiter__(self):
                async def chunks():
                    for item in pieces:
                        yield item
                return chunks()

            async def close(self):
                closed.append(True)

        async def create(**kwargs):
            received.append(kwargs)
            return Stream()

        async with example.stream_existing_clients(router, {"fast": client(create)}, request,
                                                     root_id="q", stage="answer") as output:
            actual = [chunk async for chunk in output]
        assert all(a is b for a, b in zip(actual, pieces, strict=True))
        assert received == [request]
        assert closed == [True]
        assert router.snapshot()["active"] == 0
        with pytest.raises(ValueError, match="already use streaming"):
            async with example.stream_existing_clients(router, {}, {**request, "stream": False},
                                                        root_id="q", stage="answer"):
                pass
    asyncio.run(run())
