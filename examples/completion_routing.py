"""Runnable integration demo; no GPU or network is used by default.

Run: python examples/completion_routing.py

In the application, create ONE router per event loop and pass closures around
the existing clients. Do not construct a new router for every question. Replace
only the demonstration senders below; retain the application's request, auth,
timeout, model options, and streaming choice exactly as configured.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from cnu_rag_optimization import (
    CompletionPolicy, CompletionRouter, ReceiverSpec, ReplicaSpec,
)


def build_router(on_event=None):
    # Example priors only; calibrate with a separate warmup set on the target
    # deployment. Two URLs sharing a GPU/engine must share one resource_id.
    return CompletionRouter(
        [ReceiverSpec("receiver-a", capacity=1), ReceiverSpec("receiver-b", capacity=1)],
        [ReplicaSpec("fast", "receiver-a", "same-model-and-options-v1", 20),
         ReplicaSpec("slow", "receiver-b", "same-model-and-options-v1", 30)],
        CompletionPolicy(routing="completion", ordering="coflow"), on_event=on_event,
    )


async def call_existing_clients(router, clients, request, *, root_id, stage, work_units=1.0):
    """Non-streaming OpenAI-compatible client example; request stays unchanged."""
    if request.get("stream", False):
        raise ValueError("use stream_existing_clients for a streaming request")
    senders = {
        replica: (lambda client=client: client.chat.completions.create(**request))
        for replica, client in clients.items()
    }
    return await router.call(senders, root_id=root_id, stage=stage,
                             contract_id="same-model-and-options-v1", work_units=work_units)


@asynccontextmanager
async def stream_existing_clients(router, clients, request, *, root_id, stage, work_units=1.0):
    """Keep the original stream=True request and forward SDK chunk objects unchanged."""
    if request.get("stream") is not True:
        raise ValueError("request must already use streaming; this adapter never changes it")

    @asynccontextmanager
    async def sender(client):
        upstream = await client.chat.completions.create(**request)
        try:
            yield upstream
        finally:
            await upstream.close()

    senders = {replica: (lambda client=client: sender(client)) for replica, client in clients.items()}
    async with router.stream(senders, root_id=root_id, stage=stage,
                             contract_id="same-model-and-options-v1", work_units=work_units) as output:
        yield output


async def main():
    events = []
    router = build_router(events.append)
    calls = []

    async def fake_send(replica, original_request):
        calls.append((replica, original_request))
        await asyncio.sleep(0.02 if replica == "fast" else 0.03)
        return original_request  # Identity assertion checks forwarding, not model accuracy.

    async def request(index):
        payload = {"question_id": index, "unchanged": True}
        senders = {replica: (lambda replica=replica: fake_send(replica, payload))
                   for replica in ("fast", "slow")}
        result = await router.call(senders, root_id=f"demo-{index}", stage="answer",
                                   contract_id="same-model-and-options-v1")
        assert result is payload

    await asyncio.gather(*(request(i) for i in range(8)))
    assert len(calls) == 8
    assert router.snapshot()["active"] == router.snapshot()["pending"] == 0
    print(json.dumps({"kind": "synthetic_integration_check_not_llm_benchmark",
                      "calls": len(calls), "events": events}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
