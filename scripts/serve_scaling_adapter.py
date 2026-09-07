#!/usr/bin/env python3
"""Local-only scaling adapter; preserve original app source and inference calls."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from cnu_rag_optimization import CompletionPolicy, CompletionRouter, NetworkHarness, ReceiverSpec, ReplicaSpec

MODES = ("control", "request_window", "stage_fifo", "stage_coflow", "delivery", "delivery_repeat", "delivery_fifo", "delivery_coflow", "network_harness")


def network_stream(original, harness, trace):
    async def stream(**kwargs):
        current = trace.current_trace() or {}
        root = str(current.get("request_id") or kwargs.get("_log_request_id") or "-")
        async with harness.relay(lambda: original(**kwargs), root_id=root,
                                 stage=str(kwargs.get("_schedule_phase") or "default")) as output:
            async for chunk in output:
                yield chunk
    return stream


def emit(prefix, value):
    print(prefix + " " + json.dumps(value, ensure_ascii=False), flush=True)


def observed_gate(*, request_window=None):
    class Gate:
        def __init__(self, app, max_concurrency=4):
            self.app = app
            self.limit = request_window or max_concurrency
            self.slots = asyncio.Semaphore(self.limit)

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http" or not scope.get("path", "").startswith(("/api/generate", "/api/agents")):
                return await self.app(scope, receive, send)
            headers = dict(scope.get("headers", []))
            case = re.sub(r"[^A-Za-z0-9_.:-]", "-", headers.get(b"x-experiment-case-id", b"-").decode("ascii", "ignore"))[:128]
            entered = time.monotonic()
            acquired = None
            success = False
            try:
                async with self.slots:
                    acquired = time.monotonic()
                    await self.app(scope, receive, send)
                    success = True
            finally:
                end = time.monotonic()
                emit("CNU_INGRESS_V1", {"case_id": case, "window": self.limit, "success": success,
                     "outer_wait_ms": ((acquired if acquired is not None else end) - entered) * 1000,
                     "total_ms": (end - entered) * 1000})
    return Gate


def remove_display_waits(helpers, generate_route):
    """Remove only intentional UI sleeps, preserving original token chunks.

    Update the existing function objects so aliases imported earlier see the same
    defaults. The original implementations, upstream stream and messages remain
    untouched. Explicit caller overrides are not changed.
    """
    original = {}
    for name in ("emit_streaming_delta", "emit_fake_stream"):
        function = getattr(helpers, name)
        defaults = function.__kwdefaults__
        if not defaults or "delay_s" not in defaults or "chunk_chars" not in defaults:
            raise RuntimeError("unsupported display helper; refusing to change delivery")
        original[name] = dict(defaults)
    for name in original:
        function = getattr(helpers, name)
        function.__kwdefaults__ = {**function.__kwdefaults__, "delay_s": 0.0}
    original["thinking_delay_min_s"] = generate_route.EARLY_EXIT_THINKING_DELAY_MIN_S
    original["thinking_delay_max_s"] = generate_route.EARLY_EXIT_THINKING_DELAY_MAX_S
    generate_route.EARLY_EXIT_THINKING_DELAY_MIN_S = 0.0
    generate_route.EARLY_EXIT_THINKING_DELAY_MAX_S = 0.0
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--request-window", type=int, default=32)
    args = parser.parse_args()
    if args.request_window < 1:
        parser.error("positive request window required")
    os.chdir(args.application_root)
    sys.path[:0] = [str(args.application_root / "src"), str(args.application_root)]
    topology = json.loads(args.topology.read_text())
    expanded = args.mode in {"request_window", "stage_fifo", "stage_coflow", "delivery_fifo", "delivery_coflow", "network_harness"}
    # Install before main_server registers the middleware. Only local process state changes.
    rate_limit = importlib.import_module("api.middleware.rate_limit")
    rate_limit.GlobalWaitQueueMiddleware = observed_gate(
        request_window=args.request_window if expanded else None)
    concurrency = importlib.import_module("api.concurrency")
    original_window = concurrency.generate_queue._global.max_concurrency
    if expanded:
        concurrency.generate_queue._global.max_concurrency = args.request_window
    trace = importlib.import_module("infra.rag_trace")
    receiver_specs = [ReceiverSpec(r, value["capacity"]) for r, value in topology["receivers"].items()]
    replica_specs = [ReplicaSpec(alias, resource, alias, 1000.0)
                     for alias, resource in topology["aliases"].items()]
    scheduler = CompletionRouter(receiver_specs, replica_specs,
        CompletionPolicy(ordering="fair" if args.mode == "network_harness" else
                         "fifo" if args.mode in {"stage_fifo", "delivery_fifo"} else "coflow"),
        on_event=lambda event: emit("CNU_COMPLETION_V1", event))
    network = NetworkHarness(scheduler, buffer_chunks=64,
                             on_event=lambda event: emit("CNU_HARNESS_V1", event))

    @asynccontextmanager
    async def admitted(*, endpoint, phase, request_id, work_class="default"):
        alias = endpoint.rsplit("|", 1)[-1]
        if alias not in topology["aliases"]:
            # Fail closed for new integration paths: no silently unmetered model calls.
            raise RuntimeError("unmapped model alias in stage-credit experiment")
        current = trace.current_trace() or {}
        root = str(current.get("request_id") or request_id)
        async with trace.trace_span("queue_wait", "stage_credit", {"phase": phase}) as attrs:
            ticket = await scheduler.acquire(root_id=root, stage=f"{phase}:{work_class}",
                                               contract_id=alias, allowed_replicas=[alias])
            attrs["wait_ms"] = ticket.wait_ms
        success = False
        try:
            yield ticket
            success = True
        finally:
            scheduler.release(ticket, success=success)

    if args.mode.startswith("stage_") or args.mode in {"delivery_fifo", "delivery_coflow", "network_harness"}:
        module = importlib.import_module("infra.llm.client")
        module.llm_admission_slot = admitted
        if args.mode == "network_harness":
            module.acompletion_stream_via_proxy = network_stream(module.acompletion_stream_via_proxy, network, trace)
    module = importlib.import_module("main_server")
    display_original = None
    if args.mode.startswith("delivery") or args.mode == "network_harness":
        display_original = remove_display_waits(importlib.import_module("core.query_loop._helpers"),
                                                 importlib.import_module("api.router.generate"))
    emit("CNU_SCALING_CONFIG", {"mode": args.mode, "original_generate_window": original_window,
        "effective_generate_window": concurrency.generate_queue._global.max_concurrency,
        "inference_api_changed": False, "replica_routing_tested": False,
        "llm_streaming_changed": False, "display_wait_removed": display_original is not None,
        "original_display_defaults": display_original,
        "question_queue_order": scheduler.policy.ordering,
        "bounded_stream_relay_chunks": 64 if args.mode == "network_harness" else None,
        "receiver_capacities": {r.resource_id: r.capacity for r in receiver_specs}})
    import uvicorn
    uvicorn.run(module.app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
