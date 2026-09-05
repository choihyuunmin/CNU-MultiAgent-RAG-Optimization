#!/usr/bin/env python3
"""Start an existing ASGI app with a caller-supplied admission hook.

Application imports and endpoint mapping are integration inputs. No original
application source or model request is copied into this standalone repository.
"""

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

import httpx

from cnu_rag_optimization.coflow import CoflowAdmission, CoflowPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--application", default="main_server:app")
    parser.add_argument("--hook", default="infra.llm.client:llm_admission_slot")
    parser.add_argument("--trace-module", default="infra.rag_trace")
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--mode", choices=["control", "fifo", "coflow", "feedback"], required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    os.chdir(args.application_root)
    sys.path[:0] = [str(args.application_root / "src"), str(args.application_root)]
    trace = importlib.import_module(args.trace_module)
    topology = json.loads(args.topology.read_text())
    aliases = topology["aliases"]
    receivers = topology["receivers"]
    scheduler = CoflowAdmission(CoflowPolicy(
        order="fifo" if args.mode == "fifo" else "coflow",
        initial_window=args.window, min_window=min(2, args.window),
        max_window=max(8, args.window), adaptive=args.mode == "feedback",
    ))
    state = {"sampler": None, "last_activity": 0.0, "active_slots": 0}
    sample_re = re.compile(r"^(vllm:[a-z_]+)(?:\{.*\})?\s+([-+0-9.eE]+)$")

    async def sample():
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=2.0)) as client:
            while True:
                if state["active_slots"] or time.monotonic() - state["last_activity"] <= 10:
                    for resource, url in receivers.items():
                        try:
                            response = await client.get(url)
                            response.raise_for_status()
                            values = {}
                            for line in response.text.splitlines():
                                match = sample_re.match(line)
                                if match:
                                    name, value = match.groups()
                                    values[name] = values.get(name, 0.0) + float(value)
                            # Missing telemetry freezes the window, never means empty queue.
                            waiting = values["vllm:num_requests_waiting"]
                            running = values["vllm:num_requests_running"]
                            if args.mode != "control":
                                snapshot = scheduler.feedback(resource, waiting=waiting, running=running)
                            else:
                                snapshot = {}
                            os.write(1, ("CNU_RECEIVER_V1 " + json.dumps({
                                "time": time.time(), "resource": resource,
                                "waiting": waiting, "running": running, **snapshot,
                                "counters": {key: value for key, value in values.items()
                                             if key.endswith(("_sum", "_count", "_total"))},
                            }) + "\n").encode())
                        except Exception as exc:
                            os.write(1, ("CNU_RECEIVER_ERROR " + json.dumps({
                                "time": time.time(), "resource": resource,
                                "error_type": type(exc).__name__,
                            }) + "\n").encode())
                await asyncio.sleep(2.0)

    @asynccontextmanager
    async def admitted_slot(*, endpoint, phase, request_id, work_class="default"):
        alias = endpoint.rsplit("|", 1)[-1]
        resource = aliases.get(alias)
        current = trace.current_trace() or {}
        root = str(current.get("request_id") or request_id)
        if args.mode == "control" or resource is None:
            start = time.monotonic()
            success = False
            try:
                yield None
                success = True
            finally:
                trace.record_event("coflow_admission", {"mode": args.mode, "resource": resource,
                                   "reason": "control" if args.mode == "control" else "unknown_receiver",
                                   "wait_ms": 0.0, "phase": phase, "success": success,
                                   "service_ms": round((time.monotonic() - start) * 1000, 3)})
            return
        async with trace.trace_span("queue_wait", "coflow_admission", {"phase": phase, "resource": resource}) as attrs:
            ticket = await scheduler.acquire(resource, root, work_class)
            attrs["wait_ms"] = ticket.wait_ms
        start = time.monotonic()
        success = False
        try:
            yield ticket
            success = True
        finally:
            scheduler.release(ticket, success=success)
            trace.record_event("coflow_admission", {
                "mode": args.mode, "resource": resource, "phase": phase,
                "wait_ms": round(ticket.wait_ms, 3), "window": ticket.window_at_start,
                "pending": ticket.pending_at_start,
                "service_ms": round((time.monotonic() - start) * 1000, 3),
                "success": success,
            })

    @asynccontextmanager
    async def slot(**kwargs):
        state["active_slots"] += 1
        state["last_activity"] = time.monotonic()
        if state["sampler"] is None:
            state["sampler"] = asyncio.create_task(sample())
        try:
            async with admitted_slot(**kwargs) as ticket:
                yield ticket
        finally:
            state["active_slots"] -= 1
            state["last_activity"] = time.monotonic()

    module, attribute = args.hook.split(":")
    setattr(importlib.import_module(module), attribute, slot)
    import uvicorn
    uvicorn.run(args.application, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
