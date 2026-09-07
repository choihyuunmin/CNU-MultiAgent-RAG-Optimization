"""Attach execution control to existing async completion functions, without rewriting calls."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from contextvars import ContextVar
from functools import wraps

from .network_harness import NetworkHarness

_request: ContextVar[dict | None] = ContextVar("cnu_application_request", default=None)


class ApplicationAdapter:
    """Observe original calls, optionally adding receiver credits and stream relay.

    Unknown aliases pass through unchanged and are marked unmanaged. An incomplete
    topology must not make previously valid application calls fail. No prompt,
    token budget, SDK option, timeout, retry, or model argument is changed.
    """

    def __init__(self, *, default_model, aliases, harness: NetworkHarness | None = None,
                 on_event=None):
        self.default_model = default_model
        self.aliases = frozenset(aliases)
        self.harness = harness
        self.on_event = on_event
        self.observer_errors = 0

    def emit(self, event):
        try:
            if self.on_event:
                self.on_event(event)
        except Exception:
            self.observer_errors += 1

    @asynccontextmanager
    async def call(self, kwargs, *, streaming):
        context = _request.get()
        root = context["root_id"] if context else str(kwargs.get("_log_request_id") or uuid.uuid4().hex)
        alias = str(kwargs.get("model") or self.default_model).strip()
        managed = bool(self.harness and alias in self.aliases)
        started = time.perf_counter()
        record = {"root_id": root, "case_id": context["case_id"] if context else "-",
                  "application_request_id": str(kwargs.get("_log_request_id") or "-"),
                  "model": alias, "role": str(kwargs.get("_log_role") or "-"),
                  "streaming": streaming, "managed": managed, "status": "pending",
                  "chunks": 0, "wait_ms": 0.0, "api_ms": None}
        if context is not None:
            context["calls"].append(record)
        ticket, admitted = None, None
        try:
            if managed:
                ticket = await self.harness.router.acquire(root_id=root, stage=record["role"],
                    contract_id=alias, allowed_replicas=[alias])
            admitted = time.perf_counter()
            record["wait_ms"] = (admitted - started) * 1000
            record["status"] = "active"
            yield record
            record["status"] = "ok"
        except BaseException as exc:
            record["status"] = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else "error"
            record["error_type"] = type(exc).__name__
            raise
        finally:
            ended = time.perf_counter()
            record["total_ms"] = (ended - started) * 1000
            if admitted is None:
                record["wait_ms"] = record["total_ms"]
            else:
                record["api_ms"] = (ended - admitted) * 1000
            if ticket is not None:
                self.harness.router.release(ticket, success=record["status"] == "ok")
            self.emit({"event": "llm_finished", **record})

    def completion(self, original):
        @wraps(original)
        async def wrapped(**kwargs):
            async with self.call(kwargs, streaming=False) as record:
                response = await original(**kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    for name in ("prompt_tokens", "completion_tokens"):
                        value = getattr(usage, name, None)
                        if isinstance(value, int):
                            record[name] = value
                return response
        return wrapped

    def stream(self, original):
        @wraps(original)
        async def wrapped(**kwargs):
            context = _request.get()
            root = context["root_id"] if context else str(kwargs.get("_log_request_id") or uuid.uuid4().hex)

            async def source():
                async with self.call(kwargs, streaming=True) as record:
                    async with aclosing(original(**kwargs)) as upstream:
                        async for chunk in upstream:
                            record["chunks"] += 1
                            yield chunk

            if self.harness is None:
                async with aclosing(source()) as upstream:
                    async for chunk in upstream:
                        yield chunk
            else:
                async with self.harness.relay(source, root_id=root,
                                               stage=str(kwargs.get("_log_role") or "-")) as upstream:
                    async for chunk in upstream:
                        yield chunk
        return wrapped

    def asgi(self, original):
        async def wrapped(scope, receive, send):
            if scope["type"] != "http" or not scope.get("path", "").startswith(("/api/generate", "/api/agents")):
                return await original(scope, receive, send)
            raw = dict(scope.get("headers", [])).get(b"x-experiment-case-id", b"-")
            case = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw.decode("ascii", "ignore"))[:128]
            context = {"root_id": uuid.uuid4().hex, "case_id": case, "calls": []}
            token = _request.set(context)
            started = time.perf_counter()
            status = "error"
            try:
                await original(scope, receive, send)
                status = "returned"  # HTTP/body success is evaluated separately by the client.
            finally:
                self.emit({"event": "request_finished", **context, "status": status,
                           "elapsed_ms": (time.perf_counter() - started) * 1000})
                _request.reset(token)
        return wrapped
