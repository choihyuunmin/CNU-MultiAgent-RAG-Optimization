"""Opt-in hooks layered on an isolated legacy adapter, never production code.

Only static stage labels and numeric metadata enter the new trace. Legacy paper
traces may contain visible answer text and must still be treated as private.
"""
from __future__ import annotations

from functools import wraps
import importlib
import inspect
import json
from pathlib import Path
import sys
import time

from cnu_rag_optimization import (
    ModelBudget, TokenReservation, WorkflowAdapter, diagnose_trace,
)


def load_adapter(path):
    data = json.loads(Path(path).read_text())
    allowed = {"mode", "budgets", "reservations", "stage_hooks", "serving_meter"}
    if set(data) - allowed:
        raise ValueError("unknown adapter configuration fields")
    adapter = WorkflowAdapter(mode=data.get("mode", "observe"), budgets={
        name: ModelBudget(**value) for name, value in data.get("budgets", {}).items()})
    estimates = {name: TokenReservation(**value) for name, value in data.get("reservations", {}).items()}
    adapter.serving_meter_settings = data.get("serving_meter")
    # A static reservation must be an operator-calibrated conservative estimate.
    # Do not guess tokenizer counts from string length or reduce max_tokens.
    hooks = data.get("stage_hooks", [
        {"module": "agent.validation_agent", "function": "run_preparation", "stage": "preparation"},
        {"module": "agent.law_search_agent", "function": "run_search", "stage": "retrieval"},
    ])
    return adapter, estimates, hooks


def replace_aliases(original, replacement, name):
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith(("agent.", "infra.", "core.", "api.")):
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)


def numeric_usage(response):
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return {}
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    return {"input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens")}


def install(adapter, estimates, stage_hooks):
    import infra.llm.client as llm
    meter = None
    if getattr(adapter, "serving_meter_settings", None):
        from moleg_adapter_serving import ServingMeter
        meter = ServingMeter(adapter, adapter.serving_meter_settings)
        adapter.serving_meter = meter
    # Validate all hooks before patching anything; unknown stages are not guessed
    # from prompt content or call ordinal (branch/retry dependent).
    hooks = []
    for hook in stage_hooks:
        module = importlib.import_module(hook["module"])
        original = getattr(module, hook["function"])
        if not inspect.iscoroutinefunction(original):
            raise ValueError("stage hooks must name async functions")
        hooks.append((module, hook["function"], hook["stage"], original))

    original_chat = llm.acompletion_via_proxy
    original_stream = llm.acompletion_stream_via_proxy

    @wraps(original_chat)
    async def chat(**kwargs):
        model = str(kwargs.get("model") or llm.MASTER_MODEL)
        started = time.monotonic()
        reservation = await meter.estimate(model, kwargs) if meter and model in meter.roles else estimates.get(model)
        response = await adapter.call(model, lambda: original_chat(**kwargs),
                                      reservation=reservation)
        usage = numeric_usage(response)
        observed = usage.get("input_tokens")
        adapter.record("usage", started, model=model, **usage,
                       input_estimate_error=observed-reservation.input_tokens
                       if observed is not None and reservation is not None else None)
        return response

    @wraps(original_stream)
    async def stream(**kwargs):
        model = str(kwargs.get("model") or llm.MASTER_MODEL)
        started = time.monotonic()
        reasoning_chars = 0
        reasoning_observed = False
        visible_chars = 0
        first_visible_s = None
        last_usage = {}
        reservation = await meter.estimate(model, kwargs) if meter and model in meter.roles else estimates.get(model)
        source = adapter.stream(model, lambda: original_stream(**kwargs), reservation=reservation)
        try:
            async for chunk in source:
                usage = numeric_usage(chunk)
                if any(value is not None for value in usage.values()):
                    # Some providers emit cumulative usage more than once.
                    # Keep the latest total, do not sum across stream chunks.
                    last_usage = usage
                for choice in getattr(chunk, "choices", ()) or ():
                    delta = getattr(choice, "delta", None)
                    if hasattr(delta, "model_dump"):
                        fields = delta.model_dump()
                        value = fields.get("reasoning") or fields.get("reasoning_content")
                        if isinstance(value, str):
                            reasoning_observed = True
                            reasoning_chars += len(value)
                        content = fields.get('content')
                        if isinstance(content,str) and content:
                            visible_chars += len(content)
                            if first_visible_s is None:
                                first_visible_s = time.monotonic()-started
                yield chunk  # identical chunk object, ordering and backpressure
        finally:
            await source.aclose()
            if last_usage:
                adapter.record("usage", started, model=model, **last_usage)
            adapter.record("reasoning_size", started, model=model,
                           reasoning_characters=reasoning_chars if reasoning_observed else None,
                           visible_characters=visible_chars,first_visible_s=first_visible_s)

    llm.acompletion_via_proxy, llm.acompletion_stream_via_proxy = chat, stream
    replace_aliases(original_chat, chat, "acompletion_via_proxy")
    replace_aliases(original_stream, stream, "acompletion_stream_via_proxy")
    for module, name, stage, original in hooks:
        def wrap(function, stage_name):
            @wraps(function)
            async def measured(*args, **kwargs):
                with adapter.stage(stage_name):
                    started = time.monotonic()
                    try:
                        return await function(*args, **kwargs)
                    finally:
                        adapter.record("agent_lifetime", started)
            return measured
        wrapped = wrap(original, stage)
        setattr(module, name, wrapped)
        replace_aliases(original, wrapped, name)


class TraceASGI:
    """Outer observer includes both legacy admission queues and SSE transmission.

    Does not parse/store HTTP bodies, cookies, URLs or identifiers from clients.
    It cannot infer GPU service time or TTFT from an arbitrary ASGI body frame.
    """
    def __init__(self, app, adapter, sink):
        self.app, self.adapter, self.sink = app, adapter, sink
        self.write_errors = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            try:
                return await self.app(scope, receive, send)
            finally:
                meter = getattr(self.adapter, "serving_meter", None)
                if meter:
                    await meter.close()
        if scope["type"] != "http" or scope.get("path", "").startswith("/__scaling_state"):
            return await self.app(scope, receive, send)
        try:
            with self.adapter.request() as trace:
                started = time.monotonic()
                completed = False
                status = None
                async def measured_send(message):
                    nonlocal status
                    if message["type"] == "http.response.start":
                        status = message["status"]
                    await send(message)
                try:
                    await self.app(scope, receive, measured_send)
                    completed = True
                finally:
                    self.adapter.record("http_lifetime", started, completed=completed, status=status)
        finally:
            row = {"trace_id": trace.trace_id, "spans": trace.spans,
                   "diagnosis": diagnose_trace(trace), "adapter": self.adapter.snapshot()}
            try:
                self.sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.sink.flush()
            except (OSError, ValueError):
                self.write_errors += 1  # never retry/duplicate an answer on logging error
