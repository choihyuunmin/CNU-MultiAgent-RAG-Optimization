import asyncio
from dataclasses import replace
import json
import time

import pytest

from cnu_rag_optimization import (
    TokenReservation, ServingPressure,
    WorkflowAdapter, WorkflowTrace, diagnose_trace,
    QualityEvidence, QualityThresholds, evaluate_quality_gate,
)


def test_cached_input_is_not_free_kv():
    r = TokenReservation(100, 20, 80)
    assert r.kv_tokens == 120 and r.prefill_tokens == 20


@pytest.mark.parametrize("args", [(-1, 2), (1, 2, 2), (1.5, 2), (True, 2)])
def test_invalid_token_estimates(args):
    with pytest.raises(ValueError):
        TokenReservation(*args)


def test_direct_calls_preserve_exceptions_without_private_details():
    async def run():
        adapter = WorkflowAdapter()
        async def fail():
            raise RuntimeError("private failure detail")
        with adapter.request() as trace:
            async with adapter.execution("a"):
                with pytest.raises(RuntimeError):
                    await adapter.call("b", fail)
        assert "private failure" not in json.dumps(trace.spans)
        assert any(s.get("success") is False for s in trace.spans)
    asyncio.run(run())


def test_stream_chunks_and_backpressure_preserved_on_close():
    async def run():
        adapter = WorkflowAdapter()
        chunk = object()
        closed = []
        async def source():
            try:
                yield chunk
                raise AssertionError("must not prefetch after first chunk")
            finally:
                closed.append(True)
        with adapter.request():
            stream = adapter.stream("a", source)
            assert await anext(stream) is chunk
            await stream.aclose()
        assert closed == [True]
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["missing", "stale", "future", "kv", "waiting", "preempted", "nan"])
def test_missing_or_unhealthy_pressure_suppresses_speculation(kind):
    a = WorkflowAdapter(speculation=True)
    now = time.monotonic()
    p = ServingPressure(now, .2, 0, 0)
    p = {"stale": replace(p, observed_at=now-100), "future": replace(p, observed_at=now+100),
         "kv": replace(p, kv_usage=.99), "waiting": replace(p, waiting=1),
         "preempted": replace(p, preemptions_delta=1), "nan": replace(p, kv_usage=float("nan"))}.get(kind, p)
    if kind != "missing":
        a.pressure["a"] = p
    assert not a.speculation_allowed("a")


@pytest.mark.parametrize("change", [None, "messages", "evidence", "options", "tenant"])
def test_verified_overlap_requires_full_input_match(change):
    async def run():
        a = WorkflowAdapter(speculation=True)
        a.pressure["a"] = ServingPressure(time.monotonic(), .1, 0, 0)
        payload = {"messages": ["query"], "evidence": ["same-id", "full content"],
                   "options": {"max_tokens": 80}, "tenant": "scope", "model": "a@rev"}
        actual = dict(payload)
        if change:
            actual[change] = "different"
        calls = []
        async def invoke(p):
            calls.append(p)
            return p
        async def authoritative():
            await asyncio.sleep(0)
            return actual
        out, reused = await a.verified_overlap(model="a", predicted_input=payload,
            authoritative_input=authoritative, invoke=invoke, reservation=TokenReservation(5, 80), read_only=True)
        assert out == actual and reused == (change is None)
        assert calls[-1] == actual
        assert len(calls) == (1 if change is None else 2)
        assert payload["evidence"] == ["same-id", "full content"]
    asyncio.run(run())


def test_speculation_error_fallback_and_authoritative_failure_cleanup():
    async def run():
        a = WorkflowAdapter(speculation=True)
        a.pressure["a"] = ServingPressure(time.monotonic(), .1, 0, 0)
        calls = []
        async def invoke(p):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("draft failure")
            return p
        async def authoritative():
            await asyncio.sleep(0)
            return {}
        value, reused = await a.verified_overlap(model="a", predicted_input={},
            authoritative_input=authoritative, invoke=invoke, reservation=TokenReservation(1, 1), read_only=True)
        assert value == {} and not reused and len(calls) == 2
        stopped = asyncio.Event()
        started = asyncio.Event()
        async def blocking(p):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def fail():
            await started.wait()
            raise RuntimeError("authority failed")
        with pytest.raises(RuntimeError):
            await a.verified_overlap(model="a", predicted_input={}, authoritative_input=fail,
                invoke=blocking, reservation=TokenReservation(1, 1), read_only=True)
        assert stopped.is_set()
    asyncio.run(run())


def test_trace_is_bounded_and_categories_union_not_sum_parallel_calls():
    a = WorkflowAdapter(max_spans=1)
    with a.request() as t:
        a.record("usage", time.monotonic(), prompt="PRIVATE", reasoning_tokens=None)
        a.record("usage", time.monotonic())
    assert t.dropped_spans == 1 and "PRIVATE" not in json.dumps(t.spans)
    t = WorkflowTrace(elapsed_s=5, spans=[
        {"kind": "model_rpc_lifetime", "stage": "prepare", "model": "a", "start_s": 1, "end_s": 4},
        {"kind": "model_rpc_lifetime", "stage": "classify", "model": "a", "start_s": 2, "end_s": 5}])
    d = diagnose_trace(t)
    assert d["interval_union_s"]["model_rpc_lifetime"] == 4
    assert not d["reasoning_tokens_observed"]
    assert d["before_first_instrumented_stage_s"] == 1


def test_accuracy_release_requires_answer_evidence_not_only_source_hits():
    old = QualityEvidence(64, True, 1, 1, source_delta_lower=0)
    result = evaluate_quality_gate(old)
    assert not result["release_allowed"]
    assert "expert_answer_evaluation_missing" in result["reasons"]
    complete = QualityEvidence(200, True, 1, 1, 0, 0, 0, 0, True, True, True)
    assert evaluate_quality_gate(complete)["release_allowed"]
    for modified in [replace(complete, correctness_delta_lower=-.001),
                     replace(complete, candidate_success_rate=.999),
                     replace(complete, groundedness_delta_lower=float("nan")),
                     replace(complete, independent_holdout=False)]:
        assert not evaluate_quality_gate(modified)["release_allowed"]
    assert evaluate_quality_gate(replace(complete, correctness_delta_lower=-.01),
        QualityThresholds(correctness_noninferiority_margin=.02))["release_allowed"]


def test_cancelled_active_request_propagates_cancellation():
    async def run():
        a = WorkflowAdapter()
        started = asyncio.Event()
        async def call():
            started.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(a.call("a", call))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
