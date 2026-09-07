import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from cnu_rag_optimization import CompletionPolicy, CompletionRouter, NetworkHarness, ReceiverSpec, ReplicaSpec

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_scaling_experiment")
adapter = load("serve_scaling_adapter")
analysis = load("analyze_scaling_experiment")


def test_same_real_rows_selected_deterministically(tmp_path):
    rows = [{"id": f"{level}-{i}", "country": "일본", "query": "일본 법령", "complexity": level}
            for level in ("L1", "L2", "L3", "L4") for i in range(4)]
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    first = runner.select_questions(path, 8, 7)
    assert first == runner.select_questions(path, 8, 7)
    assert len(first) == 8
    assert all(row in rows for row in first)
    with pytest.raises(ValueError, match="not enough"):
        runner.select_questions(path, 20, 7)
    path.write_text("\n".join(json.dumps(r) for r in rows + rows[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        runner.select_questions(path, 8, 7)


def test_gate_preserves_stream_bytes_and_holds_slot_to_end(monkeypatch):
    async def run():
        events, received, forwarded = [], [], []
        monkeypatch.setattr(adapter, "emit", lambda prefix, value: events.append((prefix, value)))
        finish = asyncio.Event()
        first_started = asyncio.Event()

        async def app(scope, receive, send):
            received.append(scope["case"])
            if scope["case"] == 1:
                first_started.set()
                await finish.wait()
            await send({"type": "http.response.body", "body": b"unchanged", "more_body": False})

        async def receive():
            return {"type": "http.request", "body": b"original"}

        async def send(message):
            forwarded.append(message)

        gate = adapter.observed_gate()(app, max_concurrency=1)
        scopes = [{"type": "http", "path": "/api/generate/stream", "case": i,
                   "headers": [(b"x-experiment-case-id", str(i).encode())]} for i in (1, 2)]
        first = asyncio.create_task(gate(scopes[0], receive, send))
        await first_started.wait()
        second = asyncio.create_task(gate(scopes[1], receive, send))
        await asyncio.sleep(0)
        assert received == [1]
        finish.set()
        await asyncio.gather(first, second)
        assert received == [1, 2]
        assert len(forwarded) == 2 and all(r["body"] == b"unchanged" for r in forwarded)
        assert all(e[1]["outer_wait_ms"] >= 0 for e in events)
    asyncio.run(run())


def test_gate_override_and_non_generate_passthrough():
    async def run():
        calls = []
        async def app(scope, receive, send):
            calls.append(scope)
        gate = adapter.observed_gate(request_window=32)(app, max_concurrency=4)
        assert gate.limit == 32
        scope = {"type": "lifespan"}
        await gate(scope, None, None)
        assert calls == [scope]
    asyncio.run(run())


def test_parallel_call_union_and_counter_resets():
    assert analysis.interval_union_ms([(0, 10), (2, 8), (5, 15), (20, 22)]) == 17
    assert analysis.counter_delta([{"n": 1}, {"n": 3}, {"n": 4}], "n") == 3
    assert analysis.counter_delta([{"n": 3}, {"n": 1}, {"n": 4}], "n") is None
    assert analysis.counter_delta([{}, {"n": 4}], "n") is None
    assert analysis.counter_delta([], "n") is None


def test_tagged_json_with_appended_logs_is_not_discarded():
    text = ('CNU_INGRESS_V1 {"outer_wait_ms": 1} ordinary log\n'
            'CNU_COMPLETION_V1 {"note": "RAG_TRACE_V1 {}"} '
            'CNU_COMPLETION_V1 {"event": "released"}\n'
            'RAG_TRACE_V1 {"truncated":\n')
    records, malformed = analysis.parse_tagged_records(text)
    assert records["CNU_INGRESS_V1"] == [{"outer_wait_ms": 1}]
    assert len(records["CNU_COMPLETION_V1"]) == 2
    assert records["RAG_TRACE_V1"] == []
    assert malformed == 1


def test_unaligned_or_duplicate_questions_not_silently_dropped():
    with pytest.raises(ValueError, match="duplicate"):
        analysis.indexed([{"question_id": "q1"}, {"question_id": "q1"}])
    with pytest.raises(ValueError, match="unaligned"):
        analysis.agreement([{"question_id": "q1"}], [{"question_id": "q2"}], None)


def test_metric_collector_captures_start_and_final_snapshot(tmp_path, monkeypatch):
    async def run():
        stop, unsafe, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        class Response:
            text = "metrics"
            def raise_for_status(self):
                pass

        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url):
                calls.append(url)
                return Response()

        monkeypatch.setattr(runner.httpx, "AsyncClient", Client)
        path = tmp_path / "metrics.jsonl"
        task = asyncio.create_task(runner.collect({"receivers": {"r": {"metrics": "http://example/metrics"}}},
            lambda text: [{"metric": "vllm:num_requests_running", "value": 0}], path, stop, unsafe, ready))
        await ready.wait()
        stop.set()
        await task
        assert len(calls) == 2
        assert len(analysis.rows(path)) == 2
        assert not unsafe.is_set()
    asyncio.run(run())


def test_display_adapter_preserves_chunks_and_existing_function_aliases():
    async def run():
        async def delta(queue, text, *, chunk_chars=5, delay_s=0.02):
            for i in range(0, len(text), chunk_chars):
                queue.append(text[i:i + chunk_chars])
        async def fake(queue, text, *, chunk_chars=5, delay_s=0.02):
            await delta(queue, text, chunk_chars=chunk_chars, delay_s=delay_s)
        helpers = SimpleNamespace(emit_streaming_delta=delta, emit_fake_stream=fake)
        route = SimpleNamespace(EARLY_EXIT_THINKING_DELAY_MIN_S=.6, EARLY_EXIT_THINKING_DELAY_MAX_S=1.4)
        alias = fake
        before, after = [], []
        await alias(before, "원문과 순서를 그대로 유지합니다.")
        original = adapter.remove_display_waits(helpers, route)
        await alias(after, "원문과 순서를 그대로 유지합니다.")
        assert before == after
        assert alias is helpers.emit_fake_stream
        assert alias.__kwdefaults__ == {"chunk_chars": 5, "delay_s": 0.0}
        assert original["emit_fake_stream"]["delay_s"] == .02
        assert route.EARLY_EXIT_THINKING_DELAY_MAX_S == 0
    asyncio.run(run())


def test_display_adapter_fails_closed_on_unknown_helper_signature():
    async def unknown(queue, text):
        pass
    helpers = SimpleNamespace(emit_streaming_delta=unknown, emit_fake_stream=unknown)
    with pytest.raises(RuntimeError, match="unsupported display helper"):
        adapter.remove_display_waits(helpers, None)


def test_missing_done_response_not_counted_as_valid_answer():
    assert not analysis.valid_response({"status": "ok", "response": None})
    assert analysis.valid_response({"status": "ok", "response": {"laws": [], "comment": "검색 결과 없음"}})


def test_paired_latency_keeps_failures_and_joins_by_id():
    baseline = [{"question_id": "a", "duration_ms": 100, "status": "ok"},
                {"question_id": "b", "duration_ms": 200, "status": "ok"}]
    candidate = [{"question_id": "b", "duration_ms": 250, "status": "error"},
                 {"question_id": "a", "duration_ms": 50, "status": "ok"}]
    result = analysis.paired_latency(baseline, candidate)
    assert result["questions"] == 2
    assert result["includes_failed_attempts"]
    assert result["candidate_minus_reference_mean_ms"] == 0
    assert result["candidate_faster_questions"] == result["candidate_slower_questions"] == 1
    with pytest.raises(ValueError, match="unaligned"):
        analysis.paired_latency(baseline, candidate[:1])


def test_idle_preflight_requires_two_quiet_samples_and_does_not_infer(monkeypatch):
    async def run():
        requests, sleeps = [], []
        class Response:
            text = "gauges"
            def raise_for_status(self):
                pass
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url):
                requests.append(url)
                return Response()
        async def sleep(seconds):
            sleeps.append(seconds)
        monkeypatch.setattr(runner.httpx, "AsyncClient", Client)
        monkeypatch.setattr(runner.asyncio, "sleep", sleep)
        def parse(text):
            return [{"metric": "vllm:num_requests_running", "value": 0},
                    {"metric": "vllm:num_requests_waiting", "value": 0}]
        topology = {"receivers": {"r": {"metrics": "http://example/metrics"}}}
        result = await runner.wait_for_idle(topology, parse, 10)
        assert result["quiet_samples"] == 2
        assert requests == ["http://example/metrics"] * 2
        assert sleeps == [1]
        with pytest.raises(RuntimeError, match="remain busy"):
            await runner.wait_for_idle(topology, lambda text: [{**r, "value": 1} for r in parse(text)], 0)
        with pytest.raises(RuntimeError, match="missing engine gauges"):
            await runner.wait_for_idle(topology, lambda text: [], 10)
    asyncio.run(run())


def test_network_stream_adapter_forwards_original_kwargs_once():
    async def run():
        calls = []
        chunk = object()
        async def original(**kwargs):
            calls.append(kwargs)
            yield chunk
        router = CompletionRouter([ReceiverSpec("r")], [ReplicaSpec("a", "r", "m", 1)], CompletionPolicy(ordering="fair"))
        harness = NetworkHarness(router)
        trace = SimpleNamespace(current_trace=lambda: {"request_id": "root"})
        wrapped = adapter.network_stream(original, harness, trace)
        parameters = {"model": "unchanged", "messages": [{"role": "user", "content": "unchanged"}],
                      "temperature": 0, "seed": 42, "max_tokens": 3000, "extra_body": {"x": 1},
                      "_log_request_id": "log-id"}
        assert [x async for x in wrapped(**parameters)] == [chunk]
        assert calls == [parameters]
        assert calls[0]["messages"] is parameters["messages"]
    asyncio.run(run())
