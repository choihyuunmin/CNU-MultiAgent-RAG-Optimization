import asyncio
from pathlib import Path
import sys

import pytest

from cnu_rag_optimization.admission import PressureGate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from moleg_scaling_metrics import parse_metrics, summarize_telemetry
from evaluate_moleg_scaling import request, select_cases
from summarize_moleg_scaling import event_spans
from audit_moleg_scaling import audit


def test_gate_cancellation_and_pressure_drain():
    async def run():
        gate = PressureGate(limit=2, maximum=4, adaptive=True)
        order = []
        release = asyncio.Event()
        async def hold(i):
            async with gate.slot():
                order.append(i)
                await release.wait()
        first = asyncio.create_task(hold(0))
        second = asyncio.create_task(hold(1))
        await asyncio.sleep(0)
        cancelled = asyncio.create_task(hold(2))
        fourth = asyncio.create_task(hold(3))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await gate.observe(waiting=3, kv_usage=.9)
        assert gate.limit == 1 and gate.active == 2
        release.set()
        await asyncio.gather(first, second, fourth)
        assert order == [0, 1, 3]
        assert gate.active == 0 and not gate.pending
        await gate.observe(waiting=None, kv_usage=None)
        assert gate.limit == 1
    asyncio.run(run())


def test_metrics_missing_reset_and_worst_engine():
    parsed = parse_metrics('''vllm:kv_cache_usage_perc{engine="0"} 0.6
vllm:kv_cache_usage_perc{engine="1"} 0.8
vllm:num_requests_waiting{engine="0"} 3
vllm:num_requests_waiting{engine="1"} 2
vllm:request_queue_time_seconds_sum{engine="0"} 20
vllm:request_queue_time_seconds_count{engine="0"} 4
vllm:secret{token="never-output"} 1
''')
    assert parsed["values"]["kv_cache_usage_perc"] == .8
    assert parsed["values"]["num_requests_waiting"] == 5
    assert "secret" not in str(parsed)
    after = {"values": {"request_queue_time_seconds_sum": 2,
                         "request_queue_time_seconds_count": 1}}
    result = summarize_telemetry([{"servers": {"orch": parsed}}, {"servers": {"orch": after}}], "orch")
    assert result["counter_reset_detected"]
    assert result["histograms"]["request_queue_time_seconds"]["mean"] is None
    assert result["counters"]["num_preemptions_total"] is None


def test_adaptive_growth_requires_pending_demand_and_three_healthy_samples():
    async def run():
        gate = PressureGate(limit=1, maximum=2, adaptive=True)
        for _ in range(5):
            await gate.observe(waiting=0, kv_usage=.1)
        assert gate.limit == 1
        release = asyncio.Event()
        started = asyncio.Event()
        async def waiting():
            async with gate.slot():
                started.set()
                await release.wait()
        async with gate.slot():
            task = asyncio.create_task(waiting())
            await asyncio.sleep(0)
            for _ in range(2):
                await gate.observe(waiting=0, kv_usage=.1)
            assert gate.limit == 1 and not started.is_set()
            await gate.observe(waiting=0, kv_usage=.1)
            await asyncio.wait_for(started.wait(), 1)
            assert gate.limit == 2 and gate.active == 2
            release.set()
        await task
        assert gate.active == 0
    asyncio.run(run())


def test_stratified_selection_is_unique_and_reproducible():
    cases = [{"case_id": str(i), "kind": "a" if i < 80 else "b"} for i in range(100)]
    selected = select_cases(cases, 20, 7)
    assert len(selected) == len({x["case_id"] for x in selected}) == 20
    assert sum(x["kind"] == "b" for x in selected) == 4
    assert selected == select_cases(cases, 20, 7)


def test_event_spans_cover_latency_with_repeated_stage_and_admission():
    row = {"elapsed_s": 12, "events": [
        {"stage": "preparing", "s": 3}, {"stage": "selecting", "s": 5},
        {"stage": "preparing", "s": 8}, {"stage": "done", "s": 11}]}
    spans = event_spans(row)
    assert spans == {"before_first_event": 3, "preparing": 5, "selecting": 3, "done": 1}
    assert sum(spans.values()) == row["elapsed_s"]


def test_user_latency_includes_gate_wait_and_first_progress_is_not_ttft():
    import httpx
    import time

    async def run():
        gate = PressureGate(limit=1)
        case = {"case_id": "test", "kind": "scenario", "question": "private input"}
        async def respond(req):
            return httpx.Response(200, text='data: {"stage":"progress"}\n\n'
                'data: {"stage":"token","delta":"answer"}\n\n'
                'data: {"stage":"done","result":{"comment":"answer","laws":[]}}\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            async with gate.slot():
                task = asyncio.create_task(request(client, "http://example.test", case,
                                                   gate, time.perf_counter(), 1, "test", 0))
                await asyncio.sleep(.02)
                assert gate.pending
            row = await task
        assert row["ok"] and row["status"] == 200
        assert row["admission_wait_s"] >= .015
        assert row["elapsed_s"] >= row["admission_wait_s"]
        assert row["ttft_s"] >= row["first_event_s"] >= row["admission_wait_s"]
        assert "private input" not in str(row) and "\"answer\"" not in str(row)
        assert gate.active == 0
    asyncio.run(run())


def test_total_deadline_covers_wait_and_records_failure_without_leaking_slot():
    import httpx
    import time

    async def run():
        gate = PressureGate(limit=1)
        case = {"case_id": "test", "kind": "scenario", "question": "private"}
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
            async with gate.slot():
                row = await request(client, "http://example.test", case, gate,
                                    time.perf_counter(), .01, "timeout", 0)
                assert row["ok"] is False and row["error_type"] == "TimeoutError"
                assert row["wire_s"] is None and row["ttft_s"] is None
                assert not gate.pending and gate.active == 1
            assert gate.active == 0
    asyncio.run(run())


def test_valid_stopped_campaign_is_not_called_complete(tmp_path):
    import json
    manifest = {"selected_cases": [{"case_id": "q", "question_sha256": "hash"}],
                "rates": None, "users": [1], "policies": ["baseline"], "repeats": 2}
    row = {"trial": 0, "users": 1, "arrival_rate": None, "policy": "baseline", "repeat": 0,
           "case_id": "q", "question_sha256": "hash", "elapsed_s": 1., "wire_s": 1.,
           "scheduling_lag_s": 0., "admission_wait_s": 0., "ttft_s": .2, "first_event_s": .1, "ok": True}
    trial = {k: row[k] for k in ("trial", "users", "arrival_rate", "policy", "repeat")}
    trial.update(n=1, success=1, metrics={"elapsed_s": {"mean": 1.}})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "requests.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "telemetry.jsonl").write_text("")
    (tmp_path / "summary.json").write_text(json.dumps({"complete": False, "trials": [trial]}))
    result = audit(tmp_path)
    assert result["integrity_passed"] and not result["passed"] and not result["campaign_complete"]
    assert result["planned_trials"] == 2 and result["completed_trials"] == 1
    manifest['planned_order'] = [{'repeat': 1, 'users': 1, 'policy': 'baseline'}]
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert 'trial execution order differs from declared order' in audit(tmp_path)['errors']
    manifest.pop('planned_order')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    row["users"] = 2
    (tmp_path / "requests.jsonl").write_text(json.dumps(row) + "\n")
    assert not audit(tmp_path)["integrity_passed"]


def test_isolated_immediate_emitter_preserves_text_identity_and_updates_aliases(monkeypatch):
    from types import ModuleType
    from launch_moleg_scaling_app import install_immediate_emission

    calls = []
    async def original(queue, delta, request_id="-", *, chunk_chars=5, delay_s=.02):
        calls.append((queue, delta, request_id, chunk_chars, delay_s))
        return delta
    core = ModuleType("core")
    query_loop = ModuleType("core.query_loop")
    helper = ModuleType("core.query_loop._helpers")
    alias = ModuleType("core.emitter_test_alias")
    helper.emit_streaming_delta = alias.emit_streaming_delta = original
    query_loop._helpers = helper
    core.query_loop = query_loop
    for module in (core, query_loop, helper, alias):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    install_immediate_emission()
    assert alias.emit_streaming_delta is helper.emit_streaming_delta
    queue, text = object(), "already complete 합성 텍스트"
    output = asyncio.run(alias.emit_streaming_delta(queue, text, "request-1"))
    assert output == text
    assert calls == [(queue, text, "request-1", 1000, 0)]


def test_isolated_variant_label_does_not_change_request_measurements():
    import io
    import json
    from evaluate_moleg_isolated import VariantSink
    output = io.StringIO()
    row = {'policy': 'baseline', 'users': 16, 'elapsed_s': 5.2,
           'question_sha256': 'fixed-hash', 'admission_wait_s': 0.001}
    VariantSink(output, 'emit16').write(json.dumps(row))
    assert json.loads(output.getvalue()) == row | {'policy': 'emit16'}


def test_resource_histogram_matches_expanded_quantile_and_missing_is_not_zero():
    from summarize_moleg_resources import histogram_stats, host_summary
    from moleg_scaling_metrics import describe
    hist = {0: 3, 50: 2, 90: 4, 100: 1}
    expected = describe([v for v,n in hist.items() for _ in range(n)])
    observed = histogram_stats(hist)
    for key in ['n', 'mean', 'p95', 'max']:
        assert observed[key] == expected[key]
    assert observed['fraction_ge_80'] == .5
    assert histogram_stats({})['mean'] is None
    assert host_summary([]) == {'n': 0}


def test_resource_join_uses_trial_utc_and_separate_gpu_types(tmp_path):
    import json
    import sqlite3
    from summarize_moleg_resources import aggregate
    epoch_ns = 1704067200000000000  # 2024-01-01 UTC
    trial = {'trial': 0, 'policy': 'baseline', 'users': 16, 'repeat': 0,
             'started_utc': '2024-01-01T00:00:01+00:00',
             'ended_utc': '2024-01-01T00:00:02+00:00'}
    (tmp_path/'summary.json').write_text(json.dumps({'trials': [trial]}))
    app = {'trial': 0, 'state': {'active': 4, 'queued': 0, 'max_pipelines': 4,
                               'max_http': 4, 'process_maxrss_kib': 100}}
    (tmp_path/'app-state.jsonl').write_text(json.dumps(app)+'\n')
    host = {'utc': '2024-01-01T00:00:01.5+00:00',
            'host_memory_kib': {'MemAvailable': 1024**2, 'SwapTotal': 0},
            'vm_counters': {'pswpin': 0, 'pswpout': 0, 'pgmajfault': 0, 'oom_kill': 0}}
    host_path = tmp_path/'host.jsonl'
    host_path.write_text(json.dumps(host)+'\n')
    database_path = tmp_path/'nsys.sqlite'
    with sqlite3.connect(database_path) as connection:
        connection.executescript('''CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
          CREATE TABLE TARGET_INFO_GPU_METRICS(typeId,metricId,metricName);
          CREATE TABLE GPU_METRICS(timestamp,typeId,metricId,value);
          CREATE TABLE DIAGNOSTIC_EVENT(message);''')
        connection.execute('INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)', (epoch_ns,))
        for gpu in [0,1]:
            connection.execute('INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?,?)',
                               (256+gpu,18,'DRAM Read Bandwidth [Throughput %]'))
        connection.executemany('INSERT INTO GPU_METRICS VALUES (?,?,?,?)', [
            (0,256,18,99), (1500000000,256,18,50), (1500000000,257,18,20),
            (3000000000,256,18,99)])
    result = aggregate(tmp_path, host_path, database_path)['trials'][0]
    assert result['nsys_window_covered']
    assert result['host']['minimum_mem_available_gib'] == 1
    assert result['nsys_gpus']['0']['DRAM Read Bandwidth [Throughput %]']['mean'] == 50
    assert result['nsys_gpus']['1']['DRAM Read Bandwidth [Throughput %]']['mean'] == 20
    assert result['nsys_coverage']['0']['samples'] == 1
    assert result['nsys_coverage']['0']['first_sample_offset_s'] == .5


def test_scaling_selection_does_not_accept_speed_with_source_regression():
    from select_moleg_scaling_variant import select
    def group(name, time, law=.9, chunk=.8, success=64):
        return {'policy': name, 'n': 64, 'success': success,
                'source_law_hit': law, 'source_chunk_hit': chunk,
                'metrics': {'elapsed_s': {'mean': time}}}
    data = {'complete': True, 'groups': [group('baseline', 60), group('emit', 50),
            group('slots8', 40), group('emit8', 30, law=.8), group('emit16', 20, success=63)]}
    assert select(data) == 'slots8'
    data['complete'] = False
    with pytest.raises(ValueError):
        select(data)


def test_isolated_schedule_counterbalances_arms_at_each_user_load():
    from evaluate_moleg_isolated import trial_schedule
    schedule = list(trial_schedule(['baseline', 'emit16'], [16, 32], 2))
    assert schedule == [(0,16,'baseline'), (0,16,'emit16'),
                        (0,32,'emit16'), (0,32,'baseline'),
                        (1,32,'baseline'), (1,32,'emit16'),
                        (1,16,'emit16'), (1,16,'baseline')]


def test_app_health_does_not_export_private_text_or_ids(tmp_path):
    import json
    from summarize_moleg_app_health import summarize
    path = tmp_path/'private.jsonl'
    path.write_text(json.dumps({'session_id':'scale-private-session-0',
        'stream_input':[{'delta':'confidential text'}],
        'rerank':[{'status':200}], 'guardrail':[{'status':401}],
        'llm':[{'finish_reason':'stop', 'model':'private-endpoint'}]})+'\n')
    result = summarize(path)
    assert result[0]['guardrail_http_status_counts'] == {'401':1}
    assert result[0]['rerank_http_status_counts'] == {'200':1}
    assert not result[0]['complete_64_index_set']
    assert 'private' not in json.dumps(result) and 'confidential' not in json.dumps(result)
