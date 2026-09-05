import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "coflow_summary", Path(__file__).resolve().parents[1] / "scripts" / "summarize_coflow_run.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_quality_gate_uses_frozen_protocol_and_distinguishes_missing_evaluation():
    gate = dict.fromkeys(("success_rate", "document_recall", "ndcg10", "top1", "answer_similarity", "fidelity"), 0.99)
    quality = dict.fromkeys(("success_rate", "law_recall_mean", "law_ndcg10_mean", "top1_agreement_mean",
                             "comment_similarity_mean", "baseline_fidelity_mean"), "0.98")
    assert summary.check_quality_gate({}, gate) == "not_evaluated"
    assert summary.check_quality_gate(quality, gate) == "fail"
    assert summary.check_quality_gate(quality, dict.fromkeys(gate, 0.98)) == "pass"
    quality["baseline_fidelity_mean"] = "nan"
    assert summary.check_quality_gate(quality, dict.fromkeys(gate, 0.98)) == "fail"


def test_paired_latency_uses_ratio_of_means_and_aligned_blocks():
    control = {
        "q1": {"status": "ok", "block_index": 0, "duration_ms": 100},
        "q2": {"status": "ok", "block_index": 1, "duration_ms": 300},
    }
    candidate = {key: {**row, "duration_ms": row["duration_ms"] / 2} for key, row in control.items()}
    result = summary.paired_bootstrap(control, candidate, iterations=100)
    assert result["mean_reduction_percent"] == 50
    assert result["block_bootstrap_ci95"] == [50, 50]
    assert result["blocks"] == 2
    candidate["q2"]["block_index"] = 0
    with pytest.raises(ValueError, match="mismatched question block"):
        summary.paired_bootstrap(control, candidate)


def test_missing_question_cannot_silently_enter_paired_comparison():
    with pytest.raises(ValueError, match="do not align"):
        summary.paired_bootstrap({"q": {}}, {})
    with pytest.raises(ValueError, match="do not align"):
        summary.quality_difference({"q": {}}, {}, {"q": 0})


def test_quality_difference_is_signed_excess_agreement_in_percentage_points():
    fields = ("law_recall", "law_ndcg10", "top1_agreement", "comment_similarity", "baseline_fidelity")
    reference = {f"q{i}": dict.fromkeys(fields, "0.95") for i in range(4)}
    candidate = {f"q{i}": dict.fromkeys(fields, "0.90") for i in range(4)}
    result = summary.quality_difference(reference, candidate, {f"q{i}": i // 2 for i in range(4)}, iterations=100)
    assert result["paired_questions"] == 4
    assert result["blocks"] == 2
    for value in result["metrics"].values():
        assert value["mean_difference_percentage_points"] == pytest.approx(-5)
        assert value["block_bootstrap_ci95"] == pytest.approx([-5, -5])


def test_timeline_does_not_double_count_parallel_calls_or_waits():
    trace = {"duration_ms": 100, "llm_calls": [
        {"start_offset_ms": 10, "end_offset_ms": 70},
        {"start_offset_ms": 30, "end_offset_ms": 80},
    ], "spans": [
        {"name": "coflow_admission", "start_offset_ms": 0, "end_offset_ms": 20},
        {"name": "coflow_admission", "start_offset_ms": 40, "end_offset_ms": 90},
        {"name": "irrelevant", "start_offset_ms": 0, "end_offset_ms": 100},
    ]}
    regions = summary.timeline_partition(trace)
    assert regions == {"llm_api_only_ms": 20, "local_admission_only_ms": 20,
                       "llm_api_and_admission_ms": 50, "neither_observed_ms": 10}
    assert sum(regions.values()) == 100


def test_timeline_clamps_intervals_to_request_and_handles_missing_offsets():
    trace = {"duration_ms": 20, "llm_calls": [
        {"start_offset_ms": -5, "end_offset_ms": 100}, {"duration_ms": 10},
    ]}
    regions = summary.timeline_partition(trace)
    assert regions["llm_api_only_ms"] == 20
    assert sum(regions.values()) == 20


def test_timeline_retains_interrupted_call_busy_interval_from_wrapper():
    trace = {"duration_ms": 100, "events": [{"name": "coflow_admission", "offset_ms": 90,
             "attributes": {"service_ms": 80, "success": False}}]}
    regions = summary.timeline_partition(trace)
    assert regions["llm_api_only_ms"] == 80
    assert regions["neither_observed_ms"] == 20


def test_cancelled_unrecorded_calls_and_timeouts_are_not_hidden_by_http_success(tmp_path):
    log = tmp_path / "server.log"
    trace = {"run_id": "measured", "request_id": "abc", "duration_ms": 100,
             "llm_calls": [{"call_id": "finished", "duration_ms": 10, "status": "ok"}],
             "events": [{"name": "coflow_admission", "attributes": {
                 "wait_ms": 1, "service_ms": 80, "success": False}}]}
    log.write_text(
        "INFO [application] [warmup] LLM sent | call=excluded role=worker\n"
        "ERROR [application] [warmup] tool_loop LLM timeout after 120s\n"
        "INFO [application] [abc] LLM sent | call=finished role=worker\n"
        "INFO [application] [abc] LLM sent | call=interrupted role=worker\n"
        "ERROR [application] [abc] tool_loop LLM timeout after 120s\n"
        "RAG_TRACE_V1 " + json.dumps(trace) + "\n", encoding="utf-8")
    result = summary.trace_stats(log, "measured", [])
    assert result["llm_calls"] == 1
    assert result["observed_llm_attempts_lower_bound"] == 2
    assert result["sent_calls_missing_trace"] == 1
    assert result["requests_with_tool_loop_timeout"] == 1
    assert result["interrupted_admissions"] == 1
    assert result["requests_with_interrupted_admissions"] == 1
