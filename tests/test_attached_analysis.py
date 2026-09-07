import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def analysis(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("attached_analysis", scripts / "analyze_attached_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_filters_smoke_and_exposes_missing_records(tmp_path, analysis):
    path = tmp_path / "network.log"
    entries = [
        {"event": "request_finished", "case_id": "q-c1", "root_id": "smoke"},
        {"event": "request_finished", "case_id": "q-c16", "root_id": "main"},
        {"event": "llm_finished", "case_id": "q-c16", "root_id": "main", "model": "m",
         "role": "classify", "streaming": False, "managed": True, "status": "ok",
         "wait_ms": 4, "api_ms": 100, "completion_tokens": 20, "chunks": 0},
        {"event": "llm_finished", "case_id": "q-c16", "root_id": "main", "model": "m",
         "role": "classify", "streaming": False, "managed": True, "status": "cancelled",
         "error_type": "CancelledError", "wait_ms": 8, "api_ms": None, "chunks": 0},
    ]
    path.write_text("\n".join("CNU_APP_ADAPTER_V1 " + json.dumps(e) for e in entries))
    result = analysis.trace_metrics(path, [{"case_id": "q-c16"}, {"case_id": "missing-c16"}])
    assert result["request_traces"] == 1
    assert result["missing_request_traces"] == ["missing-c16"]
    assert result["internal_call_failures"] == {"CancelledError": 1}
    model = result["calls_by_model_role"][0]
    assert model["calls"] == 2
    assert model["successful_calls"] == 1
    assert model["admission_wait_mean_ms"] == 6
    assert model["api_mean_ms"] == 100
    assert model["completion_tokens_observed_calls"] == 1
    assert model["completion_tokens_mean"] == 20


def test_gpu_uses_client_wall_interval_not_other_runs(analysis):
    samples = [{"timestamp": f"2026-09-07T00:00:{second:02}+00:00", "index": "0", "name": "test GPU",
                "gpu_utilization_percent": value, "memory_used_mib": 20, "memory_total_mib": 40,
                "memory_utilization_percent": "N/A", "power_draw_w": 10}
               for second, value in ((0, 99), (10, 20), (20, 40), (30, 99))]
    result = analysis.gpu_metrics(samples, [{"started_at": "2026-09-07T00:00:10Z",
                                            "completed_at": "2026-09-07T00:00:20Z"}])["0"]
    assert result["samples"] == 2
    assert result["gpu_utilization_percent_mean"] == 30
    assert result["memory_utilization_percent_mean"] is None


def test_duplicate_case_traces_are_not_hidden(tmp_path, analysis):
    path = tmp_path / "original.log"
    event = {"event": "request_finished", "case_id": "q", "root_id": "r"}
    path.write_text(("CNU_APP_ADAPTER_V1 " + json.dumps(event) + "\n") * 2)
    result = analysis.trace_metrics(path, [{"case_id": "q"}])
    assert result["duplicate_case_traces"] == ["q"]


def test_quality_audit_reports_omissions_and_excludes_empty_references_only_in_supplement(analysis):
    def row(key, ids):
        return {"question_id": key, "status": "ok", "law_ids": ids,
                "response": {"laws": [], "comment": "unit test"}}
    evaluation = SimpleNamespace(law_metrics=lambda a, b: {"law_recall": len(set(a) & set(b)) / len(set(a))})
    control = [row("one", ["A", "A", "B"]), row("empty", [])]
    candidate = [row("one", ["B"]), row("empty", [])]
    audit = analysis.quality_audit(control, candidate, evaluation)
    assert audit["successful_nonempty_reference_pairs"] == 1
    assert audit["law_recall_nonempty_reference_mean"] == .5
    assert audit["top1_changed_question_ids"] == ["one"]
    assert audit["reference_law_omissions"][0]["missing_reference_law_ids"] == ["A"]
    with pytest.raises(ValueError, match="unaligned"):
        analysis.quality_audit(control, candidate[:1], evaluation)
