import importlib.util
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("capability_scale", scripts / "run_capability_scale.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metrics(queue=100):
    return {"vllm:num_requests_running": 100, "vllm:num_requests_waiting": queue,
            "vllm:kv_cache_usage_perc": 1.0}


def test_full_kv_and_100_waiting_do_not_abort():
    safety = module.HighLoadSafety()
    assert safety.observe("a", metrics(), 0) is None
    assert safety.observe("a", metrics(), 300) is None


def test_sustained_queue_and_recovery():
    safety = module.HighLoadSafety()
    assert safety.observe("a", metrics(257), 0) is None
    assert safety.observe("a", metrics(257), 29) is None
    assert "queue_above_256" in safety.observe("a", metrics(257), 30)
    assert safety.observe("a", metrics(), 31) is None
    assert safety.observe("a", metrics(257), 32) is None


def test_single_receiver_telemetry_loss_aborts():
    safety = module.HighLoadSafety()
    assert safety.observe("a", None, 0) is None
    assert safety.observe("b", metrics(), 30) is None
    assert "telemetry_unavailable" in safety.observe("a", {}, 30)


def test_failure_window():
    ok, error = {"status": "ok"}, {"status": "error"}
    assert module.response_failure_reason([error] * 4 + [ok] * 15) is None
    assert module.response_failure_reason([error] * 4 + [ok] * 16)
    assert module.response_failure_reason([error] * 4 + [ok] * 20) is None


def test_background_diagnostics_require_safety():
    import asyncio
    from types import SimpleNamespace
    import pytest
    with pytest.raises(ValueError, match="require high-load safety"):
        asyncio.run(module.main(SimpleNamespace(diagnostic_background=True, high_load_safety=False)))


def test_performance_report_rejects_background_diagnostics(tmp_path):
    import json
    import pytest
    from analyze_capability_scale import summarize
    (tmp_path / "protocol.json").write_text(json.dumps({"performance_comparison_allowed": False}))
    with pytest.raises(ValueError, match="performance comparison forbidden"):
        summarize(tmp_path, None)
