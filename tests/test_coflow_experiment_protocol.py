import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_experiment_preserves_configured_stream_and_model_seed(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "coflow_experiment", Path(__file__).resolve().parents[1] / "scripts" / "run_coflow_experiment.py"
    )
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    captured = {}

    class StopBeforeNetwork(Exception):
        pass

    def preflight(env):
        captured.update(env)
        raise StopBeforeNetwork

    bundle = ModuleType("run_experiment_bundle")
    bundle._load_server_env = lambda path: {"STREAM_ENABLED": "1", "LLM_REQUEST_SEED": "42"}
    bundle._preflight_remote_services = preflight
    bundle._llm_base = bundle._wait_for_server = bundle._stop_owned_process = None
    historical = ModuleType("run_interleaved_400")
    historical.COMMON_EXPERIMENT_FLAGS = {"STREAM_ENABLED": "0", "MAX_CONCURRENT_GENERATE_REQUESTS": "4"}
    monkeypatch.setitem(sys.modules, "run_experiment_bundle", bundle)
    monkeypatch.setitem(sys.modules, "run_interleaved_400", historical)
    monkeypatch.setattr(sys, "path", list(sys.path))
    args = SimpleNamespace(application_root=tmp_path, run_id="protocol-test")
    with pytest.raises(StopBeforeNetwork):
        asyncio.run(experiment.run(args))
    assert captured["STREAM_ENABLED"] == "1"
    assert captured["LLM_REQUEST_SEED"] == "42"
    assert captured["MAX_CONCURRENT_GENERATE_REQUESTS"] == "4"
    assert "STREAM_CHUNK_CHARS" not in captured
    assert "EARLY_EXIT_THINKING_DELAY_MIN_S" not in captured
