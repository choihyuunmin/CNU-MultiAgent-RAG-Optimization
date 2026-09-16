import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("draft_stage_pilot", Path(__file__).parents[1] / "scripts/run_draft_stage_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_fixed_arms_change_only_speculative_config():
    base = {"model": "local-frozen-model", "max_model_len": 4096, "gpu_memory_utilization": .4}
    assert pilot.arm_config(base, 0) == base
    for k in [2, 4, 8, 16]:
        candidate = pilot.arm_config(base, k)
        assert candidate.pop("speculative_config")["num_speculative_tokens"] == k
        assert candidate == base
    assert "speculative_config" not in base


def test_frozen_inputs_preserve_every_token_and_sampling_option():
    cases = [{"case_id": "a", "prompt_token_ids": [1, 2, 3],
              "sampling_params": {"max_tokens": 300, "temperature": 0, "seed": 7}}]
    assert pilot.validate_cases(cases) is cases
    with pytest.raises(ValueError, match="unique"):
        pilot.validate_cases(cases + cases)


@pytest.mark.parametrize("tokens", [[], [True], [-1], "already rendered"])
def test_unsupported_input_is_rejected(tokens):
    with pytest.raises(ValueError, match="tokenized"):
        pilot.validate_cases([{"case_id": "a", "prompt_token_ids": tokens,
                               "sampling_params": {"max_tokens": 300}}])


def test_no_silent_new_output_limit():
    with pytest.raises(ValueError, match="max_tokens"):
        pilot.validate_cases([{"case_id": "a", "prompt_token_ids": [1], "sampling_params": {}}])


def test_full_runner_records_truncation_and_all_requests(tmp_path, monkeypatch):
    cases = [{"case_id": f"q{i}", "prompt_token_ids": [i + 1],
              "sampling_params": {"max_tokens": 300, "seed": 7}} for i in range(4)]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "warmup.json").write_text(json.dumps([dict(cases[0], case_id="warmup")]))
    (tmp_path / "engine.json").write_text(json.dumps({"model": "frozen"}))
    observed = []
    class FakeLLM:
        def __init__(self, **config):
            observed.append(config)
        def generate(self, prompts, sampling_params, use_tqdm):
            assert all(p["max_tokens"] == 300 and p["seed"] == 7 for p in sampling_params)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], finished=True,
                outputs=[SimpleNamespace(finish_reason="length" if p["prompt_token_ids"] == [4] else "stop",
                                         token_ids=[9, 8])],
                metrics=SimpleNamespace(arrival_time=1., first_token_time=2., finished_time=3.))
                for p in prompts]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: k))
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda _: "test-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["pilot", "--engine-config", str(tmp_path / "engine.json"),
        "--cases", str(tmp_path / "cases.json"), "--warmup-cases", str(tmp_path / "warmup.json"),
        "--k", "4", "--levels", "1", "4", "--repeats", "2", "--output", str(output)])
    pilot.main()
    status = json.loads((output / "status.json").read_text())
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert status["completed_measured_requests"] == status["expected_measured_requests"] == 16
    assert len(rows) == 16 and sum(r["ok"] for r in rows) == 12
    assert observed[0]["speculative_config"]["num_speculative_tokens"] == 4
    assert not status["promote_to_full_experiment"]
