import importlib.util
from pathlib import Path


def test_scale_bundle_preserves_original_and_isolates_capacity():
    source = Path(__file__).resolve().parents[1] / "scripts/build_capability_scale.py"
    spec = importlib.util.spec_from_file_location("scale_bundle", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"metadata": {"name": "old", "labels": {}}, "data": {
        "serve_trial.py": '    app = importlib.import_module("main_server").app\n',
        "run_rag_experiment.py": "unchanged", "questions.jsonl": "unchanged"}}
    pod = {"metadata": {"name": "old-original", "labels": {}}, "spec": {
        "volumes": [{"configMap": {"name": "old"}}],
        "containers": [{"env": [], "args": ["original"], "readinessProbe": {}, "ports": []}]}}
    import copy
    candidate = copy.deepcopy(pod)
    candidate["metadata"]["name"] = "old-capability"
    candidate["spec"]["containers"][0]["args"] = ["capability"]
    previous = {"items": [config, pod, candidate]}
    result = module.build(previous, "runner", "new")
    assert previous["items"][1]["spec"]["containers"][0]["env"] == []
    assert result["items"][0]["data"]["run_rag_experiment.py"] == "unchanged"
    assert result["items"][0]["data"]["questions.jsonl"] == "unchanged"
    assert len(result["items"]) == 4
    for p in result["items"][1:3]:
        assert p["spec"]["containers"][0]["env"] == [{"name": "MAX_CONCURRENT_GENERATE_REQUESTS", "value": "100"}]
    assert result["items"][3]["metadata"]["name"] == "new-client"
