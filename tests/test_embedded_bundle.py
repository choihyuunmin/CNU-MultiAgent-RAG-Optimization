import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("builder", Path(__file__).resolve().parents[1] / "scripts/build_embedded_adapter_bundle.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_bundle_is_pinned_isolated_and_contains_no_original_application_or_credentials():
    result = builder.build({"aliases": {}, "receivers": {}, "application_source_sha256": {}},
        namespace="trial", name="cnu-example", node="node1", image="example/app@sha256:" + "a" * 64)
    assert [x["kind"] for x in result["items"]] == ["ConfigMap", "Pod", "Pod", "Pod"]
    config, *pods = result["items"]
    assert set(config["data"]) == {"serve_embedded_adapter.py", "serve_scaling_adapter.py", "topology.json"}
    for pod in pods:
        spec = pod["spec"]
        assert pod["metadata"]["labels"]["app"] == "cnu-rag-adapter-trial"
        assert spec["automountServiceAccountToken"] is False
        assert not spec.get("hostNetwork")
        container = spec["containers"][0]
        assert container["imagePullPolicy"] == "Never"
        assert all("hostPort" not in p for p in container["ports"])
        assert container["resources"]["limits"] == {"cpu": "2", "memory": "8Gi"}
        assert {e["name"] for e in container["env"]} == {"PYTHONPATH", "PYTHONUNBUFFERED"}


def test_mutable_image_and_invalid_name_rejected():
    with pytest.raises(ValueError, match="immutable"):
        builder.build({}, namespace="trial", name="test", node="node1", image="app:latest")
    with pytest.raises(ValueError, match="DNS"):
        builder.build({}, namespace="trial", name="bad/name", node="node1", image="app@sha256:" + "a" * 64)
