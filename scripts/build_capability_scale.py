"""Package a private scaling run from the frozen application trial bundle."""
import argparse
import copy
import json
from pathlib import Path


def build(previous, runner, name):
    result = copy.deepcopy(previous)
    config, original, candidate = result["items"]
    old = config["metadata"]["name"]
    for item in result["items"]:
        item["metadata"]["name"] = item["metadata"]["name"].replace(old, name)
        item["metadata"]["labels"]["cnu-trial"] = name
    config["data"]["run_capability_scale.py"] = runner
    glue = config["data"]["serve_trial.py"]
    anchor = '    app = importlib.import_module("main_server").app\n'
    if glue.count(anchor) != 1:
        raise ValueError("unexpected frozen application glue")
    # Common capacity adjustment, not a candidate-only optimization. Preserve
    # session/auth middleware; only adjust the existing global queue capacity.
    glue = glue.replace(anchor, anchor + '''    from api.middleware.rate_limit import GlobalWaitQueueMiddleware
    for middleware in app.user_middleware:
        if middleware.cls is GlobalWaitQueueMiddleware:
            middleware.kwargs["max_concurrency"] = 100
    from api.concurrency import generate_queue
    emit({"event": "scaling_capacity", "configured": 100})
''')
    config["data"]["serve_trial.py"] = glue
    for pod in (original, candidate):
        pod["spec"]["activeDeadlineSeconds"] = 43200
        pod["spec"]["volumes"][0]["configMap"]["name"] = name
        pod["spec"]["containers"][0]["env"].append({"name": "MAX_CONCURRENT_GENERATE_REQUESTS", "value": "100"})
    # Dedicated client avoids charging the baseline alone for load generation.
    coordinator = copy.deepcopy(original)
    coordinator["metadata"]["name"] = name + "-client"
    coordinator["metadata"]["labels"]["cnu-mode"] = "client"
    container = coordinator["spec"]["containers"][0]
    container["args"] = ["-c", "import time; time.sleep(43200)"]
    container.pop("readinessProbe", None)
    container.pop("ports", None)
    container["resources"] = {"requests": {"cpu": "250m", "memory": "512Mi"}, "limits": {"cpu": "2", "memory": "2Gi"}}
    result["items"].append(coordinator)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    artifact = build(json.loads(a.previous.read_text()), Path(__file__).with_name("run_capability_scale.py").read_text(), a.name)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(artifact, stream)
