#!/usr/bin/env python3
"""Build source-only ConfigMap and isolated trial Pods from a pinned application image."""

import argparse
import base64
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(topology, *, namespace, name, node, image):
    for value in (namespace, name, node):
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value) or len(value) > 50:
            raise ValueError("short DNS names required")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("immutable image digest required")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((ROOT / "src" / "cnu_rag_optimization").glob("*.py")):
            bundle.writestr("cnu_rag_optimization/" + path.name, path.read_bytes())
    files = {filename: (ROOT / "scripts" / filename).read_text()
             for filename in ("serve_embedded_adapter.py", "serve_scaling_adapter.py")}
    files["topology.json"] = json.dumps(topology, indent=2)
    labels = {"app": "cnu-rag-adapter-trial", "cnu-trial": name}
    configmap = {"apiVersion": "v1", "kind": "ConfigMap",
                 "metadata": {"namespace": namespace, "name": name, "labels": labels},
                 "data": files, "binaryData": {"adapter.zip": base64.b64encode(archive.getvalue()).decode()}}
    pods = []
    for mode in ("original", "delivery", "network"):
        pods.append({"apiVersion": "v1", "kind": "Pod",
            "metadata": {"namespace": namespace, "name": name + "-" + mode,
                         "labels": {**labels, "cnu-mode": mode}},
            "spec": {"nodeName": node, "restartPolicy": "Never", "activeDeadlineSeconds": 7200,
                "automountServiceAccountToken": False,
                "containers": [{"name": "adapter-trial", "image": image, "imagePullPolicy": "Never",
                    "command": ["/app/.venv/bin/python"],
                    "args": ["/opt/cnu/serve_embedded_adapter.py", "--application-root", "/app",
                             "--topology", "/opt/cnu/topology.json", "--mode", mode,
                             "--host", "0.0.0.0", "--port", "28000"],
                    "env": [{"name": "PYTHONPATH", "value": "/opt/cnu/adapter.zip:/app/src"},
                            {"name": "PYTHONUNBUFFERED", "value": "1"}],
                    "ports": [{"containerPort": 28000}],
                    "resources": {"requests": {"cpu": "500m", "memory": "4Gi"},
                                  "limits": {"cpu": "2", "memory": "8Gi"}},
                    "readinessProbe": {"httpGet": {"path": "/health/live", "port": 28000},
                                       "initialDelaySeconds": 3, "periodSeconds": 5},
                    "volumeMounts": [{"name": "adapter", "mountPath": "/opt/cnu", "readOnly": True}]}],
                "volumes": [{"name": "adapter", "configMap": {"name": name}}]}})
    result = {"apiVersion": "v1", "kind": "List", "items": [configmap, *pods]}
    if len(json.dumps(configmap).encode()) > 900_000:
        raise ValueError("adapter ConfigMap exceeds conservative size bound")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    payload = build(json.loads(args.topology.read_text()), namespace=args.namespace, name=args.name,
                    node=args.node, image=args.image)
    with args.output.open("x") as target:
        json.dump(payload, target, indent=2)
        target.write("\n")
    print(f"Built {len(payload['items']) - 1} isolated Pods and one ConfigMap; no Service/Ingress or production mutation.")


if __name__ == "__main__":
    main()
