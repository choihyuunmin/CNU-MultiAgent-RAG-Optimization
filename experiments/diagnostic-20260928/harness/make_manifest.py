"""Build isolated diagnostic Pods from the archived comparison Pod specification.

Inputs are private deployment snapshots kept outside the repository. The output
is also kept outside the repository and must be inspected before kubectl apply.
"""
import argparse
import copy
import json
from pathlib import Path

NAME = "cnu-diag-20260928"
OLD = "cnu-cc-20260921"


def build(cm_path, pods_path, harness):
    cm = copy.deepcopy(json.loads(cm_path.read_text())["items"][0])
    pods = json.loads(pods_path.read_text())["items"]
    cm["metadata"]["name"] = NAME
    cm["metadata"]["labels"]["cnu-trial"] = NAME
    for script in ("serve_diag.py", "observe_path.py"):
        cm["data"][script] = (harness / script).read_text()
    out = [cm]
    for mode in ("original", "combined", "client"):
        pod = copy.deepcopy(next(p for p in pods if p["metadata"]["name"] == f"{OLD}-{mode}"))
        pod["metadata"]["name"] = f"{NAME}-{mode}"
        pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": mode})
        for volume in pod["spec"]["volumes"]:
            if "configMap" in volume:
                volume["configMap"]["name"] = NAME
            if "hostPath" in volume:
                volume["hostPath"]["path"] = f"/home/axops/{NAME}-results"
        if mode != "client":
            pod["spec"]["containers"][0]["args"] = ["/opt/cnu/serve_diag.py", "--mode", mode]
        out.append(pod)
    return {"apiVersion": "v1", "kind": "List", "items": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-cm", type=Path, required=True)
    p.add_argument("--base-pods", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = build(args.base_cm, args.base_pods, Path(__file__).parent)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("pods:", [i["metadata"]["name"] for i in manifest["items"] if i["kind"] == "Pod"])


if __name__ == "__main__":
    main()
