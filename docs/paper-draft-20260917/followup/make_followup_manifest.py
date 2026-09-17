"""Build the follow-up manifest from the 2026-09-16 review manifest.

Changes: new names/labels, two pods per arm (original-a/b, capability-a/b), the
output-hash attachment, the follow-up runner, and a new results hostPath.
Everything else (image digest, resources, probes, PYTHONPATH, adapter.zip,
questions, topology) is copied unchanged.
"""
import copy, json, sys
from pathlib import Path

NAME = "cnu-review-followup-20260917"
ARMS = {"original-a": "original", "capability-a": "capability", "original-b": "original", "capability-b": "capability"}
src = json.loads(Path(sys.argv[1]).read_text())
here = Path(__file__).resolve().parent
old = src["items"]
cm = copy.deepcopy(next(i for i in old if i["kind"] == "ConfigMap"))
cm["metadata"]["name"] = NAME
cm["metadata"]["labels"]["cnu-trial"] = NAME
cm["data"]["review_dispatch_attachment.py"] = (here/"review_dispatch_attachment.py").read_text()
cm["data"]["run_followup_trial.py"] = (here/"run_followup_trial.py").read_text()
cm["data"]["review-inventory.json"] = json.dumps({"trial": NAME, "base": "cnu-review-20260916",
    "changes": ["two instances per arm", "output_hash/result_hash/law_ids_hash in review events", "run_followup_trial.py"]}, indent=2)
template = next(i for i in old if i["metadata"]["name"].endswith("-original"))
client = next(i for i in old if i["metadata"]["name"].endswith("-client"))
items = [cm]
for arm, mode in ARMS.items():
    pod = copy.deepcopy(template)
    pod["metadata"]["name"] = f"{NAME}-{arm}"
    pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": mode, "cnu-arm": arm})
    c = pod["spec"]["containers"][0]
    c["args"] = ["/opt/cnu/serve_trial.py", "--mode", mode]
    for v in pod["spec"]["volumes"]:
        if "configMap" in v: v["configMap"]["name"] = NAME
    items.append(pod)
pod = copy.deepcopy(client)
pod["metadata"]["name"] = f"{NAME}-client"
pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": "client"})
for v in pod["spec"]["volumes"]:
    if "configMap" in v: v["configMap"]["name"] = NAME
    if "hostPath" in v: v["hostPath"]["path"] = f"/home/axops/{NAME}-results"; v["hostPath"]["type"] = "DirectoryOrCreate"  # pods run on the GPU node
items.append(pod)
out = {"apiVersion": "v1", "kind": "List", "items": items}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
print("wrote", sys.argv[2], "pods:", [i["metadata"]["name"] for i in items if i["kind"] == "Pod"])
