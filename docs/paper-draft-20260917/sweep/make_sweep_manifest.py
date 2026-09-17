"""Build the load-sweep manifest from the 2026-09-16 review manifest.

Changes: new names/labels, serve_sweep.py (admission capacity 100), the output-hash
attachment, run_sweep_trial.py, env MAX_CONCURRENT_GENERATE_REQUESTS / CNU_APP_CAPACITY,
and a new results hostPath. Image digest, resources, probes and adapter.zip are unchanged.
"""
import copy, json, sys
from pathlib import Path

NAME = "cnu-sweep-20260917"
ARMS = ["original", "direct", "capability"]
CAPACITY = "100"
src = json.loads(Path(sys.argv[1]).read_text())
here = Path(__file__).resolve().parent
old = src["items"]
cm = copy.deepcopy(next(i for i in old if i["kind"] == "ConfigMap"))
cm["metadata"]["name"] = NAME
cm["metadata"]["labels"]["cnu-trial"] = NAME
cm["data"]["review_dispatch_attachment.py"] = (here/"review_dispatch_attachment.py").read_text()
cm["data"]["serve_sweep.py"] = (here/"serve_sweep.py").read_text()
cm["data"]["run_sweep_trial.py"] = (here/"run_sweep_trial.py").read_text()
cm["data"]["review-inventory.json"] = json.dumps({"trial": NAME, "base": "cnu-review-20260916",
    "changes": ["serve_sweep.py raises admission capacity to 100 (as 2026-09-13 sweep)",
                "output_hash/result_hash/law_ids_hash in review events", "run_sweep_trial.py levels x rotated rounds"]}, indent=2)
template = next(i for i in old if i["metadata"]["name"].endswith("-original"))
client = next(i for i in old if i["metadata"]["name"].endswith("-client"))
items = [cm]
for arm in ARMS:
    pod = copy.deepcopy(template)
    pod["metadata"]["name"] = f"{NAME}-{arm}"
    pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": arm})
    c = pod["spec"]["containers"][0]
    c["args"] = ["/opt/cnu/serve_sweep.py", "--mode", arm]
    c["env"] = [e for e in c.get("env", []) if e["name"] not in ("MAX_CONCURRENT_GENERATE_REQUESTS", "CNU_APP_CAPACITY")]
    c["env"] += [{"name": "MAX_CONCURRENT_GENERATE_REQUESTS", "value": CAPACITY}, {"name": "CNU_APP_CAPACITY", "value": CAPACITY}]
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
Path(sys.argv[2]).write_text(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2, ensure_ascii=False)+"\n")
print("wrote", sys.argv[2], [i["metadata"]["name"] for i in items if i["kind"] == "Pod"])
