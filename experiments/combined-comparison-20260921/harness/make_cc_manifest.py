import base64
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

NAME = "cnu-cc-20260921"
ARMS = ["original", "direct", "reduce", "combined"]
CAPACITY = "100"
KEY_ATTR = os.environ["CNU_APP_KEY_ATTR"]  # name of the application settings attribute holding the API key

src = json.loads(Path(sys.argv[1]).read_text())
here = Path(__file__).resolve().parent
old = src["items"]
cm = copy.deepcopy(next(i for i in old if i["kind"] == "ConfigMap"))
cm["metadata"]["name"] = NAME
cm["metadata"]["labels"]["cnu-trial"] = NAME
cm["data"] = {}
SCRIPTS = ["serve_cc.py", "review_dispatch_attachment.py", "selection_budget_attachment.py", "judge_laws.py",
           "run_sweep_trial.py", "run_rag_experiment.py", "run_scaling_experiment.py", "run_capability_scale.py",
           "collect_vllm_metrics.py"]
for k in SCRIPTS + ["questions.jsonl", "smoke.jsonl", "topology.json"]:
    cm["data"][k] = (here / k).read_text()
zip_bytes = (here / "adapter.zip").read_bytes()
cm["binaryData"] = {"adapter.zip": base64.b64encode(zip_bytes).decode()}
pkg = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((here / "pkg" / "cnu_rag_optimization").glob("*.py"))}
cm["data"]["review-inventory.json"] = json.dumps({
    "trial": NAME, "base_commit": sys.argv[3], "adapter_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
    "package_files_sha256": pkg,
    "script_sha256": {k: hashlib.sha256(cm["data"][k].encode()).hexdigest() for k in SCRIPTS},
    "question_sha256": hashlib.sha256(cm["data"]["questions.jsonl"].encode()).hexdigest(),
    "application_capacity": int(CAPACITY),
    "notes": ["combined comparison: original / direct dispatch / original + selection-input reduction (content 300 chars) / "
              "direct dispatch + reduction, same questions, same frozen image, models, prompts and engines",
              "four arms in a Latin-square rotation per load level; app admission raised to 100 for the load levels"]}, indent=2)
template = next(i for i in old if i["metadata"]["name"].endswith("-original"))
client = next(i for i in old if i["metadata"]["name"].endswith("-client"))
DEADLINE = 172800  # seconds; the four-arm Latin-square run takes about eight hours
items = [cm]
for arm in ARMS:
    pod = copy.deepcopy(template)
    pod["spec"]["activeDeadlineSeconds"] = DEADLINE
    pod["metadata"]["name"] = f"{NAME}-{arm}"
    pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": arm})
    c = pod["spec"]["containers"][0]
    c["args"] = ["/opt/cnu/serve_cc.py", "--mode", arm]
    c["env"] = [e for e in c.get("env", []) if e["name"] not in ("MAX_CONCURRENT_GENERATE_REQUESTS", "CNU_APP_CAPACITY")]
    c["env"] += [{"name": "MAX_CONCURRENT_GENERATE_REQUESTS", "value": CAPACITY}, {"name": "CNU_APP_CAPACITY", "value": CAPACITY}]
    for v in pod["spec"]["volumes"]:
        if "configMap" in v:
            v["configMap"]["name"] = NAME
    items.append(pod)
pod = copy.deepcopy(client)
pod["metadata"]["name"] = f"{NAME}-client"
pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": "client"})
pod["spec"]["activeDeadlineSeconds"] = DEADLINE
c = pod["spec"]["containers"][0]
c["args"] = ["-c", f"import time; time.sleep({DEADLINE})"]
c["env"] = [e for e in c.get("env", []) if e["name"] != "CNU_APP_KEY_ATTR"] + [{"name": "CNU_APP_KEY_ATTR", "value": KEY_ATTR}]
for v in pod["spec"]["volumes"]:
    if "configMap" in v:
        v["configMap"]["name"] = NAME
    if "hostPath" in v:
        v["hostPath"]["path"] = f"/home/axops/{NAME}-results"
        v["hostPath"]["type"] = "DirectoryOrCreate"
items.append(pod)
Path(sys.argv[2]).write_text(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2, ensure_ascii=False) + "\n")
print("wrote", sys.argv[2], [i["metadata"]["name"] for i in items if i["kind"] == "Pod"], "bytes:", Path(sys.argv[2]).stat().st_size)
