"""Dependency-scheduling trial manifest: barrier + ready app pods + client, built from the stage-1 template."""
import base64, copy, hashlib, json, sys
from pathlib import Path
NAME = "cnu-deptrial-20260918"; ARMS = ["barrier", "ready"]
NODES_SHA = "75c550ca5b0727669624bdb6a81a1fd480890026f113201ce0b653fd6e942d12"
src = json.loads(Path(sys.argv[1]).read_text()); here = Path(__file__).resolve().parent
old = src["items"]; cm = copy.deepcopy(next(i for i in old if i["kind"] == "ConfigMap"))
cm["metadata"]["name"] = NAME; cm["metadata"]["labels"]["cnu-trial"] = NAME
cm["data"] = {}
for k in ["serve_dependency_trial.py", "moleg_program_overlay.py", "program_harness_attachment.py", "run_followup_trial.py",
          "run_rag_experiment.py", "run_scaling_experiment.py", "run_capability_scale.py", "collect_vllm_metrics.py",
          "questions.jsonl", "smoke20.jsonl", "topology.json"]:
    cm["data"][k] = (here / k).read_text()
zip_bytes = (here / "adapter.zip").read_bytes(); cm["binaryData"] = {"adapter.zip": base64.b64encode(zip_bytes).decode()}
pkg = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((here / "pkg" / "cnu_rag_optimization").glob("*.py"))}
cm["data"]["review-inventory.json"] = json.dumps({"trial": NAME, "base_commit": sys.argv[3], "adapter_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
    "package_files_sha256": pkg, "script_sha256": {k: hashlib.sha256(cm["data"][k].encode()).hexdigest() for k in ["serve_dependency_trial.py", "moleg_program_overlay.py", "program_harness_attachment.py", "run_followup_trial.py"]},
    "nodes_sha256": NODES_SHA, "notes": ["dependency-scheduling trial: barrier (original join) vs ready (start ontology read when its predecessor finishes) at concurrency 4",
    "scripts from commit 290fd7c; package from base_commit; same frozen image, models, prompts, engines and questions as stage 1"]}, indent=2)
template = next(i for i in old if i["metadata"]["name"].endswith("-original")); client = next(i for i in old if i["metadata"]["name"].endswith("-client"))
items = [cm]
for arm in ARMS:
    pod = copy.deepcopy(template); pod["metadata"]["name"] = f"{NAME}-{arm}"; pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": arm})
    c = pod["spec"]["containers"][0]
    c["args"] = ["/opt/cnu/serve_dependency_trial.py", "--app-root", "/app", "--topology", "/opt/cnu/topology.json", "--nodes-sha256", NODES_SHA,
                 "--schedule", arm, "--host", "0.0.0.0", "--port", "28000"]
    for v in pod["spec"]["volumes"]:
        if "configMap" in v: v["configMap"]["name"] = NAME
    items.append(pod)
pod = copy.deepcopy(client); pod["metadata"]["name"] = f"{NAME}-client"; pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": "client"})
for v in pod["spec"]["volumes"]:
    if "configMap" in v: v["configMap"]["name"] = NAME
    if "hostPath" in v: v["hostPath"]["path"] = f"/home/axops/{NAME}-results"; v["hostPath"]["type"] = "DirectoryOrCreate"
items.append(pod)
Path(sys.argv[2]).write_text(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2, ensure_ascii=False) + "\n")
print("wrote", sys.argv[2], [i["metadata"]["name"] for i in items if i["kind"] == "Pod"], "configmap keys:", sorted(cm["data"]), "bytes:", Path(sys.argv[2]).stat().st_size)
