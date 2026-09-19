"""Extended validation manifest: original + direct + overlap app pods and a client, from the stage-1 template."""
import base64, copy, hashlib, json, sys
from pathlib import Path
NAME = "cnu-ext-20260919"; ARMS = ["original", "direct", "overlap"]
src = json.loads(Path(sys.argv[1]).read_text()); here = Path(__file__).resolve().parent
old = src["items"]; cm = copy.deepcopy(next(i for i in old if i["kind"] == "ConfigMap"))
cm["metadata"]["name"] = NAME; cm["metadata"]["labels"]["cnu-trial"] = NAME
cm["data"] = {}
for k in ["serve_ext.py", "review_dispatch_attachment.py", "moleg_call_overlap.py", "run_followup_trial.py",
          "run_rag_experiment.py", "run_scaling_experiment.py", "run_capability_scale.py", "collect_vllm_metrics.py",
          "questions.jsonl", "smoke20.jsonl", "topology.json"]:
    cm["data"][k] = (here / k).read_text()
zip_bytes = (here / "adapter.zip").read_bytes(); cm["binaryData"] = {"adapter.zip": base64.b64encode(zip_bytes).decode()}
pkg = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((here / "pkg" / "cnu_rag_optimization").glob("*.py"))}
cm["data"]["review-inventory.json"] = json.dumps({"trial": NAME, "base_commit": sys.argv[3], "adapter_zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
    "package_files_sha256": pkg, "script_sha256": {k: hashlib.sha256(cm["data"][k].encode()).hexdigest() for k in ["serve_ext.py", "review_dispatch_attachment.py", "moleg_call_overlap.py", "run_followup_trial.py"]},
    "notes": ["extended validation: original vs direct (checked dispatch) vs overlap (direct + verified overlap of the preparation call) at concurrency 4, four position-rotated rounds",
              "same frozen image, models, prompts, engines, questions and application capacity as stage 1 and the dependency trial"]}, indent=2)
template = next(i for i in old if i["metadata"]["name"].endswith("-original")); client = next(i for i in old if i["metadata"]["name"].endswith("-client"))
items = [cm]
for arm in ARMS:
    pod = copy.deepcopy(template); pod["metadata"]["name"] = f"{NAME}-{arm}"; pod["metadata"]["labels"].update({"cnu-trial": NAME, "cnu-mode": arm})
    c = pod["spec"]["containers"][0]; c["args"] = ["/opt/cnu/serve_ext.py", "--mode", arm]
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
