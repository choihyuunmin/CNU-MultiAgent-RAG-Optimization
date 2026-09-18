"""Stage-1 manifest: original + direct + client, new public package bundle, new attachment/runner."""
import base64, copy, hashlib, json, sys
from pathlib import Path
NAME="cnu-stage1-20260918"; ARMS={"original":"original","direct":"direct"}
src=json.loads(Path(sys.argv[1]).read_text()); here=Path(__file__).resolve().parent
old=src["items"]; cm=copy.deepcopy(next(i for i in old if i["kind"]=="ConfigMap"))
cm["metadata"]["name"]=NAME; cm["metadata"]["labels"]["cnu-trial"]=NAME
for k in ["review_dispatch_attachment.py","run_followup_trial.py"]: cm["data"][k]=(here/k).read_text()
cm["data"]["serve_stage1.py"]=(here/"serve_stage1.py").read_text()
cm["data"]["smoke20.jsonl"]=(here/"smoke20.jsonl").read_text()
for k in ["serve_trial.py","serve_embedded_adapter.py","serve_scaling_adapter.py","run_capability_trial.py","run_review_trial.py","run_sweep_trial.py"]:
    cm["data"].pop(k,None)
zip_bytes=(here/"adapter.zip").read_bytes(); cm["binaryData"]={"adapter.zip":base64.b64encode(zip_bytes).decode()}
pkg={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((here/"pkg"/"cnu_rag_optimization").glob("*.py"))}
cm["data"]["review-inventory.json"]=json.dumps({"trial":NAME,"base_commit":sys.argv[3],"adapter_zip_sha256":hashlib.sha256(zip_bytes).hexdigest(),"package_files_sha256":pkg,
    "attachment_sha256":hashlib.sha256(cm["data"]["review_dispatch_attachment.py"].encode()).hexdigest(),"runner_sha256":hashlib.sha256(cm["data"]["run_followup_trial.py"].encode()).hexdigest(),
    "notes":["stage 1 of DISPATCH_VALIDATION_PROTOCOL_20260918: original vs unsigned direct at concurrency 4","capability (signed) arm not included","adapter.zip rebuilt from the public package at base_commit"]},indent=2)
template=next(i for i in old if i["metadata"]["name"].endswith("-original")); client=next(i for i in old if i["metadata"]["name"].endswith("-client"))
items=[cm]
for arm,mode in ARMS.items():
    pod=copy.deepcopy(template); pod["metadata"]["name"]=f"{NAME}-{arm}"; pod["metadata"]["labels"].update({"cnu-trial":NAME,"cnu-mode":mode})
    c=pod["spec"]["containers"][0]; c["args"]=["/opt/cnu/serve_stage1.py","--mode",mode]
    for v in pod["spec"]["volumes"]:
        if "configMap" in v: v["configMap"]["name"]=NAME
    items.append(pod)
pod=copy.deepcopy(client); pod["metadata"]["name"]=f"{NAME}-client"; pod["metadata"]["labels"].update({"cnu-trial":NAME,"cnu-mode":"client"})
for v in pod["spec"]["volumes"]:
    if "configMap" in v: v["configMap"]["name"]=NAME
    if "hostPath" in v: v["hostPath"]["path"]=f"/home/axops/{NAME}-results"; v["hostPath"]["type"]="DirectoryOrCreate"
items.append(pod)
Path(sys.argv[2]).write_text(json.dumps({"apiVersion":"v1","kind":"List","items":items},indent=2,ensure_ascii=False)+"\n")
print("wrote",sys.argv[2],[i["metadata"]["name"] for i in items if i["kind"]=="Pod"], "configmap keys:", sorted(cm["data"]))
