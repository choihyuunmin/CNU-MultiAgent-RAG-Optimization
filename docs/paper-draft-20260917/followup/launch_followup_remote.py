"""Follow-up supervisor (2026-09-17): smoke, main, log retention, collection, analysis.

Run on the application host with cluster access (sudo). Touches only the named
experiment Pods; never changes production or model processes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

NAME = "cnu-review-followup-20260917"
ARMS = ["original-a", "capability-a", "original-b", "capability-b"]
KUBE = ["kubectl", "--kubeconfig=/etc/kubernetes/admin.conf", "-n", "moleg"]


def command(*args):
    return subprocess.check_output(KUBE+list(args), text=True)


def main(folder, rounds):
    state_path = folder/"supervisor-status.json"
    def state(value):
        state_path.write_text(json.dumps(value, indent=2)+"\n")
    data = json.loads(command("get", "pods", "-l", f"cnu-trial={NAME}", "-o", "json"))
    pods = {p["metadata"]["name"]: p for p in data["items"]}
    expected = {NAME+"-"+x for x in ARMS+["client"]}
    if set(pods) != expected or any(p["status"]["phase"] != "Running" for p in pods.values()):
        raise RuntimeError(f"unexpected experiment pods: {sorted(pods)}")
    client = NAME+"-client"
    orders = [",".join(ARMS)]
    if rounds >= 2:
        orders.append(",".join(reversed(ARMS)))
    import time as _time
    tag = _time.strftime("%H%M%S")
    base = ["exec", client, "--", "/app/.venv/bin/python", "/opt/cnu/run_followup_trial.py",
            "--topology", "/opt/cnu/topology.json", "--idle-timeout", "900", "--idle-max-running", "1"]
    endpoints = []
    for arm in ARMS:
        ip = pods[NAME+"-"+arm]["status"]["podIP"]
        endpoints += ["--endpoint", f"{arm}=http://{ip}:28000"]
    # smoke: one question through every arm
    state({"state": "smoke"})
    rc = subprocess.call(KUBE+base+["--questions", "/opt/cnu/smoke.jsonl", "--output", f"/results/smoke-{tag}",
                                    "--order", orders[0], "--smoke"]+endpoints)
    smoke = json.loads(command("exec", client, "--", "cat", f"/results/smoke-{tag}/status.json"))
    if rc != 0 or smoke["state"] != "complete" or len(smoke["completed"]) != len(ARMS):
        raise RuntimeError("smoke not completed")
    if any(x["transport_ok"] != 1 for x in smoke["completed"]):
        raise RuntimeError("smoke failure")
    args = base+["--questions", "/opt/cnu/questions.jsonl", "--output", f"/results/main-{tag}"]
    for order in orders:
        args += ["--order", order]
    args += endpoints
    handles = []; followers = []
    try:
        for arm in ARMS:
            log = (folder/f"{arm}-application.log").open("xb")
            handles.append(log)
            followers.append(subprocess.Popen(KUBE+["logs", "--follow", "--timestamps", NAME+"-"+arm],
                                              stdout=log, stderr=subprocess.STDOUT))
        state({"state": "running", "total_requests": 200*len(ARMS)*len(orders)})
        rc = subprocess.call(KUBE+args)
        state({"state": "collecting", "trial_exit_code": rc})
    finally:
        for child in followers:
            child.terminate()
        for child in followers:
            child.wait(timeout=15)
        for log in handles:
            log.close()
    subprocess.run(KUBE+["cp", f"{client}:/results/main-{tag}", str(folder/"main")], check=True)
    subprocess.run([sys.executable, str(folder/"analyze_followup.py"), "--root", str(folder/"main"),
                    "--logs", str(folder), "--output", str(folder/"followup-summary.json")], check=False)
    state({"state": "complete" if rc == 0 else "stopped", "trial_exit_code": rc,
           "summary": str(folder/"followup-summary.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=1, help="1 (default) or 2 (second round in reversed order)")
    args = p.parse_args()
    main(args.folder, args.rounds)
