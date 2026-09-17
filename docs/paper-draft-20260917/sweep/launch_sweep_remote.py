"""Load-sweep supervisor (2026-09-17): smoke, main, log retention, collection, analysis.

Run on the application host with cluster access (sudo). Touches only the named
experiment Pods; never changes production or model processes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

NAME = "cnu-sweep-20260917"
ARMS = ["original", "direct", "capability"]
KUBE = ["kubectl", "--kubeconfig=/etc/kubernetes/admin.conf", "-n", "moleg"]


def command(*args):
    return subprocess.check_output(KUBE+list(args), text=True)


def main(folder, levels, rounds, arms, timeout):
    state_path = folder/"supervisor-status.json"
    def state(value):
        state_path.write_text(json.dumps(value, indent=2)+"\n")
    data = json.loads(command("get", "pods", "-l", f"cnu-trial={NAME}", "-o", "json"))
    pods = {p["metadata"]["name"]: p for p in data["items"]}
    expected = {NAME+"-"+x for x in ARMS+["client"]}
    if set(pods) != expected or any(p["status"]["phase"] != "Running" for p in pods.values()):
        raise RuntimeError(f"unexpected experiment pods: {sorted(pods)}")
    client = NAME+"-client"
    base = ["exec", client, "--", "/app/.venv/bin/python", "/opt/cnu/run_sweep_trial.py",
            "--topology", "/opt/cnu/topology.json", "--arms", arms, "--timeout", str(timeout)]
    endpoints = []
    for arm in ARMS:
        ip = pods[NAME+"-"+arm]["status"]["podIP"]
        endpoints += ["--endpoint", f"{arm}=http://{ip}:28000"]
    state({"state": "smoke"})
    rc = subprocess.call(KUBE+base+["--questions", "/opt/cnu/smoke.jsonl", "--output", "/results/smoke", "--smoke"]+endpoints)
    smoke = json.loads(command("exec", client, "--", "cat", "/results/smoke/status.json"))
    if rc != 0 or smoke["state"] != "complete" or len(smoke["completed"]) != len(arms.split(",")):
        raise RuntimeError("smoke not completed")
    if any(x["transport_ok"] != 1 for x in smoke["completed"]):
        raise RuntimeError("smoke failure")
    args = base+["--questions", "/opt/cnu/questions.jsonl", "--output", "/results/main",
                 "--levels", levels, "--rounds", str(rounds)]+endpoints
    handles = []; followers = []
    try:
        for arm in ARMS:
            log = (folder/f"{arm}-application.log").open("xb")
            handles.append(log)
            followers.append(subprocess.Popen(KUBE+["logs", "--follow", "--timestamps", NAME+"-"+arm],
                                              stdout=log, stderr=subprocess.STDOUT))
        state({"state": "running", "levels": levels, "rounds": rounds, "arms": arms})
        rc = subprocess.call(KUBE+args)
        state({"state": "collecting", "trial_exit_code": rc})
    finally:
        for child in followers:
            child.terminate()
        for child in followers:
            child.wait(timeout=15)
        for log in handles:
            log.close()
    subprocess.run(KUBE+["cp", f"{client}:/results/main", str(folder/"main")], check=True)
    subprocess.run([sys.executable, str(folder/"analyze_sweep.py"), "--root", str(folder/"main"),
                    "--logs", str(folder), "--output", str(folder/"sweep-summary.json")], check=False)
    state({"state": "complete" if rc == 0 else "stopped", "trial_exit_code": rc,
           "summary": str(folder/"sweep-summary.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--levels", default="8,16,32,64,100")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--arms", default="original,capability")
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()
    main(a.folder, a.levels, a.rounds, a.arms, a.timeout)
