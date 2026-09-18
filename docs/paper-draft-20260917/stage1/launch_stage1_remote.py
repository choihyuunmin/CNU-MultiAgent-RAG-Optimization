"""Stage-1 supervisor: 20-question smoke (both arms), then 200 x {original, direct} + original repeat at concurrency 4.
Runs on the application host with cluster access (sudo). Touches only the named experiment Pods."""
import argparse, json, subprocess, sys
from pathlib import Path
NAME = "cnu-stage1-20260918"; ARMS = ["original", "direct"]
KUBE = ["kubectl", "--kubeconfig=/etc/kubernetes/admin.conf", "-n", "moleg"]
def command(*args): return subprocess.check_output(KUBE + list(args), text=True)
def main(folder, timeout, idle_timeout, idle_max_running):
    state_path = folder / "supervisor-status.json"
    def state(v): state_path.write_text(json.dumps(v, indent=2) + "\n")
    pods = {p["metadata"]["name"]: p for p in json.loads(command("get", "pods", "-l", f"cnu-trial={NAME}", "-o", "json"))["items"]}
    expected = {NAME + "-" + x for x in ARMS + ["client"]}
    if set(pods) != expected or any(p["status"]["phase"] != "Running" for p in pods.values()):
        raise RuntimeError(f"unexpected experiment pods: {sorted(pods)}")
    client = NAME + "-client"
    endpoints = []
    for arm in ARMS:
        endpoints += ["--endpoint", f"{arm}=http://{pods[NAME + '-' + arm]['status']['podIP']}:28000"]
    base = ["exec", client, "--", "/app/.venv/bin/python", "/opt/cnu/run_followup_trial.py", "--topology", "/opt/cnu/topology.json",
            "--concurrency", "4", "--timeout", str(timeout), "--idle-timeout", str(idle_timeout), "--idle-max-running", str(idle_max_running)]
    handles, followers = [], []
    try:
        for arm in ARMS:
            log = (folder / f"{arm}-application.log").open("xb"); handles.append(log)
            followers.append(subprocess.Popen(KUBE + ["logs", "--follow", "--timestamps", NAME + "-" + arm], stdout=log, stderr=subprocess.STDOUT))
        state({"state": "smoke"})
        rc = subprocess.call(KUBE + base + ["--questions", "/opt/cnu/smoke20.jsonl", "--output", "/results/smoke", "--order", ",".join(ARMS), "--smoke", "--smoke-count", "20"] + endpoints)
        smoke = json.loads(command("exec", client, "--", "cat", "/results/smoke/status.json"))
        if rc != 0 or smoke["state"] != "complete" or len(smoke["completed"]) != len(ARMS) or any(x["transport_ok"] != 20 for x in smoke["completed"]):
            raise RuntimeError(f"smoke not completed: {smoke.get('completed')}")
        state({"state": "running", "plan": "round1: original,direct; round2: original (repeat)"})
        rc = subprocess.call(KUBE + base + ["--questions", "/opt/cnu/questions.jsonl", "--output", "/results/main", "--order", ",".join(ARMS), "--order", "original"] + endpoints)
        state({"state": "collecting", "trial_exit_code": rc})
    finally:
        for child in followers: child.terminate()
        for child in followers: child.wait(timeout=15)
        for log in handles: log.close()
    subprocess.run(KUBE + ["cp", f"{client}:/results/smoke", str(folder / "smoke")], check=False)
    subprocess.run(KUBE + ["cp", f"{client}:/results/main", str(folder / "main")], check=True)
    state({"state": "complete" if rc == 0 else "stopped", "trial_exit_code": rc})
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=600); p.add_argument("--idle-timeout", type=int, default=300); p.add_argument("--idle-max-running", type=int, default=0)
    a = p.parse_args(); main(a.folder, a.timeout, a.idle_timeout, a.idle_max_running)
