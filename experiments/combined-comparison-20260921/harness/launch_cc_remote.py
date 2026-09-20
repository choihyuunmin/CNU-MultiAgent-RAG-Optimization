import argparse
import json
import os
import subprocess
from pathlib import Path

NAME = "cnu-cc-20260921"
ARMS = ["original", "direct", "reduce", "combined"]
NAMESPACE = os.environ["CNU_K8S_NAMESPACE"]
KUBE = ["kubectl", "--kubeconfig=/etc/kubernetes/admin.conf", "-n", NAMESPACE]


def command(*args):
    return subprocess.check_output(KUBE + list(args), text=True)


def main(folder, levels, rounds, timeout, idle_timeout, idle_max_running, suffix="", arms=None):
    global ARMS
    if arms:
        ARMS = [a.strip() for a in arms.split(",")]
    state_path = folder / f"supervisor-status{suffix}.json"

    def state(v):
        state_path.write_text(json.dumps(v, indent=2) + "\n")

    pods = {p["metadata"]["name"]: p for p in json.loads(command("get", "pods", "-l", f"cnu-trial={NAME}", "-o", "json"))["items"]}
    expected = {NAME + "-" + x for x in ARMS + ["client"]}
    if not expected <= set(pods) or any(pods[n]["status"]["phase"] != "Running" for n in expected):
        raise RuntimeError(f"unexpected experiment pods: {sorted(pods)}")
    client = NAME + "-client"
    endpoints = []
    for arm in ARMS:
        endpoints += ["--endpoint", f"{arm}=http://{pods[NAME + '-' + arm]['status']['podIP']}:28000"]
    base = ["exec", client, "--", "/app/.venv/bin/python", "/opt/cnu/run_sweep_trial.py", "--topology", "/opt/cnu/topology.json",
            "--arms", ",".join(ARMS), "--timeout", str(timeout), "--idle-timeout", str(idle_timeout),
            "--idle-max-running", str(idle_max_running)]
    handles, followers = [], []
    try:
        for arm in ARMS:
            log = (folder / f"{arm}-application{suffix}.log").open("ab")
            handles.append(log)
            followers.append(subprocess.Popen(KUBE + ["logs", "--follow", "--timestamps", NAME + "-" + arm], stdout=log, stderr=subprocess.STDOUT))
        state({"state": "smoke"})
        rc = subprocess.call(KUBE + base + ["--questions", "/opt/cnu/smoke.jsonl", "--output", f"/results/smoke{suffix}", "--smoke"] + endpoints)
        smoke = json.loads(command("exec", client, "--", "cat", f"/results/smoke{suffix}/status.json"))
        if rc != 0 or smoke["state"] != "complete" or len(smoke["completed"]) != len(ARMS) or any(x["transport_ok"] != 1 for x in smoke["completed"]):
            raise RuntimeError(f"smoke not completed: {smoke.get('completed')}")
        state({"state": "running", "levels": levels, "rounds": rounds})
        rc = subprocess.call(KUBE + base + ["--questions", "/opt/cnu/questions.jsonl", "--output", f"/results/main{suffix}", "--levels", levels, "--rounds", str(rounds)] + endpoints)
        state({"state": "collecting", "trial_exit_code": rc})
    finally:
        for child in followers:
            child.terminate()
        for child in followers:
            child.wait(timeout=15)
        for log in handles:
            log.close()
    subprocess.run(KUBE + ["cp", f"{client}:/results/smoke{suffix}", str(folder / f"smoke{suffix}")], check=False)
    subprocess.run(KUBE + ["cp", f"{client}:/results/main{suffix}", str(folder / f"main{suffix}")], check=True)
    state({"state": "complete" if rc == 0 else "stopped", "trial_exit_code": rc})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--levels", default="20,50,100")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--idle-timeout", type=int, default=600)
    p.add_argument("--idle-max-running", type=int, default=1)
    p.add_argument("--suffix", default="")
    p.add_argument("--arms", default=None)
    a = p.parse_args()
    main(a.folder, a.levels, a.rounds, a.timeout, a.idle_timeout, a.idle_max_running, a.suffix, a.arms)
