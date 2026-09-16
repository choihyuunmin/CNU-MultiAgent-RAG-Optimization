"""Read-only runtime/source and connectivity probe; never starts/stops a model."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
from urllib.parse import urlsplit


def check_tcp(host, port):
    try:
        with socket.create_connection((host, port), timeout=3):
            return {"host": host, "port": port, "reachable": True}
    except OSError as exc:
        return {"host": host, "port": port, "reachable": False, "error_type": type(exc).__name__}


def inspect_runtime():
    result = {"version": None, "files": {}, "dynamic_field_present": False,
              "coordinated_dynamic_k_verified": False}
    try:
        dist = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        result["status"] = "vllm_not_installed_here"
        return result
    result["version"] = dist.version
    for name in ("vllm/config/speculative.py", "vllm/v1/spec_decode/ngram_proposer.py",
                 "vllm/v1/core/sched/scheduler.py", "vllm/v1/worker/gpu_model_runner.py"):
        path = Path(dist.locate_file(name))
        if path.is_file():
            content = path.read_bytes()
            result["files"][name] = hashlib.sha256(content).hexdigest()
            if name.endswith("config/speculative.py"):
                result["dynamic_field_present"] = b"num_speculative_tokens_per_batch_size" in content
    result["status"] = "source_inspected_not_model_compatibility_verified"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-host", action="append", default=[])
    p.add_argument("--model-origin", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    addresses = [(h, 22) for h in args.ssh_host]
    for value in args.model_origin:
        u = urlsplit(value)
        if (u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password
                or u.path not in {"", "/"} or u.query or u.fragment):
            raise ValueError("model-origin must be an HTTP origin without credentials or path")
        addresses.append((u.hostname, u.port or (443 if u.scheme == "https" else 80)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        connectivity = list(pool.map(lambda pair: check_tcp(*pair), addresses))
    gpu = {"nvidia_smi_available": bool(shutil.which("nvidia-smi"))}
    if gpu["nvidia_smi_available"]:
        try:
            process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            gpu["query_ok"] = process.returncode == 0
            if process.returncode == 0:
                gpu["devices"] = process.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu["error_type"] = type(exc).__name__
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "probe_scope": "runtime/GPU are local to this process; remote probes are TCP only",
              "runtime": inspect_runtime(),
              "gpu": gpu, "connectivity": connectivity,
              "new_gpu_experiment_requests": 0, "promote_to_full_experiment": False}
    result["status"] = "connection_blocked" if any(not r["reachable"] for r in connectivity) else "probe_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
