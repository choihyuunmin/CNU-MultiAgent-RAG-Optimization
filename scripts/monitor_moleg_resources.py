"""Run on the inference host to collect GPU and host-memory observations.

No process command lines, environment values or prompts are captured. NVIDIA
memory utilization is memory-active time, NOT fraction of peak HBM bandwidth.
Use DCGM/Nsight on an isolated server for a hardware bandwidth conclusion.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time


def proc_values(path, allowed):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.replace(":", "").split()
        if parts and parts[0] in allowed:
            result[parts[0]] = int(parts[1])
    return result


def sample():
    row = {"utc": datetime.now(timezone.utc).isoformat(),
           "host_memory_kib": proc_values(Path("/proc/meminfo"), {
               "MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Dirty", "Writeback"}),
           "vm_counters": proc_values(Path("/proc/vmstat"), {
               "pswpin", "pswpout", "pgmajfault", "pgfault", "oom_kill"})}
    pressure = Path("/proc/pressure/memory")
    if pressure.exists():
        row["memory_psi"] = pressure.read_text().strip().splitlines()
    names = ["index", "name", "memory.total", "memory.used", "utilization.gpu",
             "utilization.memory", "power.draw", "temperature.gpu"]
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(names),
                                 "--format=csv,noheader,nounits"], check=True,
                                capture_output=True, text=True, timeout=5)
        row["gpus"] = [dict(zip(names, (v.strip() for v in values)))
                       for values in csv.reader(io.StringIO(result.stdout))]
    except (OSError, subprocess.SubprocessError) as exc:
        row["gpu_error_type"] = type(exc).__name__
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")
    end = time.monotonic() + args.seconds
    with args.output.open("x", buffering=1) as sink:
        while time.monotonic() < end:
            start = time.monotonic()
            sink.write(json.dumps(sample()) + "\n")
            time.sleep(max(0, min(args.interval - (time.monotonic() - start), end - time.monotonic())))


if __name__ == "__main__":
    main()
