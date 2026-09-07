#!/usr/bin/env python3
"""Read GPU telemetry over SSH; flush every sample and never store credentials."""

import argparse
import json
import shlex
import subprocess

REMOTE = '''import csv,datetime,io,json,subprocess,time
while True:
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw","--format=csv,noheader,nounits"],text=True)
 for row in csv.reader(io.StringIO(raw)):
  fields=("index","name","memory_used_mib","memory_total_mib","gpu_utilization_percent","memory_utilization_percent","power_draw_w")
  print(json.dumps({"timestamp":stamp,**dict(zip(fields,[x.strip() for x in row]))}),flush=True)
 time.sleep(1)
'''


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH user@host; password entered interactively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    if args.host.startswith("-") or "\n" in args.host or args.stop_file.exists():
        parser.error("valid SSH host and absent stop file required")
    command = ["ssh", "-T", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
               "-o", "PreferredAuthentications=password,keyboard-interactive", "-o", "PubkeyAuthentication=no",
               args.host, "python3 -u -c " + shlex.quote(REMOTE)]
    count = 0
    with args.output.open("x") as target:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        try:
            for line in process.stdout:
                event = json.loads(line)
                target.write(json.dumps(event) + "\n")
                target.flush()
                count += 1
                if args.stop_file.exists():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
    if not count:
        raise RuntimeError("no GPU samples captured")
    print(f"GPU samples saved: {count}; {args.output}")


if __name__ == "__main__":
    main()
