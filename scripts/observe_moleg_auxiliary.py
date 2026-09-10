"""Read-only supplementary metrics for comparison/embedding/guardrail servers."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from moleg_scaling_metrics import parse_metrics


async def run(args):
    import httpx
    servers = dict(v.split("=", 1) for v in args.metrics)
    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
        async def one(name, url):
            try:
                r = await client.get(url)
                r.raise_for_status()
                return name, parse_metrics(r.text)
            except Exception as exc:
                return name, {"error_type": type(exc).__name__}
        with args.output.open("x", buffering=1) as sink:
            while True:
                try:
                    os.kill(args.monitor_pid, 0)
                except ProcessLookupError:
                    break
                row = {"utc": datetime.now(timezone.utc).isoformat(),
                       "scope": "supplementary whole-instance counters; coverage starts after main trial began",
                       "servers": dict(await asyncio.gather(*(one(k, v) for k, v in servers.items())))}
                sink.write(json.dumps(row) + "\n")
                await asyncio.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
