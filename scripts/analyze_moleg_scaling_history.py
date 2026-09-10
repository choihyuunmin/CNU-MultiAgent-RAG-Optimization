"""Re-audit E1 raw traces and time-aligned server logs; emit only aggregates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import statistics

from moleg_scaling_metrics import describe


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def intervals(rows):
    events = []
    for row in rows:
        start = ts(row["utc"])
        events.extend([(start, 1), (start + row["elapsed_s"], -1)])
    running = peak = 0
    area = active_time = 0.
    previous = min(x[0] for x in events)
    for when, change in sorted(events):
        duration = when - previous
        area += duration * running
        active_time += duration if running else 0
        running += change
        peak = max(peak, running)
        previous = when
    return {"peak_outstanding": peak,
            "mean_outstanding_while_arm_active": area / active_time,
            "active_time_s": active_time,
            "burst_active_throughput_rps": len(rows) / active_time,
            "warning": "excludes gaps while other arm executes; not sustained throughput"}


def server_samples(path):
    records = []
    pattern = re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?Avg prompt throughput: ([\d.]+) tokens/s, '
                         r'Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, '
                         r'Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%')
    for line in path.open():
        match = pattern.search(line)
        if match:
            date = datetime.strptime("2026-" + match[1], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone(timedelta(hours=9)))
            records.append({"ts": date.timestamp(), "prompt_tps": float(match[2]),
                            "generation_tps": float(match[3]), "running": int(match[4]),
                            "waiting": int(match[5]), "kv_percent": float(match[6])})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    traces = {arm: {r["session_id"]: r for r in read_rows(args.private / "results" / f"e1_{arm}.trace.jsonl")}
              for arm in ("w1", "w2")}
    samples = server_samples(args.private / "logs/custom_prod_gemma_spec.log")
    out = {"scope": "reanalysis of 2026-09-06/07 E1; no new GPU execution",
           "limitations": ["call position is not a semantic stage label",
                            "call duration includes queue, prefill and decode",
                            "10-second point samples can miss transient saturation",
                            "no per-request prefill or hardware bandwidth metrics in E1"], "loads": {}}
    for concurrency in (4, 8, 16):
        path = args.private / "results" / f"e1_c{concurrency}.jsonl"
        rows = read_rows(path)
        by_arm = {}
        start = min(ts(r["utc"]) for r in rows)
        end = max(ts(r["utc"]) + r["elapsed_s"] for r in rows)
        sampled = [s for s in samples if start <= s["ts"] <= end]
        for arm in ("w1", "w2"):
            selected = [r for r in rows if r["variant"] == arm]
            positions = defaultdict(list)
            totals = []
            for row in selected:
                trace = traces[arm][row["session_id"]]
                calls = [c for c in trace.get("llm", []) if c.get("model") == "orchestrator" and not c.get("stream")]
                totals.append(sum(c["elapsed_s"] for c in calls))
                for i, call in enumerate(calls):
                    positions[i + 1].append(call)
            by_arm[arm] = {"n": len(selected), "latency_s": describe([r["elapsed_s"] for r in selected]),
                "ttft_s": describe([r.get("ttft_s") for r in selected]),
                "actual_concurrency": intervals(selected), "orchestrator_sum_per_request_s": describe(totals),
                "call_positions": {str(k): {"latency_s": describe([c["elapsed_s"] for c in calls]),
                    "prompt_tokens": describe([c.get("usage", {}).get("prompt_tokens") for c in calls]),
                    "completion_tokens": describe([c.get("usage", {}).get("completion_tokens") for c in calls])}
                    for k, calls in positions.items()}}
        out["loads"][str(concurrency)] = {"arms": by_arm,
            "raw_rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "server_point_samples_both_arms": {k: describe([s[k] for s in sampled])
                                               for k in ("running", "waiting", "kv_percent", "prompt_tps", "generation_tps")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for c, data in out["loads"].items():
        w = data["arms"]["w1"]
        print(c, "latency", round(w["latency_s"]["mean"], 3), "actual mean users",
              round(w["actual_concurrency"]["mean_outstanding_while_arm_active"], 2),
              "KV peak%", data["server_point_samples_both_arms"]["kv_percent"]["max"],
              "waiting peak", data["server_point_samples_both_arms"]["waiting"]["max"])


if __name__ == "__main__":
    main()
