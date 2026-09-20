import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import httpx
from run_scaling_experiment import collect, wait_for_idle, write


class HighLoadSafety:
    """Experiment-only abort checks; KV occupancy is observation, not failure."""

    def __init__(self):
        self.since = {}

    def observe(self, resource, metrics, now):
        missing = metrics is None or any(k not in metrics for k in (
            "vllm:num_requests_running", "vllm:num_requests_waiting"))
        conditions = {"telemetry_unavailable": missing,
                      "queue_above_256": not missing and metrics["vllm:num_requests_waiting"] > 256}
        for reason, active in conditions.items():
            key = (resource, reason)
            if active:
                self.since.setdefault(key, now)
                if now - self.since[key] >= 30:
                    return f"{resource}: {reason} for 30 seconds"
            else:
                self.since.pop(key, None)
        return None


def response_failure_reason(records):
    if len(records) >= 20 and sum(r.get("status") != "ok" for r in records[-20:]) >= 4:
        return "at least 4 transport failures in the last 20 completed requests"
    return None


async def collect_high_load(topology, parse, path, stop, unsafe, ready):
    safety = HighLoadSafety()
    with path.open("w", encoding="utf-8") as stream:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) as client:
            while True:
                final = stop.is_set()
                for resource, config in topology["receivers"].items():
                    row = {"time": time.time(), "resource": resource}
                    values = None
                    try:
                        response = await client.get(config["metrics"])
                        response.raise_for_status()
                        values = defaultdict(float)
                        for metric in parse(response.text):
                            if not metric["metric"].endswith("_bucket"):
                                values[metric["metric"]] += metric["value"]
                        row["metrics"] = dict(values)
                    except Exception as exc:
                        row["error_type"] = type(exc).__name__
                    reason = safety.observe(resource, values, time.monotonic())
                    if reason:
                        row["abort_reason"] = reason
                        unsafe.set()
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                ready.set()
                if final:
                    break
                try:
                    await asyncio.wait_for(stop.wait(), 1)
                except asyncio.TimeoutError:
                    pass


def levels(value):
    result = [int(x) for x in value.split(",")]
    if not result or result != sorted(set(result)) or not 1 <= min(result) <= max(result) <= 100:
        raise ValueError("unique ascending concurrency levels between 1 and 100 required")
    return result


async def main(args):
    if args.diagnostic_background and not args.high_load_safety:
        raise ValueError("background diagnostics require high-load safety checks")
    from collect_vllm_metrics import parse_prometheus
    sys.path.insert(0, "/app/src")
    from config import settings
    key_attr = os.environ.get("CNU_APP_KEY_ATTR", "APP_API_KEY")
    os.environ[key_attr] = getattr(settings, key_attr)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    questions = Path(args.questions)
    rows = [json.loads(x) for x in questions.read_text().splitlines() if x.strip()]
    if len(rows) != 200 or len({x["id"] for x in rows}) != 200:
        raise ValueError("exactly 200 unique evaluation queries required")
    topology = json.loads(Path(args.topology).read_text())
    topology["receivers"] = {k: v for k, v in topology["receivers"].items() if "metrics" in v}
    endpoints = dict(x.split("=", 1) for x in args.endpoint)
    if set(endpoints) != {"original", "capability"}:
        raise ValueError("original and capability endpoints required")
    protocol = {"count_per_cell": 200, "concurrency": args.concurrency,
        "total_planned_requests": 400 * len(args.concurrency), "repeat": 1,
        "question_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
        "endpoints": endpoints, "timeout_seconds": 600, "seed": 20260912,
        "application_capacity": 100, "engine_and_model_unchanged": True,
        "design": "closed-loop finite 200-query batches; ascending load; alternating method order",
        "quality": "reference agreement, not expert accuracy; no acceptance threshold",
        "limitations": "one pass, reused questions, shared engines, no causal significance claim",
        "diagnostic_background": args.diagnostic_background,
        "performance_comparison_allowed": not args.diagnostic_background,
        "safety": ("high-load-v2: KV occupancy warning only; stop on per-engine telemetry loss or queue >256 for 30s; "
                   "no completed response for 180s; >=4 transport failures in last 20; cell wall time >3600s"
                   if args.high_load_safety else
                   "stop test on persistent >64 queued model requests or >95% KV usage; missing telemetry stops test")}
    write(root / "protocol.json", protocol)
    status = {"state": "running", "completed": [], "started": time.time()}
    write(root / "status.json", status)
    try:
        for index, concurrency in enumerate(args.concurrency):
            methods = ("original", "capability") if index % 2 == 0 else ("capability", "original")
            for mode in methods:
                cell = root / f"c{concurrency:03d}"
                cell.mkdir(exist_ok=True)
                status.update(current={"concurrency": concurrency, "mode": mode})
                write(root / "status.json", status)
                if not args.diagnostic_background:
                    await wait_for_idle(topology, parse_prometheus, 180)
                async with httpx.AsyncClient(timeout=10) as http:
                    (await http.get(endpoints[mode] + "/health/live")).raise_for_status()
                stop, unsafe, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
                metric_path = cell / f"{mode}-metrics.jsonl"
                collect_fn = collect_high_load if args.high_load_safety else collect
                collector = asyncio.create_task(collect_fn(topology, parse_prometheus, metric_path, stop, unsafe, ready))
                child = None
                try:
                    await asyncio.wait_for(ready.wait(), 30)
                    samples = [json.loads(x) for x in metric_path.read_text().splitlines()]
                    if any("error_type" in x for x in samples):
                        raise RuntimeError("initial engine telemetry unavailable")
                    command = [sys.executable, "/opt/cnu/run_rag_experiment.py",
                        "--base-url", endpoints[mode], "--endpoint", "/api/generate/stream",
                        "--questions", str(questions), "--output-dir", str(cell / "responses"),
                        "--run-id", "capability-trial", "--variant", mode,
                        "--concurrency", str(concurrency), "--repeats", "1", "--warmup", "0",
                        "--timeout", "600", "--store-responses", "--seed", "20260912"]
                    with (cell / f"{mode}-client.log").open("wb") as log:
                        child = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
                        cell_started = last_progress = time.monotonic()
                        last_count = 0
                        print(f"START c={concurrency} mode={mode}", flush=True)
                        while child.returncode is None:
                            if unsafe.is_set():
                                raise RuntimeError("engine safety threshold reached; see metrics abort_reason")
                            try:
                                await asyncio.wait_for(asyncio.shield(child.wait()), 15)
                            except asyncio.TimeoutError:
                                checkpoint = cell / f"responses/capability-trial/{mode}/client_requests.checkpoint.jsonl"
                                records = [json.loads(x) for x in checkpoint.read_text().splitlines() if x.strip()] if checkpoint.exists() else []
                                count = len(records)
                                if count > last_count:
                                    last_count, last_progress = count, time.monotonic()
                                if args.high_load_safety:
                                    reason = response_failure_reason(records)
                                    if reason:
                                        raise RuntimeError(reason)
                                    if time.monotonic() - last_progress > 180:
                                        raise RuntimeError("no completed response for 180 seconds")
                                    if time.monotonic() - cell_started > 3600:
                                        raise RuntimeError("cell wall time exceeded 3600 seconds")
                                status["current"]["completed_requests"] = count
                                write(root / "status.json", status)
                                print(f"PROGRESS c={concurrency} mode={mode} {count}/200", flush=True)
                                tail = [json.loads(x) for x in metric_path.read_text().splitlines()[-len(topology["receivers"])*3:]]
                                if tail and all("error_type" in x for x in tail):
                                    raise RuntimeError("engine telemetry lost")
                    output = cell / f"responses/capability-trial/{mode}/client_requests.jsonl"
                    records = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
                    if child.returncode not in (0, 2) or len(records) != 200 or len({x["question_id"] for x in records}) != 200:
                        raise RuntimeError("incomplete evaluation cell")
                    status["completed"].append({"concurrency": concurrency, "mode": mode,
                        "requests": 200, "transport_ok": sum(x["status"] == "ok" for x in records)})
                    write(root / "status.json", status)
                    print(f"DONE c={concurrency} mode={mode}", flush=True)
                finally:
                    if child is not None and child.returncode is None:
                        child.terminate()
                        await child.wait()
                    stop.set()
                    await collector
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="stopped", reason=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        status["updated"] = time.time()
        write(root / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--endpoint", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--concurrency", type=levels, default=levels("1,2,4,8,16,32,64,100"))
    p.add_argument("--high-load-safety", action="store_true", help="Explicitly enable revised experimental abort policy")
    p.add_argument("--diagnostic-background", action="store_true",
                   help="Observation only with existing traffic; timing is not a performance comparison")
    asyncio.run(main(p.parse_args()))
