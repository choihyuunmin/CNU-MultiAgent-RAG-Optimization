"""Load sweep (2026-09-17): original / unsigned direct / signed dispatch at several concurrency levels.

Per level, the arms run in rotated rounds (a Latin square when rounds == number of arms).
Same client, evaluator, metrics collection and high-load safety policy as the earlier
runners. Timeout 600 s per request as in the 2026-09-13/14 sweep. Output layout:
<output>/level-<C>/round-<r>/{<arm>-metrics.jsonl,<arm>-client.log,responses/...}.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
from collections import defaultdict
from run_scaling_experiment import wait_for_idle
from run_capability_scale import collect_high_load, response_failure_reason


async def wait_for_idle_relaxed(topology, parse_prometheus, timeout, max_running):
    """Like wait_for_idle, but tolerates up to `max_running` running requests per engine
    (for a stuck request that never finishes). Waiting must still be zero. Read-only."""
    started, stable = time.monotonic(), 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) as client:
        while True:
            observed = {}
            for resource, config in topology["receivers"].items():
                response = await client.get(config["metrics"])
                response.raise_for_status()
                gauges = defaultdict(float); present = set()
                for row in parse_prometheus(response.text):
                    if row["metric"] in {"vllm:num_requests_running", "vllm:num_requests_waiting"}:
                        gauges[row["metric"]] += row["value"]; present.add(row["metric"])
                if len(present) != 2:
                    raise RuntimeError("cannot establish idle state: missing engine gauges")
                observed[resource] = dict(gauges)
            quiet = all(g["vllm:num_requests_waiting"] == 0 and g["vllm:num_requests_running"] <= max_running
                        for g in observed.values())
            stable = stable + 1 if quiet else 0
            elapsed = time.monotonic() - started
            if stable >= 2:
                return {"idle_wait_seconds": elapsed, "quiet_samples": stable, "gauges": observed,
                        "max_running_allowed": max_running}
            if elapsed >= timeout:
                raise RuntimeError("engines remain busy before test; no inference sent; arrange a quiet measurement window")
            await asyncio.sleep(1)


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def read_records(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def rotations(arms, rounds):
    return [tuple(arms[(i+k) % len(arms)] for i in range(len(arms))) for k in range(rounds)]


async def run(args):
    from collect_vllm_metrics import parse_prometheus
    sys.path.insert(0, "/app/src")
    from config import settings
    os.environ["MOLEG_API_KEY"] = settings.MOLEG_API_KEY
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    questions = Path(args.questions)
    rows = read_records(questions)
    count = 1 if args.smoke else 200
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError("wrong frozen question count")
    topology = json.loads(Path(args.topology).read_text())
    topology["receivers"] = {k: v for k, v in topology["receivers"].items() if "metrics" in v}
    endpoints = dict(v.split("=", 1) for v in args.endpoint)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if set(arms) - set(endpoints) or len(set(arms)) != len(arms):
        raise ValueError("arms must be unique and have endpoints")
    levels = [int(x) for x in args.levels.split(",")]
    if args.smoke:
        levels, rounds = [1], 1
    else:
        rounds = args.rounds
        if levels != sorted(set(levels)) or min(levels) < 1 or max(levels) > 100:
            raise ValueError("unique ascending levels between 1 and 100 required")
    schedule = rotations(arms, rounds)
    protocol = {"count_per_cell": count, "levels": levels, "arms": arms, "rounds": rounds,
                "order_per_level": schedule, "seed": 20260912, "timeout_seconds": args.timeout,
                "total_planned_requests": count*len(levels)*len(arms)*rounds,
                "question_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path("/opt/cnu").glob("*") if p.is_file()},
                "application_capacity": int(os.environ.get("CNU_APP_CAPACITY", "100")),
                "idle_gate": {"timeout_seconds": args.idle_timeout, "max_running_allowed": args.idle_max_running},
                "design": "closed-loop 200-question batches; per level, arms in rotated rounds; separate load client",
                "quality": "baseline agreement only; no expert labels or acceptance margin",
                "settings": "same frozen image, models, engine, prompts, questions; app admission raised to capacity",
                "safety": "idle preflight; telemetry loss/queue>256 for 30s; 4 errors/last20; no progress 240s; cell limit 3600s",
                "limitations": "shared engines; one instance per arm; few rounds; production shares the GPU stack"}
    save(root/"protocol.json", protocol)
    status = {"state": "running", "started": time.time(), "completed": [],
              "total_planned_requests": protocol["total_planned_requests"]}
    save(root/"status.json", status)
    try:
        for level in levels:
            for round_index, methods in enumerate(schedule, 1):
                cell = root/f"level-{level:03d}"/f"round-{round_index}"
                cell.mkdir(parents=True, exist_ok=True)
                for mode in methods:
                    status["current"] = {"level": level, "round": round_index, "method": mode,
                                         "completed_requests": 0, "phase": "idle_preflight"}
                    save(root/"status.json", status)
                    if args.idle_max_running > 0:
                        preflight = await wait_for_idle_relaxed(topology, parse_prometheus, args.idle_timeout, args.idle_max_running)
                    else:
                        preflight = await wait_for_idle(topology, parse_prometheus, args.idle_timeout)
                    async with httpx.AsyncClient(timeout=10) as http:
                        (await http.get(endpoints[mode]+"/health/live")).raise_for_status()
                    stop, unsafe, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
                    metrics = cell/f"{mode}-metrics.jsonl"
                    collector = asyncio.create_task(collect_high_load(
                        topology, parse_prometheus, metrics, stop, unsafe, ready))
                    child = None
                    started = time.monotonic()
                    try:
                        await asyncio.wait_for(ready.wait(), 30)
                        if any("error_type" in r for r in read_records(metrics)):
                            raise RuntimeError("initial telemetry unavailable")
                        command = [sys.executable, "/opt/cnu/run_rag_experiment.py",
                                   "--base-url", endpoints[mode], "--endpoint", "/api/generate/stream",
                                   "--questions", str(questions), "--output-dir", str(cell/"responses"),
                                   "--run-id", "sweep-trial", "--variant", mode,
                                   "--concurrency", str(level), "--repeats", "1",
                                   "--warmup", "0", "--timeout", str(args.timeout), "--store-responses",
                                   "--seed", "20260912"]
                        with (cell/f"{mode}-client.log").open("wb") as log:
                            child = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log)
                            last_count, last_progress = 0, time.monotonic()
                            print(f"START level={level} round={round_index} method={mode} n={count}", flush=True)
                            while child.returncode is None:
                                if unsafe.is_set():
                                    raise RuntimeError("engine safety threshold; see telemetry")
                                try:
                                    await asyncio.wait_for(asyncio.shield(child.wait()), 10)
                                except asyncio.TimeoutError:
                                    checkpoint = cell/f"responses/sweep-trial/{mode}/client_requests.checkpoint.jsonl"
                                    completed = read_records(checkpoint)
                                    done = len(completed)
                                    if done > last_count:
                                        last_count, last_progress = done, time.monotonic()
                                    reason = response_failure_reason(completed)
                                    if reason:
                                        raise RuntimeError(reason)
                                    if time.monotonic()-last_progress > max(240, args.timeout*0.5):
                                        raise RuntimeError("no completed request for too long")
                                    if time.monotonic()-started > 3600:
                                        raise RuntimeError("cell wall time exceeded 3600 seconds")
                                    status["current"].update(completed_requests=done, phase="requests",
                                                             elapsed_seconds=time.monotonic()-started)
                                    save(root/"status.json", status)
                                    print(f"PROGRESS level={level} round={round_index} method={mode} {done}/{count}", flush=True)
                        output = cell/f"responses/sweep-trial/{mode}/client_requests.jsonl"
                        records = read_records(output)
                        if child.returncode not in (0, 2) or len(records) != count:
                            raise RuntimeError("incomplete evaluator output")
                        if {r["question_id"] for r in records} != {r["id"] for r in rows}:
                            raise RuntimeError("question alignment mismatch")
                        if args.smoke and any(r["status"] != "ok" for r in records):
                            raise RuntimeError("smoke failed")
                        status["completed"].append({"level": level, "round": round_index, "method": mode,
                                                    "preflight": preflight, "requests": count,
                                                    "transport_ok": sum(r["status"] == "ok" for r in records),
                                                    "wall_seconds": time.monotonic()-started})
                        status["current"]["completed_requests"] = count
                        save(root/"status.json", status)
                    finally:
                        if child is not None and child.returncode is None:
                            child.terminate()
                            try:
                                await asyncio.wait_for(child.wait(), 15)
                            except asyncio.TimeoutError:
                                child.kill()
                                await child.wait()
                        stop.set()
                        await collector
        status["state"] = "complete"
        status["current"] = None
        print("COMPLETE", flush=True)
    except BaseException as exc:
        status.update(state="stopped", reason=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        status["updated"] = time.time()
        save(root/"status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--endpoint", action="append", required=True, help="arm=url; repeatable")
    p.add_argument("--arms", default="original,capability")
    p.add_argument("--levels", default="8,16,32,64,100")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--idle-timeout", type=int, default=180)
    p.add_argument("--idle-max-running", type=int, default=0,
                   help="tolerate this many running requests per engine at preflight (stuck request); waiting must be 0")
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    asyncio.run(run(p.parse_args()))
