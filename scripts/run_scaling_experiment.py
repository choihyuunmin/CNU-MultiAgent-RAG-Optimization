#!/usr/bin/env python3
"""Run identical real question sets at increasing concurrency, with queue/KV telemetry.

Never edits environment files, source, model parameters, or engine options.
Uses the integrating application's original request runner and response storage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODES = {"control", "request_window", "stage_fifo", "stage_coflow", "delivery", "delivery_repeat", "delivery_fifo", "delivery_coflow", "network_harness"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_questions(path, count, seed):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate question IDs")
    groups = defaultdict(list)
    for row in rows:
        if not row.get("country") or not row.get("query"):
            raise ValueError("country and query required")
        groups[row["complexity"]].append(row)
    if count % len(groups):
        raise ValueError("count must be divisible by number of difficulty groups")
    chosen, rng = [], random.Random(seed)
    for level in sorted(groups):
        rng.shuffle(groups[level])
        subset = groups[level][:count // len(groups)]
        if len(subset) != count // len(groups):
            raise ValueError("not enough real questions")
        chosen.extend(subset)
    return chosen


def topology_from_gateway(env, llm_base, orchestrator_capacity, worker_capacity):
    base = llm_base(env)
    host = urlsplit(base).hostname
    headers = {"Authorization": "Bearer " + (env.get("LITELLM_API_KEY") or env.get("OPENAI_API_KEY") or "")}
    with httpx.Client(timeout=10) as client:
        response = client.get(base.removesuffix("/v1") + "/model/info", headers=headers)
        response.raise_for_status()
        aliases, receivers, identities = {}, {}, {}
        for row in response.json().get("data", []):
            params = row.get("litellm_params", {})
            if not str(params.get("model", "")).startswith("openai/"):
                continue
            parsed = urlsplit(params.get("api_base") or "")
            if not parsed.hostname:
                raise ValueError("missing upstream topology")
            actual_host = host if parsed.hostname in {"localhost", "127.0.0.1"} else parsed.hostname
            identity = (parsed.scheme, actual_host, parsed.port)
            resource = identities.setdefault(identity, f"receiver-{len(identities)+1}")
            alias = row["model_name"]
            capacity = orchestrator_capacity if alias == "orchestrator" else worker_capacity
            if resource not in receivers:
                receivers[resource] = {"metrics": f"{parsed.scheme}://{actual_host}:{parsed.port}/metrics",
                                       "capacity": capacity, "model": params.get("model")}
            else:
                receivers[resource]["capacity"] = min(receivers[resource]["capacity"], capacity)
            aliases[alias] = resource
    if not aliases:
        raise ValueError("no OpenAI-compatible topology")
    return {"aliases": aliases, "receivers": receivers}


async def collect(topology, parse_prometheus, path, stop, unsafe, ready):
    streak = defaultdict(int)
    with path.open("w", encoding="utf-8") as stream:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) as client:
            while True:
                final_sample = stop.is_set()
                for resource, config in topology["receivers"].items():
                    row = {"time": time.time(), "resource": resource}
                    try:
                        response = await client.get(config["metrics"])
                        response.raise_for_status()
                        values = defaultdict(float)
                        for metric in parse_prometheus(response.text):
                            # Counters and gauges only; avoid per-bucket duplication.
                            if not metric["metric"].endswith("_bucket"):
                                values[metric["metric"]] += metric["value"]
                        row["metrics"] = dict(values)
                        waiting = values.get("vllm:num_requests_waiting")
                        cache = values.get("vllm:kv_cache_usage_perc")
                        risky = (waiting is not None and waiting > 64) or (cache is not None and cache > 0.95)
                        streak[resource] = streak[resource] + 1 if risky else 0
                        if streak[resource] >= 5:
                            unsafe.set()
                    except Exception as exc:
                        row["error_type"] = type(exc).__name__
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                ready.set()
                if final_sample:
                    break
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass


async def wait_for_idle(topology, parse_prometheus, timeout):
    """Wait read-only for two quiet samples; never stop someone else's traffic."""
    started, stable, last_log = time.monotonic(), 0, 0.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2)) as client:
        while True:
            observed = {}
            for resource, config in topology["receivers"].items():
                response = await client.get(config["metrics"])
                response.raise_for_status()
                gauges = defaultdict(float)
                present = set()
                for row in parse_prometheus(response.text):
                    if row["metric"] in {"vllm:num_requests_running", "vllm:num_requests_waiting"}:
                        gauges[row["metric"]] += row["value"]
                        present.add(row["metric"])
                if len(present) != 2:
                    raise RuntimeError("cannot establish idle state: missing engine gauges")
                observed[resource] = dict(gauges)
            quiet = all(value == 0 for gauges in observed.values() for value in gauges.values())
            stable = stable + 1 if quiet else 0
            elapsed = time.monotonic() - started
            if stable >= 2:
                return {"idle_wait_seconds": elapsed, "quiet_samples": stable, "gauges": observed}
            if elapsed >= timeout:
                raise RuntimeError("engines remain busy before test; no inference sent; arrange a quiet measurement window")
            if elapsed - last_log >= 15:
                print(f"PREFLIGHT waiting for existing inference to finish ({elapsed:.0f}s)", flush=True)
                last_log = elapsed
            await asyncio.sleep(1)


async def run(args):
    app = args.application_root.resolve()
    scripts = app / "scripts" / "experiment"
    sys.path.insert(0, str(scripts))
    from run_experiment_bundle import _load_server_env, _preflight_remote_services, _llm_base, _wait_for_server, _stop_owned_process
    from collect_vllm_metrics import parse_prometheus
    env = _load_server_env(app / ".env")
    # Do NOT import COMMON_EXPERIMENT_FLAGS: it overrides streaming and model behavior.
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(app / "src"), str(app)]),
                "PYTHONUNBUFFERED": "1", "RAG_EXPERIMENT_TRACE_ENABLED": "1",
                "RAG_EXPERIMENT_RUN_ID": args.run_id})
    models = await asyncio.to_thread(_preflight_remote_services, env)
    topology = await asyncio.to_thread(topology_from_gateway, env, _llm_base,
                                       args.orchestrator_capacity, args.worker_capacity)
    questions = select_questions(args.questions, args.count, args.seed)
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    if not concurrencies or min(concurrencies) < 1 or max(concurrencies) > 32 or len(set(concurrencies)) != len(concurrencies):
        raise ValueError("unique concurrency levels from 1 to 32 required")
    if args.count < max(concurrencies):
        raise ValueError("at least as many questions as simultaneous users required")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("unique supported modes required")
    folder = args.output_dir.resolve() / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    topology_path, question_path = folder / "topology.local.json", folder / "questions.local.jsonl"
    write(topology_path, topology)
    question_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "questions_per_cell": len(questions),
                "question_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
                "strata": dict(Counter(q["complexity"] for q in questions)), "concurrency": concurrencies,
                "modes": modes, "model_aliases": models, "stream_enabled": env.get("STREAM_ENABLED"),
                "request_window_candidate": args.request_window, "app_env_unchanged": True,
                "engine_unchanged": True, "replica_routing_tested": False,
                "llm_streaming_unchanged": True,
                "display_wait_removed_modes": [m for m in modes if m.startswith("delivery") or m == "network_harness"],
                "app_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=app, text=True).strip(),
                "optimization_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for sub in ("src", "scripts") for p in sorted((ROOT / sub).rglob("*.py"))},
                "application_source_sha256": {str(p.relative_to(app)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for sub in ("src", "scripts/experiment")
                                              for p in sorted((app / sub).rglob("*.py"))},
                "quality_status": "pending: compare stored responses; expert gold not provided",
                "gpu_device_metrics_status": "unavailable without separate SSH/exporter access",
                "safety_stop": "5 samples with waiting >64 or KV cache fraction >0.95",
                "idle_gate": {"quiet_samples": 2, "timeout_seconds": args.idle_timeout},
                "design": "closed-loop users, cyclic method order, same question set/seed, no retries by runner"}
    write(folder / "protocol.json", protocol)
    python = str(app / ".venv" / "bin" / "python")
    completed = []
    for index, concurrency in enumerate(concurrencies):
        offset = index % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            cell = f"c{concurrency:02}-{mode}"
            cell_dir = folder / cell
            cell_dir.mkdir()
            log = (cell_dir / "server.log").open("wb")
            client_log = (cell_dir / "client.log").open("wb")
            proc = None
            child = None
            stop, unsafe, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
            collector = None
            try:
                command = [python, str(ROOT / "scripts" / "serve_scaling_adapter.py"),
                           "--application-root", str(app), "--topology", str(topology_path),
                           "--port", str(args.port), "--mode", mode, "--request-window", str(args.request_window)]
                proc = subprocess.Popen(command, env={**env, "RAG_EXPERIMENT_VARIANT": cell},
                                        cwd=app, stdout=log, stderr=subprocess.STDOUT)
                await asyncio.to_thread(_wait_for_server, f"http://127.0.0.1:{args.port}", proc, 60)
                idle = await wait_for_idle(topology, parse_prometheus, args.idle_timeout)
                write(cell_dir / "idle_preflight.local.json", idle)
                print(f"START {cell} questions={len(questions)}", flush=True)
                collector = asyncio.create_task(collect(topology, parse_prometheus, cell_dir / "receiver_metrics.local.jsonl", stop, unsafe, ready))
                await ready.wait()
                command = [python, str(scripts / "run_rag_experiment.py"), "--base-url", f"http://127.0.0.1:{args.port}",
                           "--endpoint", "/api/generate/stream", "--questions", str(question_path),
                           "--output-dir", str(folder / "responses"), "--run-id", args.run_id, "--variant", cell,
                           "--concurrency", str(concurrency), "--repeats", "1", "--warmup", "0",
                           "--timeout", "180", "--store-responses", "--seed", str(args.seed)]
                child = await asyncio.create_subprocess_exec(*command, cwd=app, env=env,
                                                             stdout=client_log, stderr=asyncio.subprocess.STDOUT)
                result_path = folder / "responses" / args.run_id / cell
                while child.returncode is None:
                    try:
                        await asyncio.wait_for(child.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        checkpoint = result_path / "client_requests.checkpoint.jsonl"
                        done = len(checkpoint.read_text().splitlines()) if checkpoint.exists() else 0
                        print(f"PROGRESS {cell} {done}/{len(questions)}", flush=True)
                    if unsafe.is_set():
                        raise RuntimeError("safety stop: receiver queue/cache pressure; partial results retained")
                if child.returncode not in (0, 2):
                    raise RuntimeError(f"request runner failed rc={child.returncode}; partial results retained")
                rows = [json.loads(line) for line in (result_path / "client_requests.jsonl").read_text().splitlines() if line.strip()]
                if len(rows) != len(questions):
                    raise RuntimeError("incomplete cell; partial results retained")
                # A saturated control can time out. Keep failures in the denominator
                # and measure the other arms; never reinterpret this as a passed cell.
                successes = sum(row["status"] == "ok" for row in rows)
                summary = json.loads((result_path / "summary.json").read_text())
                completed.append({"cell": cell, "status": "complete" if successes == len(rows) else "complete_with_failures",
                                  "result_path": str(result_path.relative_to(folder)), "summary": summary["cells"][0]})
                write(folder / "progress.json", {"completed_cells": completed, "expected_cells": len(modes)*len(concurrencies)})
                print(f"DONE {cell} completed={len(rows)}/{len(questions)} successful={successes}", flush=True)
            finally:
                if child is not None and child.returncode is None:
                    child.terminate()
                    await child.wait()
                stop.set()
                if collector is not None:
                    await collector
                if proc is not None:
                    await asyncio.to_thread(_stop_owned_process, proc)
                log.close()
                client_log.close()
    print(f"COMPLETE {folder}; quality evaluation pending", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--modes", default="control,request_window,stage_coflow")
    parser.add_argument("--request-window", type=int, default=32)
    parser.add_argument("--orchestrator-capacity", type=int, default=8)
    parser.add_argument("--worker-capacity", type=int, default=4)
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--port", type=int, default=28740)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment-results")
    args = parser.parse_args()
    if min(args.count, args.request_window, args.orchestrator_capacity, args.worker_capacity, args.idle_timeout) < 1:
        parser.error("counts and capacities must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
