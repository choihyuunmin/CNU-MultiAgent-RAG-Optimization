#!/usr/bin/env python3
"""Interleave identical question blocks across unchanged app and admission arms.

Uses an integrating application's existing request runner and quality evaluator.
Artifacts stay local. Authentication is read from the application's environment
loader and never included in the topology or experiment protocol.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def topology_from_gateway(env, base):
    headers = {"Authorization": "Bearer " + (env.get("LITELLM_API_KEY") or env.get("OPENAI_API_KEY") or "")}
    response = httpx.get(base.removesuffix("/v1") + "/model/info", headers=headers, timeout=10)
    response.raise_for_status()
    aliases, receivers, identities = {}, {}, {}
    gateway = urlsplit(base)
    for row in response.json()["data"]:
        params = row.get("litellm_params", {})
        upstream = urlsplit(params.get("api_base") or "")
        # Only the existing OpenAI-compatible served roles in this integration.
        if not str(params.get("model", "")).startswith("openai/") or not upstream.hostname:
            continue
        identity = (upstream.scheme, upstream.hostname, upstream.port)
        if identity not in identities:
            identities[identity] = f"receiver-{len(identities) + 1}"
        resource = identities[identity]
        aliases[row["model_name"]] = resource
        host = gateway.hostname if upstream.hostname in {"localhost", "127.0.0.1"} else upstream.hostname
        receivers[resource] = f"{upstream.scheme}://{host}:{upstream.port}/metrics"
    if not aliases:
        raise RuntimeError("no receiver topology found")
    return {"aliases": aliases, "receivers": receivers}


async def run(args):
    app = args.application_root.resolve()
    scripts = app / "scripts" / "experiment"
    sys.path.insert(0, str(scripts))
    from run_experiment_bundle import _load_server_env, _preflight_remote_services, _llm_base, _wait_for_server, _stop_owned_process
    from run_interleaved_400 import COMMON_EXPERIMENT_FLAGS
    env = _load_server_env(app / ".env")
    common_flags = dict(COMMON_EXPERIMENT_FLAGS)
    # The historical benchmark disabled streaming. Preserve the application's
    # configured invocation mode instead: changing it violates this study's
    # unchanged-inference-API constraint even when all arms share the override.
    common_flags.pop("STREAM_ENABLED", None)
    env.update(common_flags)
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(app / "src"), str(app)]),
                "PYTHONUNBUFFERED": "1", "RAG_EXPERIMENT_TRACE_ENABLED": "1",
                "RAG_EXPERIMENT_RUN_ID": args.run_id})
    models = await asyncio.to_thread(_preflight_remote_services, env)
    topology = await asyncio.to_thread(topology_from_gateway, env, _llm_base(env))
    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["complexity"]].append(row)
    levels = sorted(grouped)
    if args.count % len(levels) or args.block_size % len(levels):
        raise ValueError("count and block size must be divisible by number of strata")
    if args.count % args.block_size:
        raise ValueError("count must be divisible by block size")
    rng = random.Random(args.seed)
    for level in levels:
        rng.shuffle(grouped[level])
        grouped[level] = grouped[level][:args.count // len(levels)]
        if len(grouped[level]) != args.count // len(levels):
            raise ValueError("insufficient questions in stratum")
    selected = [row for level in levels for row in grouped[level]]
    run_root = args.output_dir.resolve() / args.run_id
    if run_root.exists():
        raise FileExistsError("choose a new run ID; this runner never overwrites a prior run")
    run_root.mkdir(parents=True)
    (run_root / "blocks").mkdir()
    topology_path = run_root / "topology.local.json"
    write_json(topology_path, topology)
    selected_path = run_root / "questions.local.jsonl"
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or arms[0] != "control" or any(arm not in {"control", "control_repeat", "fifo", "coflow", "feedback"} for arm in arms):
        raise ValueError("unique supported arms starting with control required")
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
                "questions_per_arm": args.count, "strata": {level: len(grouped[level]) for level in levels},
                "question_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                "arms": arms, "initial_receiver_window": args.window, "max_receiver_window": 8,
                "concurrency": 4, "warmup_per_arm": args.warmup, "seed": args.seed,
                "repeats": 1, "block_size": args.block_size, "design": "cyclic block interleaving",
                "model_api_mutated": False, "request_payload_mutated": False,
                "models": models, "common_flags": common_flags,
                "stream_mode": {"source": "unchanged application environment", "STREAM_ENABLED": env.get("STREAM_ENABLED", "0")},
                "quality_gate": {"success_rate": 1.0, "document_recall": 0.98, "ndcg10": 0.97,
                                 "top1": 0.95, "answer_similarity": 0.97, "fidelity": 0.97},
                "app_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=app, text=True).strip(),
                "optimization_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "optimization_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in (ROOT / "src", ROOT / "scripts") for p in sorted(folder.rglob("*.py"))}}
    write_json(run_root / "protocol.json", protocol)
    python = str(app / ".venv" / "bin" / "python")
    servers, successful, latencies, cells = {}, {arm: 0 for arm in arms}, defaultdict(list), []
    try:
        for index, arm in enumerate(arms):
            folder = run_root / arm
            folder.mkdir()
            log = (folder / "server.log").open("wb")
            port = args.port + index
            arm_env = {**env, "RAG_EXPERIMENT_VARIANT": arm, "PORT": str(port)}
            command = [python, str(ROOT / "scripts" / "serve_admission_adapter.py"),
                       "--application-root", str(app), "--mode", "control" if arm == "control_repeat" else arm, "--window", str(args.window),
                       "--port", str(port), "--topology", str(topology_path)]
            proc = subprocess.Popen(command, cwd=app, env=arm_env, stdout=log, stderr=subprocess.STDOUT)
            servers[arm] = (proc, log, port)
            await asyncio.to_thread(_wait_for_server, f"http://127.0.0.1:{port}", proc, 90)
            print(f"ready arm={arm}", flush=True)
        per_level = args.block_size // len(levels)
        for block in range(args.count // args.block_size):
            block_rows = [row for level in levels for row in grouped[level][block * per_level:(block + 1) * per_level]]
            rng.shuffle(block_rows)
            path = run_root / "blocks" / f"block-{block + 1:03}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in block_rows), encoding="utf-8")
            offset = block % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                proc, _, port = servers[arm]
                if proc.poll() is not None:
                    raise RuntimeError(f"application stopped: {arm}")
                print(f"start block={block + 1} arm={arm} completed={successful[arm]}/{args.count}", flush=True)
                output = run_root / "cells" / f"block-{block + 1:03}"
                command = [python, str(scripts / "run_rag_experiment.py"), "--base-url", f"http://127.0.0.1:{port}",
                           "--endpoint", "/api/generate/stream", "--questions", str(path),
                           "--output-dir", str(output), "--run-id", args.run_id, "--variant", arm,
                           "--concurrency", "4", "--repeats", "1", "--warmup", str(args.warmup if block == 0 else 0),
                           "--timeout", "300", "--store-responses", "--seed", str(args.seed)]
                child = await asyncio.create_subprocess_exec(*command, cwd=app, env=env)
                try:
                    rc = await child.wait()
                finally:
                    if child.returncode is None:
                        child.terminate()
                        await child.wait()
                result = output / args.run_id / arm
                records = [json.loads(raw) for raw in (result / "client_requests.jsonl").read_text().splitlines() if raw.strip()]
                if rc or len(records) != args.block_size:
                    raise RuntimeError(f"incomplete cell: {arm}, rc={rc}")
                with (run_root / arm / "client_requests.jsonl").open("a", encoding="utf-8") as stream:
                    for row in records:
                        row["block_index"] = block
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                successful[arm] += sum(row["status"] == "ok" for row in records)
                latencies[arm].extend(row["duration_ms"] for row in records if row["status"] == "ok")
                cells.append({"block": block + 1, "arm": arm, "result": str(result.relative_to(run_root))})
                write_json(run_root / "progress.json", {"successful": successful, "expected": args.count,
                           "completed_cells": len(cells), "mean_seconds": {key: sum(val) / len(val) / 1000 for key, val in latencies.items()},
                           "cells": cells})
                print(f"done block={block + 1} arm={arm} completed={successful[arm]}/{args.count}", flush=True)
                if not all(row["status"] == "ok" for row in records):
                    raise RuntimeError("request failure: retained all results and stopped without survivor-only retry")
    finally:
        for proc, log, _ in servers.values():
            await asyncio.to_thread(_stop_owned_process, proc)
            log.close()
    if args.skip_quality:
        print(f"complete requests; quality pending: {run_root}", flush=True)
        return
    command = [python, str(scripts / "evaluate_baseline_fidelity.py"),
               "--baseline", str(run_root / "control" / "client_requests.jsonl"),
               "--env-file", str(app / ".env"), "--output-dir", str(run_root / "quality")]
    for arm in arms[1:]:
        command.extend(["--variant", f"{arm}={run_root / arm / 'client_requests.jsonl'}"])
    child = await asyncio.create_subprocess_exec(*command, cwd=app, env=env)
    if await child.wait():
        raise RuntimeError("quality evaluator failed")
    print(f"complete results={run_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiment-results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms", default="control,fifo,coflow,feedback")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--port", type=int, default=28720)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--skip-quality", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
