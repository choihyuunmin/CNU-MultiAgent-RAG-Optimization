#!/usr/bin/env python3
"""Measure opt-in application endpoints with the integrating app's unchanged evaluator."""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from run_scaling_experiment import collect, select_questions, wait_for_idle, write


async def run(args):
    app = args.application_root.resolve()
    sys.path.insert(0, str(app / "scripts/experiment"))
    from collect_vllm_metrics import parse_prometheus
    endpoints = dict(item.split("=", 1) for item in args.endpoint)
    if len(endpoints) != len(args.endpoint):
        raise ValueError("unique endpoint names required")
    for name, url in endpoints.items():
        parsed = urlsplit(url)
        if name not in {"original", "delivery", "network", "original_repeat"} or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("known mode and credential-free HTTP URL required")
    if not 1 <= args.concurrency <= 32 or args.concurrency > args.count:
        raise ValueError("concurrency must be 1..32 and no greater than question count")
    questions = select_questions(args.questions, args.count, args.seed)
    folder = args.output_dir / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    question_path = folder / "questions.local.jsonl"
    question_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    topology = json.loads(args.topology.read_text())
    topology["receivers"] = {k: v for k, v in topology["receivers"].items() if "metrics" in v}
    write(folder / "protocol.json", {"run_id": args.run_id, "questions": len(questions),
        "question_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
        "concurrency": args.concurrency, "endpoints": endpoints, "application_root": str(app),
        "application_image": args.application_image, "source_modified": False,
        "display_wait_removed_modes": [n for n in endpoints if n in {"delivery", "network"}],
        "request_window_unchanged": True, "engine_unchanged": True,
        "design": "same image; sequential conditions; same questions; no evaluator retries; expert accuracy unavailable"})
    completed = []
    for mode, url in endpoints.items():
        cell = f"c{args.concurrency:02}-{mode}"
        cell_dir = folder / cell
        cell_dir.mkdir()
        idle = await wait_for_idle(topology, parse_prometheus, 60)
        write(cell_dir / "idle.local.json", idle)
        stop, unsafe, ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
        collector = asyncio.create_task(collect(topology, parse_prometheus,
            cell_dir / "receiver_metrics.local.jsonl", stop, unsafe, ready))
        child = None
        try:
            await ready.wait()
            command = [str(app / ".venv/bin/python"), str(app / "scripts/experiment/run_rag_experiment.py"),
                "--base-url", url, "--endpoint", "/api/generate/stream", "--questions", str(question_path),
                "--output-dir", str(folder / "responses"), "--run-id", args.run_id, "--variant", cell,
                "--concurrency", str(args.concurrency), "--repeats", "1", "--warmup", "0",
                "--timeout", "180", "--store-responses", "--seed", str(args.seed), "--api-key-env", args.api_key_env]
            with (cell_dir / "client.log").open("wb") as log:
                child = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
                print(f"START {cell} {len(questions)} questions", flush=True)
                result_dir = folder / "responses" / args.run_id / cell
                while child.returncode is None:
                    if unsafe.is_set():
                        raise RuntimeError("engine safety limit reached; stopping own trial client")
                    try:
                        await asyncio.wait_for(asyncio.shield(child.wait()), timeout=15)
                    except asyncio.TimeoutError:
                        checkpoint = result_dir / "client_requests.checkpoint.jsonl"
                        count = len(checkpoint.read_text().splitlines()) if checkpoint.exists() else 0
                        print(f"PROGRESS {cell} {count}/{len(questions)}", flush=True)
            if child.returncode not in (0, 2):
                raise RuntimeError(f"evaluator exited {child.returncode}; preserve partial results")
            records = [json.loads(s) for s in (result_dir / "client_requests.jsonl").read_text().splitlines()]
            if len(records) != len(questions) or {r["question_id"] for r in records} != {q["id"] for q in questions}:
                raise ValueError("incomplete or unaligned result set")
            completed.append({"cell": cell, "completed": len(records),
                              "successful": sum(r["status"] == "ok" for r in records)})
            write(folder / "progress.json", {"completed": completed, "expected": len(endpoints)})
            print(f"DONE {cell} {completed[-1]}", flush=True)
        finally:
            if child is not None and child.returncode is None:
                child.terminate()
                await child.wait()
            stop.set()
            await collector
    print(f"COMPLETE {folder}; answer agreement not yet evaluated", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--application-image", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True, help="mode=http://127.0.0.1:port")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--api-key-env", default="MOLEG_API_KEY")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment-results"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
