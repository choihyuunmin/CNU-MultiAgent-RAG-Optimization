"""Supervise the two-arm combined-harness experiment on the GPU host.

Arms (both run with global admission queues removed and immediate SSE emission):

  baseline  - stock application behaviour, observe-only adapter.
  combined  - worker `reasoning_effort=low` with an answer-free stall guard, plus
              the ProgramHarness ready-successor overlay on the search node.

Only owned loopback apps are started and stopped; the shared model servers are
read-only and never restarted. Identical 200-question set, identical order per
(load, repeat) across arms. Arm order is rotated per (repeat, load) so neither
arm is systematically favoured by time-of-day or cache drift. Failures are kept,
never retried into successes; load escalation stops on the first failed request.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_moleg_scaling import request, select_cases, digest  # noqa: E402
from moleg_scaling_metrics import describe, delta, parse_metrics, summarize_telemetry  # noqa: E402

MODEL_PORTS = [8000, 8001, 8005, 8006, 8020, 8030]
USERS = [1, 4, 16]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def model_inventory():
    models = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            argv = path.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if "--port" not in argv or not any("vllm" in s for s in argv):
                continue
            port = int(argv[argv.index("--port") + 1])
            if port not in MODEL_PORTS:
                continue
            digest_ = hashlib.sha256(path.joinpath("cmdline").read_bytes()).hexdigest()
            models.append({"pid": int(path.name), "port": port, "argv_sha256": digest_})
        except (OSError, ValueError, IndexError):
            continue
    return sorted(models, key=lambda m: m["port"])


async def wait_ready(base, timeout_s=180):
    import httpx
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(base + "/__scaling_state")
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            await asyncio.sleep(2)
    raise RuntimeError(f"app at {base} did not become ready")


async def models_healthy(metrics, headers):
    import httpx
    async with httpx.AsyncClient(timeout=5, trust_env=False, headers=headers) as client:
        for _, url in metrics.items():
            r = await client.get(url)
            r.raise_for_status()
            if "vllm:" not in r.text:
                return False
    return True


def launch_app(arm, root, port, python, app_root, app_env, serving_env, proxy_config,
               fingerprint):
    trace = root / "apps" / f"{arm}.trace.jsonl"
    workflow = root / "apps" / f"{arm}.workflow.jsonl"
    config = root / (f"{arm}.adapter.json")
    argv = [python, str(root / "scripts" / "launch_moleg_scaling_app.py"),
            "--app-root", str(app_root), "--app-env", str(app_env),
            "--serving-env", str(serving_env), "--proxy-config", str(proxy_config),
            "--trace", str(trace), "--port", str(port), "--emission", "immediate",
            "--adapter-config", str(config), "--adapter-trace", str(workflow)]
    if arm == "combined":
        argv += ["--program-overlap", "--program-fingerprint", fingerprint]
    log = (root / "apps" / f"{arm}.server.log").open("w")
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                            cwd=str(root / "scripts"), start_new_session=True)
    return proc, {"arm": arm, "pid": proc.pid, "port": port, "argv": argv}


async def scrape(observer, metrics, meta, gate_active, samples, sink, previous_preempt):
    async def one(name, url):
        try:
            r = await observer.get(url)
            r.raise_for_status()
            return name, parse_metrics(r.text)
        except Exception as exc:
            return name, {"error_type": type(exc).__name__}
    sample = {**meta, "utc": utc(), "active": gate_active(),
              "servers": dict(await asyncio.gather(*(one(k, v) for k, v in metrics.items())))}
    samples.append(sample)
    sink.write(json.dumps(sample) + "\n")


async def run_trial(args, base, cases, concurrency, arm, repeat, run_index, sink,
                    private_sink, telemetry_sink, metrics, headers):
    import httpx
    metadata = {"trial": run_index, "arm": arm, "repeat": repeat, "users": concurrency}
    rows, samples = [], []
    in_flight = [0]
    previous_preempt = {}
    run_id = uuid.uuid4().hex[:12]
    stop = asyncio.Event()

    class Args:
        pass
    ra = Args()
    ra.base = base
    ra.timeout = args.timeout
    ra.variant_name = arm
    ra.private_response_sink = private_sink

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=10),
            trust_env=False, headers=headers,
            limits=httpx.Limits(max_connections=max(concurrency, len(cases)),
                                max_keepalive_connections=concurrency)) as client, \
            httpx.AsyncClient(timeout=3, trust_env=False, headers=headers) as observer:
        async def monitor():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), args.sample_interval)
                except (asyncio.TimeoutError, TimeoutError):
                    await scrape(observer, metrics, metadata, lambda: in_flight[0],
                                 samples, telemetry_sink, previous_preempt)
        await scrape(observer, metrics, metadata, lambda: in_flight[0], samples,
                     telemetry_sink, previous_preempt)
        start = time.perf_counter()
        monitor_task = asyncio.create_task(monitor())
        queue = iter(enumerate(cases))

        def capture(raw):
            raw.update(metadata)
            raw["policy"] = arm
            private_sink.write(json.dumps(raw, ensure_ascii=False) + "\n")

        async def invoke(index, case):
            in_flight[0] += 1
            try:
                row = await request(client, base, case, None, time.perf_counter(),
                                    args.timeout, run_id, index, capture)
                row.update(metadata)
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) % 20 == 0:
                    print(json.dumps({**metadata, "completed": len(rows), "total": len(cases),
                                      "elapsed_s": round(time.perf_counter() - start, 1)}), flush=True)
            finally:
                in_flight[0] -= 1

        async def worker():
            for index, case in queue:
                await invoke(index, case)
        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            wall = time.perf_counter() - start
            stop.set()
            await monitor_task
            await scrape(observer, metrics, metadata, lambda: in_flight[0], samples,
                         telemetry_sink, previous_preempt)
    metric_keys = ("elapsed_s", "wire_s", "ttft_s", "first_event_s")
    summary = {**metadata, "n": len(rows), "success": sum(r["ok"] for r in rows),
               "pipeline_success": sum(r.get("pipeline_ok", r["ok"]) for r in rows),
               "wall_s": wall, "throughput_rps": sum(r["ok"] for r in rows) / wall if wall else 0,
               "slo_s": args.slo,
               "slo_attainment": sum(r["ok"] and r["elapsed_s"] <= args.slo for r in rows) / len(rows) if rows else 0,
               "metrics": {k: describe([r.get(k) for r in rows]) for k in metric_keys},
               "errors": dict(Counter(r.get("error_type", "pipeline") for r in rows if not r["ok"])),
               "source_law_hit": describe([float(r["source_law_hit"]) if r.get("source_law_hit") is not None else None for r in rows]),
               "source_chunk_hit": describe([float(r["source_chunk_hit"]) if r.get("source_chunk_hit") is not None else None for r in rows]),
               "telemetry": {k: summarize_telemetry(samples, k) for k in metrics}}
    print(json.dumps({**metadata, "done": len(rows), "success": summary["success"],
                      "mean_s": summary["metrics"]["elapsed_s"]["mean"],
                      "p95_s": summary["metrics"]["elapsed_s"]["p95"],
                      "rps": summary["throughput_rps"]}), flush=True)
    return summary


async def execute(args):
    os.umask(0o077)
    root = args.directory
    plan = json.loads((root / "plan.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    if len(cases) != plan["unique_questions"] or len({c["question"] for c in cases}) != len(cases):
        raise ValueError("frozen case set mismatch")
    headers = {"Authorization": "Bearer " + os.environ[args.api_key_env]} if args.api_key_env else {}
    metrics = dict(v.split("=", 1) for v in args.metrics)
    bases = {"baseline": f"http://127.0.0.1:{args.baseline_port}",
             "combined": f"http://127.0.0.1:{args.combined_port}"}

    status = root / "campaign-status.json"
    state = {"status": "preflight", "started_utc": utc(),
             "planned_requests": plan["planned_primary_requests"],
             "completed_requests": 0, "arms": list(bases), "release_allowed": False}
    write_json(status, state)

    write_json(root / "model-inventory-before.json", model_inventory())
    if not await models_healthy(metrics, headers):
        raise RuntimeError("model health check failed before start")

    procs = {}
    infos = []
    for arm, port in [("baseline", args.baseline_port), ("combined", args.combined_port)]:
        proc, info = launch_app(arm, root, port, args.python, args.app_root, args.app_env,
                                args.serving_env, args.proxy_config, plan["law_search_nodes_sha256"])
        procs[arm] = proc
        infos.append(info)
    write_json(root / "apps" / "apps.json", infos)

    stopped = None
    try:
        app_states = {}
        for arm, base in bases.items():
            app_states[arm] = await wait_ready(base)
        write_json(root / "app-state-before.json", app_states)
        if app_states["combined"].get("program_overlap") is not True:
            raise RuntimeError("combined arm did not enable the program overlay")

        selected = select_cases(cases, args.limit, plan["seed"])
        (root / "selected.json").write_text(json.dumps(
            [{"case_id": c["case_id"], "kind": c["kind"], "question_sha256": digest(c["question"])}
             for c in selected], ensure_ascii=False, indent=2) + "\n")

        state["status"] = "running"
        write_json(status, state)
        summaries = []
        requests_path = root / "requests.jsonl"
        private_path = root / "private-responses.jsonl"
        telemetry_path = root / "telemetry.jsonl"
        with requests_path.open("x", buffering=1) as sink, \
                private_path.open("x", buffering=1) as private_sink, \
                telemetry_path.open("x", buffering=1) as telemetry_sink:
            for repeat in range(plan["repeats"]):
                loads = list(USERS)
                if repeat % 2:
                    loads.reverse()
                for load_index, load in enumerate(loads):
                    arms = ["baseline", "combined"]
                    if (load_index + repeat) % 2:
                        arms.reverse()
                    for arm in arms:
                        entry = await run_trial(args, bases[arm], selected, load, arm, repeat,
                                                len(summaries), sink, private_sink,
                                                telemetry_sink, metrics, headers)
                        summaries.append(entry)
                        state["completed_requests"] += entry["n"]
                        write_json(status, state)
                        (root / "summary.json").write_text(json.dumps(
                            {"complete": False, "trials": summaries}, indent=2) + "\n")
                        if entry["success"] < entry["n"]:
                            stopped = "failed_requests_stop_escalation"
                            raise RuntimeError(stopped)
        (root / "summary.json").write_text(json.dumps({"complete": True, "trials": summaries}, indent=2) + "\n")
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "stopped"
        state["stop_reason"] = stopped or f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after_states = {}
        for arm, base in bases.items():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    after_states[arm] = (await client.get(base + "/__scaling_state")).json()
            except Exception as exc:
                after_states[arm] = {"error": type(exc).__name__}
        write_json(root / "app-state-after.json", after_states)
        for arm, proc in procs.items():
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for proc in procs.values():
            try:
                proc.wait(timeout=30)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        write_json(root / "model-inventory-after.json", model_inventory())
        state["ended_utc"] = utc()
        write_json(status, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--app-root", type=Path, required=True)
    p.add_argument("--app-env", type=Path, required=True)
    p.add_argument("--serving-env", type=Path, required=True)
    p.add_argument("--proxy-config", type=Path, required=True)
    p.add_argument("--baseline-port", type=int, default=28380)
    p.add_argument("--combined-port", type=int, default=28381)
    p.add_argument("--metrics", action="append", default=[], help="role=http://host:port/metrics")
    p.add_argument("--api-key-env")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--slo", type=float, default=30.0)
    args = p.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
