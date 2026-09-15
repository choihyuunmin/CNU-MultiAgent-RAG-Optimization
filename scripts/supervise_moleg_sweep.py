"""General multi-arm sweep over concurrency levels, for the shared MOLEG backend.

Drives one or more isolated loopback app arms over an increasing concurrency list
to (a) compare optimization methods and (b) locate the hardware capacity limit.
The question pool is cycled to reach a target request count at each level. Single
repeat by default. Escalation auto-stops once the reference arm collapses (mass
failure or near-timeout mean), so the shared production models are not blasted at
maximum concurrency after the limit is already evident.

Only owned loopback apps are started/stopped; model servers are read-only and
never restarted. Every failure is retained; failures are never retried.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_moleg_scaling import request, select_cases, digest  # noqa: E402
from moleg_scaling_metrics import describe, parse_metrics, summarize_telemetry  # noqa: E402


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def launch_app(arm, root, python, app_root, app_env, serving_env, proxy_config, out):
    apps = out / "apps"
    apps.mkdir(mode=0o700, exist_ok=True)
    name = arm["name"]
    trace = apps / f"{name}.trace.jsonl"
    workflow = apps / f"{name}.workflow.jsonl"
    argv = [python, str(root / "scripts" / "launch_moleg_scaling_app.py"),
            "--app-root", str(app_root), "--app-env", str(app_env),
            "--serving-env", str(serving_env), "--proxy-config", str(proxy_config),
            "--trace", str(trace), "--port", str(arm["port"]), "--emission", "immediate",
            "--adapter-config", str(root / arm["adapter_config"]),
            "--adapter-trace", str(workflow)]
    if arm.get("program_overlap"):
        argv += ["--program-overlap", "--program-fingerprint", arm["fingerprint"]]
    if arm.get("no_ontology"):
        argv += ["--no-ontology"]
    log = (apps / f"{name}.server.log").open("w")
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                            cwd=str(root / "scripts"), start_new_session=True)
    return proc, {"name": name, "pid": proc.pid, "port": arm["port"], "argv": argv}


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
    raise RuntimeError(f"app at {base} not ready")


async def models_healthy(metrics, headers):
    import httpx
    async with httpx.AsyncClient(timeout=6, trust_env=False, headers=headers) as client:
        for _, url in metrics.items():
            try:
                r = await client.get(url)
                r.raise_for_status()
                if "vllm:" not in r.text:
                    return False
            except Exception:
                return False
    return True


async def scrape(observer, metrics, meta, in_flight, samples, sink):
    async def one(name, url):
        try:
            r = await observer.get(url)
            r.raise_for_status()
            return name, parse_metrics(r.text)
        except Exception as exc:
            return name, {"error_type": type(exc).__name__}
    sample = {**meta, "utc": utc(), "in_flight": in_flight[0],
              "servers": dict(await asyncio.gather(*(one(k, v) for k, v in metrics.items())))}
    samples.append(sample)
    sink.write(json.dumps(sample) + "\n")


async def run_trial(base, pool, level, arm, repeat, run_index, timeout, slo, sample_interval,
                    sink, private_sink, telemetry_sink, metrics, headers):
    import httpx
    meta = {"trial": run_index, "arm": arm, "repeat": repeat, "level": level, "requests": len(pool)}
    rows, samples = [], []
    in_flight = [0]
    run_id = uuid.uuid4().hex[:12]
    stop = asyncio.Event()
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), trust_env=False,
            headers=headers, limits=httpx.Limits(max_connections=len(pool) + level,
            max_keepalive_connections=level)) as client, \
            httpx.AsyncClient(timeout=4, trust_env=False, headers=headers) as observer:
        async def monitor():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), sample_interval)
                except (asyncio.TimeoutError, TimeoutError):
                    await scrape(observer, metrics, meta, in_flight, samples, telemetry_sink)
        await scrape(observer, metrics, meta, in_flight, samples, telemetry_sink)
        start = time.perf_counter()
        monitor_task = asyncio.create_task(monitor())
        queue = iter(enumerate(pool))

        def capture(raw):
            raw.update(meta)
            raw["policy"] = arm
            private_sink.write(json.dumps(raw, ensure_ascii=False) + "\n")

        async def invoke(index, case):
            in_flight[0] += 1
            try:
                row = await request(client, base, case, None, time.perf_counter(), timeout,
                                    run_id, index, capture)
                row.update(meta)
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                in_flight[0] -= 1

        async def worker():
            for index, case in queue:
                await invoke(index, case)
        try:
            await asyncio.gather(*(worker() for _ in range(level)))
        finally:
            wall = time.perf_counter() - start
            stop.set()
            await monitor_task
            await scrape(observer, metrics, meta, in_flight, samples, telemetry_sink)
    ok = sum(r["ok"] for r in rows)
    within_slo = sum(r["ok"] and r["elapsed_s"] <= slo for r in rows)
    summary = {**meta, "n": len(rows), "success": ok, "success_rate": ok / len(rows) if rows else 0,
               "wall_s": wall, "throughput_rps": ok / wall if wall else 0,
               "slo_s": slo, "slo_goodput_rps": within_slo / wall if wall else 0,
               "slo_attainment": within_slo / len(rows) if rows else 0,
               "metrics": {k: describe([r.get(k) for r in rows]) for k in ("elapsed_s", "ttft_s")},
               "successful_elapsed_s": describe([r["elapsed_s"] for r in rows if r["ok"]]),
               "errors": dict(Counter(r.get("error_type", "pipeline") for r in rows if not r["ok"])),
               "source_law_hit": describe([float(r["source_law_hit"]) if r.get("source_law_hit") is not None else None for r in rows]),
               "telemetry": {k: summarize_telemetry(samples, k) for k in metrics}}
    print(json.dumps({**meta, "n": summary["n"], "ok": ok, "succ_rate": round(summary["success_rate"], 3),
                      "mean_s": summary["metrics"]["elapsed_s"]["mean"],
                      "p95_s": summary["metrics"]["elapsed_s"]["p95"],
                      "good_rps": round(summary["slo_goodput_rps"], 3),
                      "rps": round(summary["throughput_rps"], 3)}), flush=True)
    return summary


async def execute(args):
    os.umask(0o077)
    root = args.directory
    out = root / args.run_subdir if args.run_subdir else root
    out.mkdir(mode=0o700, exist_ok=True)
    arms = json.loads((root / args.arms).read_text())
    cases = json.loads((root / args.cases_file).read_text())
    if len({c["question"] for c in cases}) != len(cases):
        raise ValueError("duplicate questions")
    selected = select_cases(cases, min(args.pool, len(cases)), args.seed)
    headers = {"Authorization": "Bearer " + os.environ[args.api_key_env]} if args.api_key_env else {}
    metrics = dict(v.split("=", 1) for v in args.metrics)
    reference = args.reference or arms[-1]["name"]

    status = out / "campaign-status.json"
    state = {"status": "preflight", "started_utc": utc(), "levels": args.levels,
             "arms": [a["name"] for a in arms], "reference_arm": reference,
             "completed_levels": [], "hardware_limit_level": None}
    write_json(status, state)
    if not await models_healthy(metrics, headers):
        raise RuntimeError("model health failed before start")

    procs, infos = {}, []
    for arm in arms:
        proc, info = launch_app(arm, root, args.python, args.app_root, args.app_env,
                                args.serving_env, args.proxy_config, out)
        procs[arm["name"]] = proc
        infos.append(info)
    write_json(out / "apps.json", infos)
    bases = {a["name"]: f"http://127.0.0.1:{a['port']}" for a in arms}

    try:
        app_states = {name: await wait_ready(base) for name, base in bases.items()}
        write_json(out / "app-state-before.json", app_states)
        state["status"] = "running"
        write_json(status, state)
        summaries = []
        collapsed = False
        with (out / "requests.jsonl").open("x", buffering=1) as sink, \
                (out / "private-responses.jsonl").open("x", buffering=1) as private_sink, \
                (out / "telemetry.jsonl").open("x", buffering=1) as telemetry_sink:
            for repeat in range(args.repeats):
                for li, level in enumerate(args.levels):
                    n = min(args.max_requests, max(args.min_requests, level))
                    rng = random.Random(args.seed + repeat * 1000 + level)
                    pool = [selected[i % len(selected)] for i in range(n)]
                    rng.shuffle(pool)
                    order = [a["name"] for a in arms]
                    if (li + repeat) % 2:
                        order = order[::-1]
                    ref_summary = None
                    for name in order:
                        entry = await run_trial(bases[name], pool, level, name, repeat, len(summaries),
                                                args.timeout, args.slo, args.sample_interval,
                                                sink, private_sink, telemetry_sink, metrics, headers)
                        summaries.append(entry)
                        if name == reference:
                            ref_summary = entry
                        write_json(out / "summary.json", {"complete": False, "trials": summaries})
                    state["completed_levels"].append(level)
                    write_json(status, state)
                    # Auto-stop escalation: reference arm collapsed at this level.
                    if ref_summary and (ref_summary["success_rate"] < args.collapse_success
                            or (ref_summary["successful_elapsed_s"]["mean"] or 0) >= args.timeout * 0.9):
                        state["hardware_limit_level"] = level
                        collapsed = True
                        break
                    if not await models_healthy(metrics, headers):
                        state["hardware_limit_level"] = level
                        state["stop_reason"] = "model_health_failed"
                        collapsed = True
                        break
                if collapsed:
                    break
        write_json(out / "summary.json", {"complete": True, "trials": summaries,
                                          "hardware_limit_level": state["hardware_limit_level"]})
        state["status"] = "completed"
    except Exception as exc:
        state["status"] = "stopped"
        state["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after = {}
        for name, base in bases.items():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    after[name] = (await client.get(base + "/__scaling_state")).json()
            except Exception as exc:
                after[name] = {"error": type(exc).__name__}
        write_json(out / "app-state-after.json", after)
        for name, proc in procs.items():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
        for proc in procs.values():
            try:
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
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
    p.add_argument("--arms", default="arms.json", help="relative to --directory")
    p.add_argument("--cases-file", default="cases.json")
    p.add_argument("--run-subdir", default="")
    p.add_argument("--levels", type=int, nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--pool", type=int, default=200, help="distinct questions to cycle")
    p.add_argument("--min-requests", type=int, default=200)
    p.add_argument("--max-requests", type=int, default=500)
    p.add_argument("--reference", help="arm whose collapse stops escalation (default: last arm)")
    p.add_argument("--collapse-success", type=float, default=0.5)
    p.add_argument("--metrics", action="append", default=[])
    p.add_argument("--api-key-env")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--slo", type=float, default=30.0)
    args = p.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
