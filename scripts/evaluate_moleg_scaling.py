"""Real SSE pipeline load: continuously refilled users or scheduled arrivals.

All client admission and scheduling delay is charged to user latency. Requests,
models, retrieval, generation and server settings are unchanged. Public outputs
contain hashes and metrics only. Each output directory is new and immutable on
rerun. No credentials, raw queries, answers or endpoint addresses are persisted.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnu_rag_optimization.admission import PressureGate
from moleg_scaling_metrics import describe, delta, parse_metrics, summarize_telemetry


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_cases(cases, count, seed):
    """Proportional stratified selection; identical set at every load and policy."""
    if count < 1 or count > len(cases):
        raise ValueError("case count outside available unique cases")
    groups = defaultdict(list)
    for case in cases:
        groups[case["kind"]].append(case)
    rng = random.Random(seed)
    quotas = {k: int(count * len(v) / len(cases)) for k, v in groups.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda k: (-(count * len(groups[k]) / len(cases) - quotas[k]), k))
    for k in order[:remaining]:
        quotas[k] += 1
    selected = []
    for kind in sorted(groups):
        rng.shuffle(groups[kind])
        selected.extend(groups[kind][:quotas[kind]])
    rng.shuffle(selected)
    return selected


async def request(client, base, case, gate, due, timeout, run_id, index, capture=None):
    entered = time.perf_counter()
    row = {"case_id": case["case_id"], "kind": case["kind"], "index": index,
           "question_sha256": digest(case["question"]), "ok": False,
           "scheduling_lag_s": max(0, entered - due), "events": []}
    wire_start = None
    first_token = first_event = None
    pieces = []
    result = None
    terminal_error = False
    async def execute():
        nonlocal wire_start, first_token, first_event, result, terminal_error
        async with gate.slot() if gate is not None else nullcontext():
            wire_start = time.perf_counter()
            row["admission_wait_s"] = wire_start - entered if gate is not None else 0.0
            row["gate_limit_at_start"] = gate.limit if gate is not None else None
            async with client.stream("POST", base.rstrip("/") + "/api/generate/stream",
                    json={"prompt": case["question"], "session_id": f"scale-{run_id}-{index}"}) as response:
                row["status"] = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    event = json.loads(data)
                    now = time.perf_counter()
                    if first_event is None:
                        first_event = now
                    stage = event.get("stage", "")
                    if stage == "token" and event.get("delta"):
                        if first_token is None:
                            first_token = now
                        pieces.append(event["delta"])
                    elif stage == "done":
                        result = event.get("result")
                    elif stage == "error":
                        terminal_error = True
                    if not row["events"] or row["events"][-1]["stage"] != stage:
                        row["events"].append({"stage": stage, "s": now - due})
            row["ok"] = isinstance(result, dict) and not terminal_error
    try:
        # A total deadline includes admission, HTTP time and response streaming.
        await asyncio.wait_for(execute(), timeout=max(.001, timeout - (entered - due)))
    except Exception as exc:
        row["error_type"] = type(exc).__name__
    finished = time.perf_counter()
    row.update(elapsed_s=finished - due,
               wire_s=finished - wire_start if wire_start is not None else None,
               ttft_s=first_token - due if first_token is not None else None,
               first_event_s=first_event - due if first_event is not None else None,
               streamed_chars=sum(map(len, pieces)), answer_sha256=digest("".join(pieces)))
    row.setdefault("admission_wait_s", finished - entered if wire_start is None else 0)
    result = result if isinstance(result, dict) else {}
    ids = [str(d.get("item_id") or d.get("id") or "") for d in result.get("laws", [])]
    row["evidence_ids_sha256"] = [digest(x) for x in ids]
    row["answer_chars"] = len(result.get("comment") or "")
    row["hitl"] = bool(result.get("hitl"))
    row["source_law_hit"] = (any(x.split("_")[0] == case["source_law_id"] for x in ids)
                              if case.get("source_law_id") else None)
    row["source_chunk_hit"] = (case["source_id"] in ids if case.get("source_id")
                               and case.get("reference_answer") else None)
    if capture is not None:
        from summarize_moleg_paper import known_error_response
        row['pipeline_error'] = known_error_response({'result': result})
        row['pipeline_ok'] = row['ok'] and row['pipeline_error'] is None
        capture({**row, 'result': result, 'streamed_answer': ''.join(pieces)})
    return row


async def trial(args, cases, concurrency, rate, policy, repeat, run_index, sink, telemetry_sink):
    import httpx
    maximum = concurrency if rate is None else args.open_max_active
    limit = maximum if policy == "baseline" else min(args.initial_limit, maximum)
    gate = None if policy == "baseline" else PressureGate(limit=limit, maximum=maximum, adaptive=policy == "adaptive")
    servers = dict(v.split("=", 1) for v in args.metrics)
    headers = {"Authorization": "Bearer " + os.environ[args.api_key_env]} if args.api_key_env else {}
    metadata = {"trial": run_index, "policy": policy, "repeat": repeat,
                "users": concurrency if rate is None else None, "arrival_rate": rate}
    samples, rows = [], []
    in_flight = 0
    stop = asyncio.Event()
    previous_preempt = {}
    run_id = uuid.uuid4().hex[:12]
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=10),
            trust_env=False, headers=headers,
            limits=httpx.Limits(max_connections=max(maximum, len(cases)),
                               max_keepalive_connections=maximum)) as client, \
            httpx.AsyncClient(timeout=3, trust_env=False, headers=headers) as observer:
        async def scrape():
            async def one(name, url):
                try:
                    response = await observer.get(url)
                    response.raise_for_status()
                    return name, parse_metrics(response.text)
                except Exception as exc:
                    return name, {"error_type": type(exc).__name__}
            sample = {**metadata, "utc": datetime.now(timezone.utc).isoformat(),
                      "gate_limit": gate.limit if gate else None,
                      "active": gate.active if gate else in_flight, "pending": len(gate.pending) if gate else 0,
                      "servers": dict(await asyncio.gather(*(one(k, v) for k, v in servers.items())))}
            samples.append(sample)
            telemetry_sink.write(json.dumps(sample) + "\n")
            values = [sample["servers"][k].get("values", {}) for k in servers]
            complete = bool(values) and all("num_requests_waiting" in v and
                ("kv_cache_usage_perc" in v or "gpu_cache_usage_perc" in v) for v in values)
            preempt = 0
            for k in servers:
                current = sample["servers"][k].get("values", {})
                change = delta(previous_preempt.get(k, {}), current, "num_preemptions_total")
                preempt += change or 0
                previous_preempt[k] = current
            if gate is not None:
                await gate.observe(waiting=max(v["num_requests_waiting"] for v in values) if complete else None,
                    kv_usage=max(v.get("kv_cache_usage_perc", v.get("gpu_cache_usage_perc", 0))
                                 for v in values) if complete else None, preemptions=preempt)
        async def monitor():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), args.sample_interval)
                except TimeoutError:
                    await scrape()
        await scrape()
        start = time.perf_counter()
        monitor_task = asyncio.create_task(monitor())
        queue = iter(enumerate(cases))
        in_flight = 0
        max_in_flight = 0
        active_area = 0.0
        last_change = start
        def active_change(change):
            nonlocal in_flight, max_in_flight, active_area, last_change
            now = time.perf_counter()
            active_area += in_flight * (now - last_change)
            last_change = now
            in_flight += change
            max_in_flight = max(max_in_flight, in_flight)
        async def invoke(index, case, due):
            active_change(1)
            try:
                capture = None
                if getattr(args, 'private_response_sink', None) is not None:
                    def capture(raw):
                        raw.update(metadata)
                        raw['policy'] = args.variant_name
                        args.private_response_sink.write(json.dumps(raw, ensure_ascii=False) + '\n')
                row = await request(client, args.base, case, gate, due, args.timeout, run_id, index, capture)
                row.update(metadata)
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) % 16 == 0:
                    print(json.dumps({**metadata, "completed": len(rows), "total": len(cases),
                                      "elapsed_s": round(time.perf_counter() - start, 1)}), flush=True)
            finally:
                active_change(-1)
        async def worker():
            for index, case in queue:
                await invoke(index, case, time.perf_counter())
        async def scheduled(index, case, offset):
            due = start + offset
            await asyncio.sleep(max(0, due - time.perf_counter()))
            await invoke(index, case, due)
        try:
            if rate is None:
                await asyncio.gather(*(worker() for _ in range(concurrency)))
            else:
                rng = random.Random(args.seed + repeat)
                offsets, offset = [], 0.0
                for _ in cases:
                    offsets.append(offset)
                    offset += rng.expovariate(rate) if args.arrivals == "poisson" else 1 / rate
                await asyncio.gather(*(scheduled(i, c, offset) for i, (c, offset) in enumerate(zip(cases, offsets))))
        finally:
            wall = time.perf_counter() - start
            stop.set()
            await monitor_task
            await scrape()
    metrics = {k: describe([r.get(k) for r in rows]) for k in (
        "elapsed_s", "wire_s", "ttft_s", "admission_wait_s", "scheduling_lag_s")}
    summary = {**metadata, "n": len(rows), "success": sum(r["ok"] for r in rows),
               "pipeline_success": sum(r.get("pipeline_ok", r["ok"]) for r in rows),
               "wall_s": wall, "throughput_rps": sum(r["ok"] for r in rows) / wall,
               "slo_goodput_rps": sum(r["ok"] and r["elapsed_s"] <= args.slo for r in rows) / wall,
               "slo_s": args.slo, "slo_attainment": sum(r["ok"] and r["elapsed_s"] <= args.slo for r in rows) / len(rows),
               "max_outstanding": max_in_flight, "mean_outstanding": active_area / wall,
               "gate_changes": gate.changes if gate else 0,
               "final_gate_limit": gate.limit if gate else None, "metrics": metrics,
               "errors": dict(Counter(r.get("error_type", "pipeline") for r in rows if not r["ok"])),
               "source_law_hit": describe([float(r["source_law_hit"]) if r["source_law_hit"] is not None else None for r in rows]),
               "source_chunk_hit": describe([float(r["source_chunk_hit"]) if r["source_chunk_hit"] is not None else None for r in rows]),
               "telemetry": {k: summarize_telemetry(samples, k) for k in servers}}
    print(json.dumps({**metadata, "done": len(rows), "success": summary["success"],
                      "mean_s": metrics["elapsed_s"]["mean"], "p95_s": metrics["elapsed_s"]["p95"],
                      "rps": summary["throughput_rps"]}), flush=True)
    return summary


async def run(args):
    cases = json.loads(args.cases.read_text())
    if len({c["question"] for c in cases}) != len(cases):
        raise ValueError("duplicate questions in source")
    selected = select_cases(cases, args.limit, args.seed)
    if not args.rates and max(args.users) > len(selected):
        raise ValueError("at least as many cases as users are needed")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"utc": datetime.now(timezone.utc).isoformat(),
        "scope": "live full SSE application; finite continuously refilled closed loop or scheduled open loop",
        "users": args.users, "rates": args.rates, "arrivals": args.arrivals,
        "policies": args.policies, "repeats": args.repeats, "seed": args.seed,
        "selected_cases": [{"case_id": c["case_id"], "kind": c["kind"], "question_sha256": digest(c["question"])} for c in selected],
        "input_file_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gate_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/cnu_rag_optimization/admission.py").read_bytes()).hexdigest(),
        "sample_interval_s": args.sample_interval, "timeout_s": args.timeout, "slo_s": args.slo,
        "initial_limit": args.initial_limit, "open_max_active": args.open_max_active,
        "client_host_gpu_not_used_for_inference": True,
        "limitations": ["finite workload has startup and drain; report actual outstanding concurrency",
                         "shared-instance metrics are not attributable to this client alone",
                         "no host GPU/DRAM bandwidth counters without external host telemetry",
                         "source labels are silver labels, not expert answer correctness"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summaries = []
    with (args.output / "requests.jsonl").open("x", buffering=1) as sink, \
         (args.output / "telemetry.jsonl").open("x", buffering=1) as telemetry_sink:
        for repeat in range(args.repeats):
            ordered = list(selected)
            random.Random(args.seed + repeat).shuffle(ordered)
            loads = list(args.rates or args.users)
            if repeat % 2:
                loads.reverse()
            for load_index, load in enumerate(loads):
                policies = list(args.policies)
                shift = (load_index + repeat) % len(policies)
                policies = policies[shift:] + policies[:shift]
                for policy in policies:
                    entry = await trial(args, ordered, load if not args.rates else 1,
                                        load if args.rates else None, policy, repeat,
                                        len(summaries), sink, telemetry_sink)
                    summaries.append(entry)
                    (args.output / "summary.json").write_text(json.dumps({"complete": False, "trials": summaries}, indent=2) + "\n")
                    if entry["success"] < entry["n"]:
                        raise RuntimeError("failed requests: stopping further load escalation; retained all rows")
    (args.output / "summary.json").write_text(json.dumps({"complete": True, "trials": summaries}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", action="append", default=[], help="role=http://host:port/metrics")
    parser.add_argument("--api-key-env", help="name of environment variable; value is never written")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--rates", type=float, nargs="+")
    parser.add_argument("--arrivals", choices=["regular", "poisson"], default="poisson")
    parser.add_argument("--policies", nargs="+", choices=["baseline", "fixed", "adaptive"], default=["baseline"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--initial-limit", type=int, default=8)
    parser.add_argument("--open-max-active", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--slo", type=float, default=30)
    args = parser.parse_args()
    if min(args.users) < 1 or args.repeats < 1 or args.sample_interval <= 0 or args.timeout <= 0:
        parser.error("users, repeats and time values must be positive")
    if args.rates and min(args.rates) <= 0:
        parser.error("arrival rates must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
