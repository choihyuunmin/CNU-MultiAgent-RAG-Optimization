"""Isolated fixed-K GPU smoke/pilot using frozen tokenized model inputs.

Run on the GPU host only after checking existing jobs and free memory. Each
invocation loads one engine, never restarts a service, and writes a new directory.
This measures stage latency; it does NOT produce engine decode calibration rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("nonempty frozen tokenized cases required")
    ids = set()
    for case in cases:
        if set(case) != {"case_id", "prompt_token_ids", "sampling_params"}:
            raise ValueError("case must have case_id, prompt_token_ids and sampling_params")
        if not isinstance(case["case_id"], str) or not case["case_id"] or case["case_id"] in ids:
            raise ValueError("unique nonempty case IDs required")
        ids.add(case["case_id"])
        tokens = case["prompt_token_ids"]
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("tokenized prompts required; do not retokenize with another model")
        settings = case["sampling_params"]
        if not isinstance(settings, dict) or settings.get("n", 1) != 1:
            raise ValueError("one output per frozen request required")
        if settings.get("max_tokens") is None:
            raise ValueError("preserve the original max_tokens explicitly")
    return cases


def arm_config(base, k):
    if type(k) is not int or k not in {0, 2, 4, 8, 16}:
        raise ValueError("unsupported pilot K")
    if not isinstance(base, dict) or not base.get("model"):
        raise ValueError("frozen engine config requires model")
    if "speculative_config" in base:
        raise ValueError("base config must describe the plain engine")
    # Warmup is outside timing; source and execution inputs are fingerprinted.
    if base.get("trust_remote_code") or base.get("hf_token"):
        raise ValueError("remote code/inline credentials are not part of this pilot")
    config = dict(base)
    if k:
        config["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": k,
            "prompt_lookup_min": 2, "prompt_lookup_max": 8,
        }
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-config", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--warmup-cases", type=Path, required=True)
    p.add_argument("--k", type=int, choices=[0, 2, 4, 8, 16], required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.levels != sorted(set(args.levels)) or args.levels[0] < 1:
        raise ValueError("positive repeats and sorted unique loads required")
    config = arm_config(json.loads(args.engine_config.read_text()), args.k)
    cases = validate_cases(json.loads(args.cases.read_text()))
    warmup = validate_cases(json.loads(args.warmup_cases.read_text()))
    if set(c["case_id"] for c in cases) & set(c["case_id"] for c in warmup):
        raise ValueError("warmup cases must be separate")
    if args.levels[-1] > len(cases):
        raise ValueError("requested concurrency exceeds number of cases")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("select an inspected experiment GPU via CUDA_VISIBLE_DEVICES")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "initializing", "kind": "fixed_k_gpu_stage_pilot", "k": args.k,
             "frozen_utc": datetime.now(timezone.utc).isoformat(), "levels": args.levels,
             "repeats": args.repeats, "cases_sha256": digest(cases),
             "warmup_sha256": digest(warmup), "engine_config_sha256": digest(config),
             "base_engine_config_sha256": digest(json.loads(args.engine_config.read_text())),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "promote_to_full_experiment": False, "decode_cost_calibration": False,
             "expected_measured_requests": len(cases) * len(args.levels) * args.repeats,
             "completed_measured_requests": 0}
    def save():
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")
    save()
    try:
        from vllm import LLM, SamplingParams
        state["vllm_version"] = importlib.metadata.version("vllm")
        engine = LLM(**config)
        def generate(batch):
            return engine.generate(
                prompts=[{"prompt_token_ids": c["prompt_token_ids"]} for c in batch],
                sampling_params=[SamplingParams(**c["sampling_params"]) for c in batch],
                use_tqdm=False)
        generate(warmup)
        state["status"] = "running"
        save()
        with (args.output / "rows.jsonl").open("x") as rows:
            for repeat in range(args.repeats):
                for concurrency in args.levels:
                    for start in range(0, len(cases), concurrency):
                        batch = cases[start:start+concurrency]
                        began = time.perf_counter()
                        try:
                            outputs = generate(batch)
                        except Exception as exc:
                            elapsed = time.perf_counter() - began
                            for case in batch:
                                rows.write(json.dumps({"case_id": case["case_id"], "repeat": repeat,
                                    "concurrency": concurrency, "batch_size": len(batch),
                                    "ok": False, "batch_wall_s": elapsed,
                                    "error_type": type(exc).__name__}) + "\n")
                            rows.flush()
                            raise
                        elapsed = time.perf_counter() - began
                        if len(outputs) != len(batch):
                            raise RuntimeError("engine output count mismatch")
                        for case, out in zip(batch, outputs):
                            if list(out.prompt_token_ids) != case["prompt_token_ids"] or len(out.outputs) != 1:
                                raise RuntimeError("engine changed request ordering, input or output count")
                            completion = out.outputs[0]
                            metrics = out.metrics
                            row = {"case_id": case["case_id"], "repeat": repeat, "concurrency": concurrency,
                                "batch_size": len(batch), "batch_wall_s": elapsed,
                                "ok": bool(out.finished and completion.finish_reason == "stop"),
                                "finish_reason": completion.finish_reason,
                                "input_sha256": digest(case), "output_sha256": digest(list(completion.token_ids)),
                                "completion_tokens": len(completion.token_ids),
                                "request_metrics_source": "vllm_request_metrics_not_decode_windows"}
                            if metrics is not None:
                                arrival = getattr(metrics, "arrival_time", None)
                                finished = getattr(metrics, "finished_time", None)
                                first = getattr(metrics, "first_token_time", None)
                                row["request_elapsed_s"] = finished - arrival if finished is not None and arrival is not None else None
                                row["ttft_s"] = first - arrival if first is not None and arrival is not None else None
                            rows.write(json.dumps(row) + "\n")
                            state["completed_measured_requests"] += 1
                        rows.flush()
                        save()
        state["status"] = "completed_requires_paired_analysis"
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
