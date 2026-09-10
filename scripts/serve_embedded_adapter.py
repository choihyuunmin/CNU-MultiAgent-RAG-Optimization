#!/usr/bin/env python3
"""Serve original LLM calls with metadata observation, without admission control.

Legacy delivery/network modes are intentionally rejected. This runner does not
change application/session/DB gates or inference options. Sources must identify
the original unscheduled call functions in the fingerprint-pinned application.
Read overlap requires explicit dependency-safe attachment; it is not auto-enabled.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter


def attach_original_calls(client, adapter, *, completion_source, stream_source):
    # Resolve both before mutating either entry point.
    completion = getattr(client, completion_source)
    stream = getattr(client, stream_source)
    if not callable(completion) or not callable(stream):
        raise TypeError("original call sources must be callable")
    client.acompletion_via_proxy = adapter.completion(completion)
    client.acompletion_stream_via_proxy = adapter.stream(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True,
                        help="Private manifest containing application_source_sha256")
    parser.add_argument("--mode", choices=("original",), default="original")
    parser.add_argument("--completion-source", required=True)
    parser.add_argument("--stream-source", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28000)
    args = parser.parse_args()
    app_root = args.application_root.resolve()
    topology = json.loads(args.topology.read_text())
    fingerprints = topology.get("application_source_sha256", {})
    if not fingerprints:
        raise RuntimeError("nonempty application fingerprints required")
    for relative, expected in fingerprints.items():
        path = (app_root / relative).resolve()
        if not path.is_relative_to(app_root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("application fingerprint mismatch; refusing to attach")
    os.chdir(app_root)
    sys.path[:0] = [str(app_root / "src"), str(app_root)]
    if "main_server" in sys.modules:
        raise RuntimeError("adapter must be installed before main_server")
    client = importlib.import_module("infra.llm.client")
    adapter = InferenceOverlapAdapter(
        on_event=lambda event: print("CNU_PHASE_V1 " + json.dumps(event), flush=True))
    attach_original_calls(client, adapter, completion_source=args.completion_source,
                          stream_source=args.stream_source)
    app = importlib.import_module("main_server")
    print("CNU_ORIGINAL_CONFIG " + json.dumps({
        "mode": "original", "adapter_admission": False, "llm_arguments_unchanged": True,
        "engine_unchanged": True, "application_gates_unchanged": True,
        "read_overlap_enabled": False, "raw_text_logged": False}), flush=True)
    import uvicorn
    uvicorn.run(app.app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
