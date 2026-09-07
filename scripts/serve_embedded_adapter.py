#!/usr/bin/env python3
"""Run an opt-in adapter against an existing application image. No source edits."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from cnu_rag_optimization import CompletionPolicy, CompletionRouter, NetworkHarness, ReceiverSpec, ReplicaSpec
from cnu_rag_optimization.application_adapter import ApplicationAdapter
from serve_scaling_adapter import emit, observed_gate, remove_display_waits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--mode", choices=("original", "delivery", "network"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28000)
    args = parser.parse_args()
    app_root = args.application_root.resolve()
    os.chdir(app_root)
    sys.path[:0] = [str(app_root / "src"), str(app_root)]
    topology = json.loads(args.topology.read_text())
    for relative, expected in topology["application_source_sha256"].items():
        path = (app_root / relative).resolve()
        if not path.is_relative_to(app_root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("application fingerprint mismatch; refusing to attach")
    settings = importlib.import_module("config.settings")
    client = importlib.import_module("infra.llm.client")
    # Install before the application imports aliases of these original functions.
    if "main_server" in sys.modules:
        raise RuntimeError("adapter must be installed before main_server")
    rate_limit = importlib.import_module("api.middleware.rate_limit")
    rate_limit.GlobalWaitQueueMiddleware = observed_gate()  # Preserve the original limit.
    scheduler = CompletionRouter(
        [ReceiverSpec(name, config["capacity"]) for name, config in topology["receivers"].items()],
        [ReplicaSpec(alias, resource, alias, 1000.0) for alias, resource in topology["aliases"].items()],
        CompletionPolicy(ordering="fair"), on_event=lambda e: emit("CNU_COMPLETION_V1", e))
    network = NetworkHarness(scheduler, buffer_chunks=64, on_event=lambda e: emit("CNU_HARNESS_V1", e))
    adapter = ApplicationAdapter(default_model=settings.MASTER_MODEL, aliases=topology["aliases"],
        harness=network if args.mode == "network" else None, on_event=lambda e: emit("CNU_APP_ADAPTER_V1", e))
    client.acompletion_via_proxy = adapter.completion(client.acompletion_via_proxy)
    client.acompletion_stream_via_proxy = adapter.stream(client.acompletion_stream_via_proxy)
    app = importlib.import_module("main_server")
    original_display = None
    if args.mode != "original":
        original_display = remove_display_waits(importlib.import_module("core.query_loop._helpers"),
                                                importlib.import_module("api.router.generate"))
    concurrency = importlib.import_module("api.concurrency")
    emit("CNU_EMBEDDED_CONFIG", {"mode": args.mode, "application_fingerprints_verified": True,
        "generate_window_unchanged": concurrency.generate_queue._global.max_concurrency,
        "llm_arguments_unchanged": True, "engine_unchanged": True,
        "original_display_defaults": original_display,
        "network_harness_enabled": args.mode == "network"})
    import uvicorn
    uvicorn.run(adapter.asgi(app.app), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
