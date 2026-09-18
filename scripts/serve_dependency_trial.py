"""Serve one isolated dependency-scheduling trial arm using the existing app.

Both schedules run the original model/tool path. Unlike older combined launchers,
this entry point changes no model option, stream pacing, application limit,
dispatch function, or cache. Credentials are supplied by the app environment.
It neither provisions infrastructure nor launches a load test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def verify_sources(app_root, topology, nodes_sha256):
    root = Path(app_root).resolve()
    digests = topology.get("application_source_sha256")
    target = "src/core/query_loop/law_search/nodes.py"
    if not isinstance(digests, dict) or digests.get(target) != nodes_sha256:
        raise ValueError("topology must pin the reviewed search-node source")
    for relative, expected in digests.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("source path outside application or missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("application source mismatch")


class EventSink:
    """File-like bridge; each overlay JSON event becomes one stdout line."""

    def write(self, line):
        sys.stdout.write(line)

    def flush(self):
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--nodes-sha256", required=True)
    parser.add_argument("--schedule", choices=["barrier", "ready"], required=True)
    parser.add_argument("--port", type=int, default=28010)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use a non-loopback address only in an isolated private test instance")
    args = parser.parse_args()
    topology = json.loads(args.topology.read_text())
    verify_sources(args.app_root, topology, args.nodes_sha256)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(args.app_root.resolve() / "src"))
    from cnu_rag_optimization.application_adapter import ApplicationAdapter
    from moleg_program_overlay import apply_program_overlay

    settings = importlib.import_module("config.settings")
    client = importlib.import_module("infra.llm.client")
    adapter = ApplicationAdapter(default_model=settings.MASTER_MODEL, aliases=topology["aliases"],
        on_event=lambda event: print("CNU_APP_ADAPTER_V1 " + json.dumps(event), flush=True))
    client.acompletion_via_proxy = adapter.completion(client.acompletion_via_proxy)
    client.acompletion_stream_via_proxy = adapter.stream(client.acompletion_stream_via_proxy)
    sink = EventSink()
    _, record = apply_program_overlay(fingerprint=args.nodes_sha256,
        start_when_ready=args.schedule == "ready", event_sink=sink, harness_sink=sink)
    app = importlib.import_module("main_server").app
    print("CNU_DEPENDENCY_TRIAL " + json.dumps({
        **record, "model_options_changed": False, "dispatch_elision": False,
        "application_limits_changed": False, "stream_pacing_changed": False,
        "application_source_sha256": topology["application_source_sha256"],
    }), flush=True)
    import uvicorn
    uvicorn.run(adapter.asgi(app), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
