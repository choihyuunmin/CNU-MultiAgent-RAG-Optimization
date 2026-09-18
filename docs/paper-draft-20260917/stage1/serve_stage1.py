import argparse
import hashlib
import importlib
import json
import secrets
import sys
import time
import uuid
from pathlib import Path

from cnu_rag_optimization.application_adapter import ApplicationAdapter
from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter


def emit(event):
    print("CNU_CAPABILITY_V1 " + json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "direct"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/app/src")
    topology = json.loads(Path("/opt/cnu/topology.json").read_text())
    for relative, expected in topology["application_source_sha256"].items():
        path = (Path("/app") / relative).resolve()
        if not path.is_relative_to("/app") or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("source mismatch")
    settings = importlib.import_module("config.settings")
    client = importlib.import_module("infra.llm.client")
    # Public package adapter: aliases only (no alias_to_resource/call_id fields).
    adapter = ApplicationAdapter(default_model=settings.MASTER_MODEL, aliases=topology["aliases"],
        on_event=lambda e: print("CNU_APP_ADAPTER_V1 " + json.dumps(e), flush=True))
    phases = InferenceOverlapAdapter(on_event=emit)
    client.acompletion_via_proxy = adapter.completion(phases.completion(client.acompletion_via_proxy))
    client.acompletion_stream_via_proxy = adapter.stream(phases.stream(client.acompletion_stream_via_proxy))
    search = importlib.import_module("agent.law_search_agent")
    original_search = search.run_search

    from review_dispatch_attachment import install
    install(search, args.mode, emit=emit)
    app = importlib.import_module("main_server").app
    emit({"event": "trial_config", "mode": args.mode, "source_verified": True,
          "engine_unchanged": True, "new_concurrency_limit": False,
          "scope": "search dispatch only", "call_elision": args.mode != "original", "stage": "dispatch-validation-stage1"})
    import uvicorn
    uvicorn.run(adapter.asgi(app), host="0.0.0.0", port=28000, access_log=False)


if __name__ == "__main__":
    main()
