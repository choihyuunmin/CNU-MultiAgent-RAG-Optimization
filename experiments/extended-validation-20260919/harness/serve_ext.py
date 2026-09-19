"""Extended validation server: original / direct / overlap arms on the frozen search service.

original: unchanged search path (LLM tool-call rewrite), tracing only.
direct:   checked direct dispatch of the prepared search arguments (public package).
overlap:  direct dispatch + verified overlap of the preparation call with the
          classification call (CallOverlapAdapter). Prompts, models, options and
          application limits are unchanged in every arm.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from cnu_rag_optimization.application_adapter import ApplicationAdapter
from cnu_rag_optimization.call_overlap import CallOverlapAdapter
from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter


def emit(event):
    print("CNU_CAPABILITY_V1 " + json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "direct", "overlap"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/app/src")
    topology = json.loads(Path("/opt/cnu/topology.json").read_text())
    for relative, expected in topology["application_source_sha256"].items():
        path = (Path("/app") / relative).resolve()
        if not path.is_relative_to("/app") or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("source mismatch")
    settings = importlib.import_module("config.settings")
    client = importlib.import_module("infra.llm.client")
    adapter = ApplicationAdapter(default_model=settings.MASTER_MODEL, aliases=topology["aliases"],
        on_event=lambda e: print("CNU_APP_ADAPTER_V1 " + json.dumps(e), flush=True))
    phases = InferenceOverlapAdapter(on_event=emit)
    client.acompletion_via_proxy = adapter.completion(phases.completion(client.acompletion_via_proxy))
    client.acompletion_stream_via_proxy = adapter.stream(phases.stream(client.acompletion_stream_via_proxy))
    search = importlib.import_module("agent.law_search_agent")

    from review_dispatch_attachment import install as install_dispatch
    install_dispatch(search, "original" if args.mode == "original" else "direct", emit=emit)

    from moleg_call_overlap import install as install_overlap
    overlap = CallOverlapAdapter(on_event=emit) if args.mode == "overlap" else None
    overlap_record = install_overlap(adapter=overlap, emit=emit)

    app = importlib.import_module("main_server").app
    emit({"event": "trial_config", "mode": args.mode, "source_verified": True,
          "engine_unchanged": True, "new_concurrency_limit": False, "prompts_changed": False,
          "dispatch": "original" if args.mode == "original" else "direct",
          "call_overlap": overlap_record, "stage": "extended-validation-20260919"})
    import uvicorn
    uvicorn.run(adapter.asgi(app), host="0.0.0.0", port=28000, access_log=False)


if __name__ == "__main__":
    main()
