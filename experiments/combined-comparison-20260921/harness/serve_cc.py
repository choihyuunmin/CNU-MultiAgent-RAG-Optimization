import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from cnu_rag_optimization.application_adapter import ApplicationAdapter
from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter
from cnu_rag_optimization.selection_budget import SelectionBudget

# arm -> (dispatch path at the search boundary, selection-input reduction on/off)
MODES = {
    "original": ("original", False),
    "direct": ("direct", False),
    "reduce": ("original", True),
    "combined": ("direct", True),
}


def emit(event):
    print("CNU_CAPABILITY_V1 " + json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--text-max-chars", type=int, default=int(os.environ.get("CNU_SELECTION_TEXT_MAX_CHARS", "300")))
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

    dispatch_mode, use_budget = MODES[args.mode]
    from review_dispatch_attachment import install as install_dispatch
    install_dispatch(search, dispatch_mode, emit=emit)

    from selection_budget_attachment import install as install_budget
    budget = SelectionBudget(text_max_chars=args.text_max_chars) if use_budget else None
    budget_record = install_budget(budget=budget, emit=emit)

    app = importlib.import_module("main_server").app
    capacity = int(os.environ.get("CNU_APP_CAPACITY", "100"))
    from api.middleware.rate_limit import GlobalWaitQueueMiddleware
    for middleware in app.user_middleware:
        if middleware.cls is GlobalWaitQueueMiddleware:
            middleware.kwargs["max_concurrency"] = capacity
    emit({"event": "scaling_capacity", "configured": capacity,
          "generate_queue_env": os.environ.get("MAX_CONCURRENT_GENERATE_REQUESTS")})
    emit({"event": "trial_config", "mode": args.mode, "source_verified": True,
          "engine_unchanged": True, "new_concurrency_limit": False, "prompts_changed": False,
          "dispatch": dispatch_mode, "selection_budget": budget_record,
          "stage": "combined-comparison-20260921"})
    import uvicorn
    uvicorn.run(adapter.asgi(app), host="0.0.0.0", port=28000, access_log=False)


if __name__ == "__main__":
    main()
