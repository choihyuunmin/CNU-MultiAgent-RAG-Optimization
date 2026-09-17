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
from cnu_rag_optimization.capability_rpc import CapabilityRPC, CapabilityRejected
from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter


def emit(event):
    print("CNU_CAPABILITY_V1 " + json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "direct", "capability"], required=True)
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
        alias_to_resource=topology["aliases"],
        on_event=lambda e: print("CNU_APP_ADAPTER_V1 " + json.dumps(e), flush=True))
    phases = InferenceOverlapAdapter(on_event=emit)
    client.acompletion_via_proxy = adapter.completion(phases.completion(client.acompletion_via_proxy))
    client.acompletion_stream_via_proxy = adapter.stream(phases.stream(client.acompletion_stream_via_proxy))
    search = importlib.import_module("agent.law_search_agent")
    original_search = search.run_search

    def validate(a):
        return (type(a) is dict and set(a) == {"search_terms", "country", "keywords"}
            and type(a["search_terms"]) is str and bool(a["search_terms"].strip())
            and (a["country"] is None or type(a["country"]) is str)
            and type(a["keywords"]) is list and all(type(k) is str for k in a["keywords"]))

    async def wrapped_search(*, country, merged_keywords, search_terms, request_id):
        expected = {"search_terms": search_terms, "country": country if country else None,
                    "keywords": list(merged_keywords)}
        if not validate(expected):
            emit({"event": "dispatch_fallback", "reason": "contract_ineligible"})
            return await original_search(country=country, merged_keywords=merged_keywords,
                                         search_terms=search_terms, request_id=request_id)
        scope = uuid.uuid4().hex
        async def handler(arguments):
            return await search.tool_search_laws(arguments, request_id)
        rpc = CapabilityRPC(key=secrets.token_bytes(32), audience="search-dispatch-v1",
            validators={"search": validate}, handlers={"search": handler})
        t0 = time.perf_counter()
        envelope = rpc.issue(scope=scope, operation="search", arguments=expected)
        issued_ms = (time.perf_counter() - t0) * 1000
        value = await rpc.execute(envelope, scope=scope)
        laws, ids, source = search._parse_search_payload(value)
        emit({"event": "sealed_dispatch", "request_id": request_id,
              "issue_ms": issued_ms, "elapsed_ms": (time.perf_counter() - t0) * 1000,
              "saved_dispatch_calls": 1, "law_count": len(laws), "raw_text_logged": False})
        return {"search_tool_result": value, "laws_list": laws, "search_law_ids": ids,
            "search_source": source, "search_params": {"search_terms": search_terms,
            "keywords": merged_keywords, "country": country}}

    from review_dispatch_attachment import install
    install(search, args.mode, emit=emit)
    app = importlib.import_module("main_server").app
    # Load sweep only: raise the isolated instance's admission capacity (same as the
    # 2026-09-13 sweep). Models, engines, prompts and the search path are unchanged.
    import os
    capacity = int(os.environ.get("CNU_APP_CAPACITY", "100"))
    from api.middleware.rate_limit import GlobalWaitQueueMiddleware
    for middleware in app.user_middleware:
        if middleware.cls is GlobalWaitQueueMiddleware:
            middleware.kwargs["max_concurrency"] = capacity
    emit({"event": "scaling_capacity", "configured": capacity,
          "generate_queue_env": os.environ.get("MAX_CONCURRENT_GENERATE_REQUESTS")})
    emit({"event": "trial_config", "mode": args.mode, "source_verified": True,
          "engine_unchanged": True, "new_concurrency_limit": False,
          "scope": "search dispatch only", "call_elision": args.mode != "original"})
    import uvicorn
    uvicorn.run(adapter.asgi(app), host="0.0.0.0", port=28000, access_log=False)


if __name__ == "__main__":
    main()
