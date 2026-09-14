"""Isolated application adapter for the 2026-09 paper experiments.

Loads application code and secrets from operator-supplied paths. Never changes
the deployed source or model server. Import before importing main_server.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

TRACE = contextvars.ContextVar("paper_trace", default=None)
SPEC = contextvars.ContextVar("paper_preparation", default=None)


def bootstrap():
    from dotenv import dotenv_values, load_dotenv
    import yaml

    root = Path(os.environ["MOLEG_RAG_ROOT"]).resolve()
    load_dotenv(root / ".env")
    serving = dotenv_values(os.environ["MOLEG_SERVING_ENV"])
    config = yaml.safe_load(Path(os.environ["MOLEG_PROXY_CONFIG"]).read_text())

    def resolve(value):
        if isinstance(value, str) and value.startswith("os.environ/"):
            name = value.split("/", 1)[1]
            return os.environ.get(name) or serving.get(name)
        return value

    key = resolve(config["general_settings"]["master_key"])
    os.environ["LITELLM_API_KEY"] = key
    os.environ["OPENAI_API_KEY"] = key
    os.environ["EMBEDDING_API_KEY"] = serving["VLLM_API_KEY"]
    os.environ["LOG_LEVEL"] = "ERROR"
    sys.path.insert(0, str(root / "src"))
    return root, serving, config, resolve


def trace_add(kind, **fields):
    row = TRACE.get()
    if row is not None:
        row.setdefault(kind, []).append(fields)


def evidence_text(query, doc, budget=600, window=False):
    """Bounded reranker view; original returned evidence is never truncated."""
    header = " | ".join(str(doc.get(k) or "") for k in ("country", "title", "subject"))
    header = header[: min(240, budget // 2)]
    body = str(doc.get("content") or "")
    remaining = max(0, budget - len(header) - 1)
    if not window or len(body) <= remaining:
        return (header + "\n" + body[:remaining])[:budget]
    terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", query.lower()))
    # Search all fixed windows. Selection uses only the request and candidate,
    # never the source labels or evaluation answers.
    stride = max(1, remaining // 2)
    starts = list(range(0, len(body), stride))
    start = max(starts, key=lambda p: (sum(t in body[p:p + remaining].lower() for t in terms), -p))
    return (header + "\n" + body[start:start + remaining])[:budget]


def install(profile="baseline", *, allow_preparation_overlap=True):
    root, serving, config, resolve = bootstrap()
    from openai import AsyncOpenAI
    import requests
    import infra.llm.client as llm
    from agent import validation_agent, law_search_agent
    import core.query_loop.runner as runner
    import api.controller.generate_controller as controller
    import tools.search_tool.engine as engine
    from infra.vector_db.vector_store import VectorStore

    # Common prerequisite: the guardrail endpoint requires the serving key too.
    # Keep its existing inference, timeout, and fallback logic in every arm.
    from agent import guardrail
    import httpx
    class GuardrailClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs['headers'] = {**kwargs.get('headers', {}),
                                 'Authorization': 'Bearer ' + serving['VLLM_API_KEY']}
            super().__init__(*args, **kwargs)
        async def post(self, *args, **kwargs):
            start = time.perf_counter()
            response = await super().post(*args, **kwargs)
            trace_add('guardrail', elapsed_s=time.perf_counter()-start, status=response.status_code)
            return response
    guardrail.httpx = SimpleNamespace(AsyncClient=GuardrailClient)

    original_post = requests.post
    def measured_post(url, *args, **kwargs):
        start = time.perf_counter()
        response = original_post(url, *args, **kwargs)
        trace_add('rerank', elapsed_s=time.perf_counter()-start, status=response.status_code)
        return response
    engine.requests = SimpleNamespace(post=measured_post, RequestException=requests.RequestException)

    fast = profile != "baseline"
    quality = profile in ("quality", "window", "balanced")
    # Keys and endpoints are runtime configuration, not part of artifacts.
    clients = {}
    routes = {}
    if fast:
        for item in config.get("model_list", []):
            p = item.get("litellm_params", {})
            model = str(p.get("model", ""))
            if not model.startswith("openai/"):
                continue
            base = resolve(p["api_base"])
            key = resolve(p.get("api_key")) or serving["VLLM_API_KEY"]
            if base not in clients:
                clients[base] = AsyncOpenAI(base_url=base, api_key=key, timeout=120, max_retries=0)
            routes[item["model_name"]] = (clients[base], model[len("openai/"):])

    from moleg_model_transport import install_standard_transport, chat_payload
    install_standard_transport()
    original_chat = llm.acompletion_via_proxy
    original_stream = llm.acompletion_stream_via_proxy

    def direct_kwargs(kwargs, streaming=False):
        model = str(kwargs.get("model") or llm.MASTER_MODEL)
        route = routes.get(model)
        if route is None:
            return None, None
        payload = chat_payload(kwargs, model=route[1], streaming=streaming,
                               create=route[0].chat.completions.create)
        if streaming and profile == 'balanced' and route[1] == 'openai/gpt-oss-20b':
            payload.setdefault('reasoning_effort', 'low')
        return route[0], payload

    async def chat(**kwargs):
        start = time.perf_counter()
        cli, payload = direct_kwargs(kwargs) if fast else (None, None)
        try:
            response = await cli.chat.completions.create(**payload) if cli else await original_chat(**kwargs)
            usage = response.usage.model_dump() if response.usage else {}
            trace_add("llm", model=kwargs.get("model"), elapsed_s=time.perf_counter()-start,
                      stream=False, usage=usage,
                      input_sha256=hashlib.sha256(json.dumps(kwargs,sort_keys=True,default=str).encode()).hexdigest(),
                      output_sha256=hashlib.sha256(json.dumps(response.choices[0].message.model_dump(),sort_keys=True,default=str).encode()).hexdigest(),
                      finish_reason=response.choices[0].finish_reason)
            return response
        except Exception as e:
            trace_add("errors", component="llm", error=type(e).__name__)
            raise

    async def stream(**kwargs):
        start = time.perf_counter()
        cli, payload = direct_kwargs(kwargs, True) if fast else (None, None)
        source = await cli.chat.completions.create(**payload) if cli else original_stream(**kwargs)
        chunks, chars, reasoning_chars = 0, 0, 0
        capture_path = os.environ.get('MOLEG_CAPTURE_PATH')
        if capture_path:
            with open(capture_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'session_id':(TRACE.get() or {}).get('session_id'),
                    'profile':profile,'kwargs':kwargs}, ensure_ascii=False)+'\n')
        try:
            async for chunk in source:
                chunks += 1
                if chunk.choices:
                    chars += len(chunk.choices[0].delta.content or "")
                    delta = chunk.choices[0].delta.model_dump()
                    reasoning_chars += len(delta.get('reasoning') or delta.get('reasoning_content') or '')
                yield chunk
        finally:
            trace_add("llm", model=kwargs.get("model"), elapsed_s=time.perf_counter()-start,
                      stream=True, chunks=chunks, characters=chars, reasoning_characters=reasoning_chars)
            if cli:
                await source.close()
            else:
                await source.aclose()

    llm.acompletion_via_proxy = chat
    llm.acompletion_stream_via_proxy = stream
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith(('agent.', 'infra.', 'core.')):
            continue
        if getattr(module, 'acompletion_via_proxy', None) is original_chat:
            module.acompletion_via_proxy = chat
        if getattr(module, 'acompletion_stream_via_proxy', None) is original_stream:
            module.acompletion_stream_via_proxy = stream

    # Record rank before the endpoint later sorts returned laws by item_id.
    original_search = engine.SearchEngine.search_laws
    def search(self, *args, **kwargs):
        start = time.perf_counter()
        out = original_search(self, *args, **kwargs)
        trace_add("retrieval", elapsed_s=time.perf_counter()-start,
                  laws=[{k: d.get(k) for k in ("id", "country", "title", "subject")} for d in out.get("laws", [])])
        return out
    engine.SearchEngine.search_laws = search

    # Rerank authentication + fixed candidate cardinality. Both variants expose
    # failures in the trace, so HTTP 200 cannot masquerade as a healthy pipeline.
    local = threading.local()
    original_rerank = engine.SearchEngine._rerank_results
    def rerank(self, query, documents, top_k=None, score_min=None):
        if not quality:
            return original_rerank(self, query, documents, top_k, score_min)
        if not documents:
            return documents
        if not hasattr(local, "session"):
            local.session = requests.Session()
        texts = ([str(d.get('content') or d.get('title') or '')[:engine.RERANK_DOCUMENT_TEXT_MAX_CHARS]
                  for d in documents] if profile == 'balanced' else
                 [evidence_text(query, d, engine.RERANK_DOCUMENT_TEXT_MAX_CHARS,
                               profile == "window") for d in documents])
        start = time.perf_counter()
        response = local.session.post(engine.RERANK_API_URL,
            headers={"Authorization": "Bearer " + serving["VLLM_API_KEY"]},
            json={"model": engine.RERANK_MODEL_NAME, "query": query[:engine.RERANK_QUERY_MAX_CHARS],
                  "documents": texts}, timeout=30)
        trace_add("rerank", elapsed_s=time.perf_counter()-start, status=response.status_code,
                  candidates=len(documents))
        response.raise_for_status()
        ranked = response.json()["results"]
        if sorted(x["index"] for x in ranked) != list(range(len(documents))):
            raise ValueError("reranker returned an incomplete candidate permutation")
        ranked.sort(key=lambda x: (-x["relevance_score"], x["index"]))
        return [documents[x["index"]] for x in ranked
                if score_min is None or x["relevance_score"] >= score_min][:(top_k or engine.RERANK_TOP_K)]
    engine.SearchEngine._rerank_results = rerank

    # Preserve real incoming visible deltas for a later no-inference byte-level
    # transport replay. This also measures deliberately paced whole-text paths.
    import core.query_loop._helpers as helpers
    original_emit = helpers.emit_streaming_delta
    async def measured_emit(queue, delta, request_id='-', **kwargs):
        trace_add('stream_input', delta=delta)
        return await original_emit(queue, delta, request_id, **kwargs)
    helpers.emit_streaming_delta = measured_emit
    for name in ('agent.law_analysis_agent','core.query_loop.assistant.nodes'):
        module=sys.modules.get(name)
        if module is not None and hasattr(module,'emit_streaming_delta'):
            module.emit_streaming_delta=measured_emit

    if fast:
        async def typed_search(*, country, merged_keywords, search_terms, request_id):
            payload = {"country": country or None, "keywords": list(merged_keywords), "search_terms": search_terms}
            raw = await law_search_agent.tool_search_laws(payload, request_id)
            laws, ids, source = law_search_agent._parse_search_payload(raw)
            return {"search_tool_result": raw, "laws_list": laws, "search_law_ids": ids,
                    "search_source": source, "search_params": payload}
        law_search_agent.run_search = typed_search

        # Share *in-flight* identical embeddings only. No completed-result cache.
        from concurrent.futures import Future
        vs = engine.search_engine.vector_store
        original_embed = vs.embeddings.embed_query
        pending, lock = {}, threading.Lock()
        def embed(text):
            with lock:
                owner = text not in pending
                future = pending.setdefault(text, Future())
            if not owner:
                return list(future.result())
            try:
                result = original_embed(text)
                future.set_result(result)
                return result
            except BaseException as exc:
                future.set_exception(exc)
                raise
            finally:
                with lock:
                    pending.pop(text, None)
        vs.embeddings.embed_query = embed
        ensure = vs._ensure_search_pipeline
        ready, pipeline_lock = False, threading.Lock()
        def ensure_once():
            nonlocal ready
            with pipeline_lock:
                if not ready:
                    ensure()
                    ready = True
        vs._ensure_search_pipeline = ensure_once

        # Exact-input speculative preparation; classification remains authoritative.
        # A different subquestion/history never consumes the speculative result.
        original_prep = validation_agent.run_preparation
        def signature(history):
            return json.dumps(history, ensure_ascii=False, sort_keys=True)
        async def preparation(*, history, request_id):
            spec = SPEC.get()
            if spec and spec[0] == signature(history) and not spec[2]:
                spec[2] = True
                trace_add("overlap", consumed=True)
                return await spec[1]
            return await original_prep(history=history, request_id=request_id)
        validation_agent.run_preparation = preparation
        unoptimized_execute = controller.execute_generate
        async def execute(**kwargs):
            history = kwargs.get("history", [])
            # Only speculate on new single-turn sessions containing a legal
            # request; no session changes and no search before authoritative routing.
            q = kwargs.get("user_prompt", "")
            allowed = allow_preparation_overlap and len(history) == 1 and bool(re.search(r"법|조문|규정", q))
            spec = [signature(history), asyncio.create_task(original_prep(
                history=copy.deepcopy(history), request_id=kwargs["request_id"])), False] if allowed else None
            token = SPEC.set(spec)
            try:
                return await unoptimized_execute(**kwargs)
            finally:
                SPEC.reset(token)
                if spec:
                    if not spec[1].done():
                        spec[1].cancel()
                    await asyncio.gather(spec[1], return_exceptions=True)
        controller.execute_generate = execute

        # Remove artificial sleeps; preserve each emitted character in order.
        import core.query_loop._helpers as helpers
        async def emit(queue, delta, request_id="-", **kwargs):
            trace_add('stream_input', delta=delta)
            if queue is not None and delta:
                await queue.put({"stage": "token", "delta": delta})
                helpers.mark_tokens_emitted(queue)
        helpers.emit_streaming_delta = emit
        for name in ("agent.law_analysis_agent", "core.query_loop.assistant.nodes"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "emit_streaming_delta"):
                module.emit_streaming_delta = emit

    # Trace logs deliberately omit prompts, credentials and full document text.
    before_trace_prep=validation_agent.run_preparation
    async def trace_prep(**kwargs):
        result=await before_trace_prep(**kwargs)
        fields={k:result.get(k) for k in ['country','transformed_query','keywords_original','keywords_transformed']}
        trace_add('preparation',country=result.get('country'),
                  output_sha256=hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
        return result
    validation_agent.run_preparation=trace_prep
    original_execute = controller.execute_generate
    trace_path = os.environ.get("MOLEG_TRACE_PATH")
    async def traced_execute(**kwargs):
        row = {"request_id": kwargs["request_id"], "session_id": kwargs["session_id"], "profile": profile}
        try:
            from cnu_rag_optimization.adaptive import current_workflow_trace_id
            workflow_id = current_workflow_trace_id()
            if workflow_id:
                row['workflow_trace_id'] = workflow_id
        except ImportError:
            pass  # legacy isolated runs do not require the workflow package
        token = TRACE.set(row)
        start = time.perf_counter()
        try:
            result = await original_execute(**kwargs)
            row["response_chars"] = len(result.comment or "")
            return result
        finally:
            row["elapsed_s"] = time.perf_counter()-start
            TRACE.reset(token)
            if trace_path:
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    controller.execute_generate = traced_execute
    logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    install(os.environ.get("MOLEG_STUDY_PROFILE", "baseline"))
    import uvicorn
    from main_server import app
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["MOLEG_STUDY_PORT"]), log_level="error")
