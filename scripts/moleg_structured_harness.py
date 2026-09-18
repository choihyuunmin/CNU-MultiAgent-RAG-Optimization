"""Opt-in structured-call harness on the isolated, fingerprinted MOLEG app."""
from __future__ import annotations

from functools import wraps
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from cnu_rag_optimization.structured_harness import (
    compact_selection_messages, compact_structured_payload, strict_json,
    encode_selection_ids, decode_selection_ids,
)


def encode_portable_selection(messages, *, min_candidates=16):
    """Thin MOLEG envelope binding; the reference compiler remains generic.

    Uses the separately measured 16-candidate policy, not the synthetic study's
    calibration. Stronger escape checks may bypass previously unseen inputs.
    """
    from cnu_rag_optimization.reference_harness import ReferenceContract, compile_references
    contract = ReferenceContract('moleg-selection-v1', ('laws',), 'id', ('selected_ids',))
    result = deepcopy(messages)
    for index, message in enumerate(result):
        text = message.get('content')
        begin, finish = '# 법령 검색 결과\n', '\n</search_results>'
        if not isinstance(text, str) or text.count(begin) != 1 or text.count(finish) != 1:
            continue
        start = text.index(begin) + len(begin)
        end = text.find(finish, start)
        if end < start:
            continue
        try:
            data = strict_json(text[start:end])
            if not isinstance(data.get('laws'), list) or len(data['laws']) < min_candidates:
                continue
        except (ValueError, TypeError, AttributeError):
            continue
        protected = [m['content'] for i, m in enumerate(messages)
                     if i != index and isinstance(m.get('content'), str)]
        compiled = compile_references(text[start:end], contract,
                                      protected=(*protected, text[:start], text[end:]))
        if compiled.enabled:
            message['content'] = text[:start] + compiled.encoded + text[end:]
            return result, {str(i): value for i, value in enumerate(compiled.originals)}
    return deepcopy(messages), {}


def install(config_path, trace_path):
    import infra.llm.client as llm
    from openai import AsyncOpenAI
    from moleg_model_transport import chat_payload
    from moleg_paper_runtime import bootstrap, trace_add
    from moleg_workflow_adapter import replace_aliases

    settings = json.loads(Path(config_path).read_text())
    allowed = {"compact_input", "compact_output", "short_ids", "min_short_id_candidates",
               "direct_structured", "typed_search", "capture", "fingerprints", "generic_reference_codec"}
    if set(settings) - allowed:
        raise ValueError("unknown structured harness setting")
    min_candidates = settings.get("min_short_id_candidates", 16)
    if type(min_candidates) is not int or min_candidates < 1:
        raise ValueError("min_short_id_candidates must be a positive integer")
    import agent.law_analysis_agent as selection
    import agent.law_search_agent as retrieval
    for module in (selection, retrieval):
        observed = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if settings.get("fingerprints", {}).get(module.__name__) != observed:
            raise ValueError("structured harness application fingerprint mismatch")
    if settings.get("compact_output") and not settings.get("direct_structured"):
        raise ValueError("compact decoding requires a verified direct vLLM route")

    _, serving, config, resolve = bootstrap()
    routes, clients = {}, {}
    for item in config.get("model_list", []):
        p = item.get("litellm_params", {})
        if not str(p.get("model", "")).startswith("openai/"):
            continue
        base = resolve(p["api_base"])
        if base not in clients:
            clients[base] = AsyncOpenAI(base_url=base,
                api_key=resolve(p.get("api_key")) or serving["VLLM_API_KEY"],
                timeout=120, max_retries=0)
        routes[item["model_name"]] = (clients[base], p["model"][7:])

    sink = None
    if settings.get("capture"):
        fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        sink = os.fdopen(fd, "w", buffering=1)
    original = llm.acompletion_via_proxy

    @wraps(original)
    async def chat(**kwargs):
        fmt = kwargs.get("response_format") or {}
        spec = fmt.get("json_schema") or {}
        name = spec.get("name", "")
        # Explicit schema identity; no guessing roles from prompt words.
        targeted = name in {"law_selection_response", "search_preparation"}
        incoming = kwargs
        started = time.perf_counter()
        if targeted and settings.get("compact_input") and name == "law_selection_response":
            kwargs = dict(kwargs, messages=compact_selection_messages(kwargs.get("messages", [])))
        mapping = {}
        if name == "law_selection_response" and settings.get("short_ids"):
            encoder = encode_portable_selection if settings.get("generic_reference_codec") else encode_selection_ids
            messages, mapping = encoder(kwargs.get("messages", []), min_candidates=min_candidates)
            kwargs = dict(kwargs, messages=messages)
        route = routes.get(str(kwargs.get("model") or llm.MASTER_MODEL))
        direct = targeted and settings.get("direct_structured") and route is not None
        compact = False
        if direct:
            client, model = route
            payload = chat_payload(kwargs, model=model, streaming=False,
                                   create=client.chat.completions.create)
            if settings.get("compact_output"):
                payload = compact_structured_payload(payload)
                compact = "structured_outputs" in (payload.get("extra_body") or {})
            response = await client.chat.completions.create(**payload)
            # Invalid output is an explicit failure, never silently cached/reused.
            if compact:
                strict_json(response.choices[0].message.content or "")
        else:
            response = await original(**kwargs)
        if mapping:
            try:
                restored = decode_selection_ids(response.choices[0].message.content or "", mapping)
            except (ValueError, TypeError, AttributeError):
                trace_add("structured_fallback", reason="invalid_selection_handle")
                response = await original(**incoming)
            else:
                response = response.model_copy(deep=True)
                response.choices[0].message.content = restored
        if targeted:
            usage = response.usage.model_dump() if response.usage else {}
            row = {"schema": name, "elapsed_s": time.perf_counter() - started,
                   "direct": bool(direct), "compact_output": compact,
                   "short_ids": len(mapping),
                   "input_chars_before": sum(len(str(m.get("content", ""))) for m in incoming.get("messages", [])),
                   "input_chars_after": sum(len(str(m.get("content", ""))) for m in kwargs.get("messages", [])),
                   "usage": usage}
            trace_add("structured_harness", **row)
            if sink:
                # Private only: used for stage replay, never exported as public metrics.
                sink.write(json.dumps({**row, "kwargs": incoming,
                    "output": response.choices[0].message.content}, ensure_ascii=False) + "\n")
        return response

    llm.acompletion_via_proxy = chat
    replace_aliases(original, chat, "acompletion_via_proxy")
    if settings.get("typed_search"):
        original_search = retrieval.run_search

        @wraps(original_search)
        async def typed_search(*, country, merged_keywords, search_terms, request_id):
            # Exactly the original agent's direct-fallback argument contract.
            payload = {"search_terms": search_terms, "country": country if country else None,
                       "keywords": list(merged_keywords)}
            raw = await retrieval.tool_search_laws(payload, request_id)
            laws, ids, source = retrieval._parse_search_payload(raw)
            trace_add("typed_dispatch", calls_elided=1, candidates=len(laws))
            return {"search_tool_result": raw, "laws_list": laws, "search_law_ids": ids,
                    "search_source": source, "search_params": payload}

        retrieval.run_search = typed_search
        replace_aliases(original_search, typed_search, "run_search")

    async def close():
        for client in clients.values():
            await client.close()
        if sink:
            sink.close()

    return close
