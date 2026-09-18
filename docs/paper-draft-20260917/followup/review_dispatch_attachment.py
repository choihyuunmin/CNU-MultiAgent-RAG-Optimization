"""Trial-only attachment (follow-up 2026-09-17): adds output hashes.

Same three arms and observations as the 2026-09-16 review attachment. Two
fields are added and nothing else changes: `output_hash` (SHA-256 of the raw
handler result string) on review_handler events, and `result_hash` (SHA-256 of
the canonical JSON of the object returned to the next stage) on review_boundary
events. Raw arguments, raw results and law text are still not emitted.
"""
import contextvars
import hashlib
import json
import secrets
import time
import uuid

from cnu_rag_optimization.typed_dispatch import try_typed_single_tool_dispatch

_branch = contextvars.ContextVar("review_search_branch", default=None)


def validate(a):
    return (type(a) is dict and set(a) == {"search_terms", "country", "keywords"}
            and type(a["search_terms"]) is str and bool(a["search_terms"].strip())
            and (a["country"] is None or type(a["country"]) is str)
            and type(a["keywords"]) is list
            and all(type(k) is str for k in a["keywords"]))


def freeze(a):
    return json.loads(json.dumps(a, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False))


def digest(a):
    return hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest_any(value):
    """Hash a handler string or a JSON-like object; None when unhashable."""
    try:
        if isinstance(value, (bytes, bytearray)):
            return hashlib.sha256(bytes(value)).hexdigest()
        if isinstance(value, str):
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), default=str).encode()).hexdigest()
    except Exception:
        return None


def install(search, mode, *, emit):
    if mode not in {"original", "direct", "capability"}:
        raise ValueError("unsupported review arm")
    # Signed execution is an optional legacy ablation. Original/direct need
    # only public package code and must not import an unpublished prototype.
    if mode == 'capability':
        from cnu_rag_optimization.capability_rpc import CapabilityRPC
    original_search, original_handler = search.run_search, search.tool_search_laws

    async def measured_handler(arguments, request_id):
        state = _branch.get()
        if state is None:
            return await original_handler(arguments, request_id)
        # Hashing is common trace work, not included in the handler's search time.
        emitted_hash = digest(arguments)
        changed = sorted(k for k in set(state["expected"]) | set(arguments)
                         if state["expected"].get(k) != arguments.get(k))
        entry = time.perf_counter_ns()
        if state["execute_started_ns"] is not None and state["verify_dispatch_ms"] is None:
            state["verify_dispatch_ms"] = (entry-state["execute_started_ns"])/1e6
        status = "error"
        result = None
        try:
            result = await original_handler(arguments, request_id)
            status = "ok"
            return result
        finally:
            duration = (time.perf_counter_ns()-entry)/1e6
            state["handler_ms"] += duration
            state["handler_calls"] += 1
            output_hash = digest_any(result) if result is not None else None
            state["output_hashes"].append(output_hash)
            emit({"event": "review_handler", "request_id": request_id,
                  "branch_id": state["branch_id"], "method": mode, "status": status,
                  "prepared_hash": state["prepared_hash"], "executed_hash": emitted_hash,
                  "arguments_equal": emitted_hash == state["prepared_hash"],
                  "changed_fields": changed, "handler_ms": duration,
                  "output_hash": output_hash,
                  "output_chars": len(result) if isinstance(result, str) else None})

    search.tool_search_laws = measured_handler

    async def wrapped(*, country, merged_keywords, search_terms, request_id):
        expected = {"search_terms": search_terms, "country": country if country else None,
                    "keywords": list(merged_keywords)}
        state = {"branch_id": uuid.uuid4().hex, "expected": freeze(expected),
                 "prepared_hash": digest(expected), "handler_ms": 0.0, "handler_calls": 0,
                 "execute_started_ns": None, "verify_dispatch_ms": None, "output_hashes": []}
        emit({"event": "review_prepared", "request_id": request_id,
              "branch_id": state["branch_id"], "method": mode,
              "prepared_hash": state["prepared_hash"]})
        token = _branch.set(state)
        started = time.perf_counter_ns()
        status, effective = "error", mode
        setup_ms = issue_ms = copy_validate_ms = 0.0
        result = None
        try:
            if mode == "original" or not validate(expected):
                effective = "original" if mode == "original" else "original_fallback"
                result = await original_search(country=country, merged_keywords=merged_keywords,
                                               search_terms=search_terms, request_id=request_id)
            else:
                async def handler(arguments):
                    return await measured_handler(arguments, request_id)
                if mode == "capability":
                    scope = uuid.uuid4().hex
                    rpc = CapabilityRPC(key=secrets.token_bytes(32), audience="search-dispatch-v1",
                                        validators={"search": validate}, handlers={"search": handler})
                    setup_end = time.perf_counter_ns()
                    envelope = rpc.issue(scope=scope, operation="search", arguments=expected)
                    issue_end = time.perf_counter_ns()
                    setup_ms = (setup_end-started)/1e6
                    issue_ms = (issue_end-setup_end)/1e6
                    state["execute_started_ns"] = time.perf_counter_ns()
                    value = await rpc.execute(envelope, scope=scope)
                else:
                    state["execute_started_ns"] = time.perf_counter_ns()
                    dispatched = await try_typed_single_tool_dispatch(
                        available_tools={'search': handler}, candidate_tool='search',
                        arguments=expected, argument_validator=validate)
                    if not dispatched.dispatched:
                        raise ValueError('validated dispatch rejected: ' + dispatched.reason)
                    value = dispatched.value
                laws, ids, source = search._parse_search_payload(value)
                result = {"search_tool_result": value, "laws_list": laws,
                          "search_law_ids": ids, "search_source": source,
                          "search_params": {"search_terms": search_terms,
                                            "keywords": merged_keywords, "country": country}}
            status = "ok"
            return result
        finally:
            total_ms = (time.perf_counter_ns()-started)/1e6
            # Reset context even if the trace sink fails. A failed invocation
            # must not leak its branch state into the caller's next operation.
            _branch.reset(token)
            emit({"event": "review_boundary", "request_id": request_id,
                  "branch_id": state["branch_id"], "method": mode, "effective_method": effective,
                  "status": status, "total_ms": total_ms, "handler_ms": state["handler_ms"],
                  "non_handler_ms": total_ms-state["handler_ms"],
                  "handler_calls": state["handler_calls"], "setup_ms": setup_ms,
                  "issue_ms": issue_ms, "copy_validate_ms": copy_validate_ms,
                  "verify_dispatch_ms": state["verify_dispatch_ms"],
                  "output_hashes": list(state["output_hashes"]),
                  "result_hash": digest_any(result) if result is not None else None,
                  "law_ids_hash": digest_any(list(result.get("search_law_ids", []))) if isinstance(result, dict) else None,
                  "timing_note": "includes common tracing; excludes prepared-trace setup; search timed separately; hashing after timer"})

    search.run_search = wrapped
