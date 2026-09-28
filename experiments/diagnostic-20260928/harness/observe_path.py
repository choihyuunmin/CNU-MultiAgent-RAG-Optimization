"""Read-only, content-free observations of the frozen legal-search graph.

Install after the two existing attachments and before importing main_server.
The wrappers return the original objects and never catch an application error.
"""

import contextvars
import hashlib
import importlib
import json
import time
import uuid
from functools import wraps

from cnu_rag_optimization.application_adapter import _request

_branch = contextvars.ContextVar("cnu_diag_branch", default=None)
_selection = contextvars.ContextVar("cnu_diag_selection", default=None)
EXPERIMENT = "diagnostic-20260928"


def _ids(items, key="id"):
    if not isinstance(items, (list, tuple)):
        return []
    out = []
    for item in items:
        value = item.get(key) if isinstance(item, dict) else item
        value = str(value or "").strip()
        if value:
            out.append(value)
    return out


def _summary(value):
    if not isinstance(value, dict):
        return {}
    out = {}
    for name, key in (("candidates", "laws_list"), ("selected", "selected_ids"),
                      ("final", "final_laws"), ("search_ids", "search_law_ids")):
        if key in value:
            ids = _ids(value.get(key), "law_id" if key == "final_laws" else "id")
            out[name + "_count"] = len(value.get(key) or [])
            out[name + "_id_count"] = len(ids)
            out[name + "_ids_sha256"] = hashlib.sha256(json.dumps(sorted(set(ids))).encode()).hexdigest()
    for key in ("search_mode", "search_source", "validation_result", "response_mode"):
        if key in value:
            out[key] = str(value[key] or "")[:80]
    for key in ("early_exit", "has_relevant_laws"):
        if key in value:
            out[key] = bool(value[key])
    if "search_tool_result" in value:
        out["search_tool_chars"] = len(value.get("search_tool_result") or "")
    if "classification" in value:
        c = value["classification"]
        out["classification_type"] = str(getattr(c, "task_type", ""))[:80]
    return out


def install(*, emit, client):
    def event(kind, **fields):
        context = _request.get() or {}
        emit({"event": kind, "experiment_id": EXPERIMENT,
              "case_id": context.get("case_id"), "root_id": context.get("root_id"),
              "branch_id": _branch.get(), **fields})

    runner = importlib.import_module("core.query_loop.runner")
    graph = importlib.import_module("core.query_loop.law_search.graph")
    builder = importlib.import_module("core.query_loop.builder")
    analysis = importlib.import_module("agent.law_analysis_agent")

    original_run = runner._run_single

    @wraps(original_run)
    async def run_single(initial_state, *args, **kwargs):
        branch_id = uuid.uuid4().hex
        token = _branch.set(branch_id)
        started = time.perf_counter()
        timeout = kwargs.get("timeout", args[0] if args else runner._GRAPH_TIMEOUT_SECONDS)
        event("branch_start", request_id=initial_state.get("request_id"), timeout_s=timeout,
              state=_summary(initial_state))
        try:
            result = await original_run(initial_state, *args, **kwargs)
            event("branch_end", request_id=initial_state.get("request_id"), status="ok",
                  duration_ms=round((time.perf_counter() - started) * 1000, 3),
                  state=_summary(result))
            return result
        except BaseException as exc:
            event("branch_end", request_id=initial_state.get("request_id"), status="error",
                  error_type=type(exc).__name__,
                  duration_ms=round((time.perf_counter() - started) * 1000, 3))
            raise
        finally:
            _branch.reset(token)

    runner._run_single = run_single

    def node_wrapper(name, original):
        @wraps(original)
        async def wrapped(state):
            started = time.perf_counter()
            event("stage_start", stage=name, request_id=state.get("request_id"),
                  state=_summary(state))
            try:
                result = await original(state)
                event("stage_end", stage=name, request_id=state.get("request_id"),
                      status="ok", duration_ms=round((time.perf_counter() - started) * 1000, 3),
                      state=_summary(result))
                return result
            except BaseException as exc:
                event("stage_end", stage=name, request_id=state.get("request_id"),
                      status="error", error_type=type(exc).__name__,
                      duration_ms=round((time.perf_counter() - started) * 1000, 3))
                raise
        return wrapped

    for name in ("analyze_query", "enrich_context", "determine_search_scope",
                 "validate_and_resolve", "reanalyze_query", "save_search_params",
                 "execute_search", "global_search_finalize", "select_relevant_laws",
                 "generate_analysis", "normalize_and_enrich", "fallback_chain"):
        setattr(graph, name, node_wrapper(name, getattr(graph, name)))
    builder.nodes.branch_entry = node_wrapper("branch_entry", builder.nodes.branch_entry)
    builder.nodes.assemble_response = node_wrapper("assemble_response", builder.nodes.assemble_response)

    original_filter = analysis._filter_documents

    @wraps(original_filter)
    def filtered(docs, *args, **kwargs):
        result = original_filter(docs, *args, **kwargs)
        if _selection.get() is not None:
            event("candidate_filter", request_id=_selection.get(),
                  before_count=len(docs or []), after_count=len(result or []),
                  before=_summary({"laws_list": docs}), after=_summary({"laws_list": result}))
        return result

    analysis._filter_documents = filtered

    original_call_llm = analysis.call_llm

    @wraps(original_call_llm)
    async def selection_llm(*args, **kwargs):
        result = await original_call_llm(*args, **kwargs)
        if _selection.get() is not None:
            parsed = analysis._loads_json_object_loose(result) if result else {}
            event("selection_parse", request_id=_selection.get(),
                  output_chars=len(result or ""), json_object=bool(parsed),
                  has_relevant_field="has_relevant_laws" in parsed,
                  selected_ids_field="selected_ids" in parsed)
        return result

    analysis.call_llm = selection_llm

    original_select = analysis.run_select_laws

    @wraps(original_select)
    async def select(*args, **kwargs):
        request_id = kwargs.get("request_id")
        token = _selection.set(request_id)
        started = time.perf_counter()
        event("selection_call_start", request_id=request_id,
              candidate=_summary({"laws_list": kwargs.get("laws_list") or []}),
              input_chars=len(kwargs.get("search_tool_result") or ""))
        try:
            result = await original_select(*args, **kwargs)
            event("selection_call_end", request_id=request_id, status="ok",
                  duration_ms=round((time.perf_counter() - started) * 1000, 3),
                  state=_summary(result))
            return result
        except BaseException as exc:
            event("selection_call_end", request_id=request_id, status="error",
                  error_type=type(exc).__name__,
                  duration_ms=round((time.perf_counter() - started) * 1000, 3))
            raise
        finally:
            _selection.reset(token)

    analysis.run_select_laws = select

    original_completion = client.acompletion_via_proxy

    @wraps(original_completion)
    async def completion(**kwargs):
        try:
            response = await original_completion(**kwargs)
            choices = getattr(response, "choices", None) or []
            event("model_finish", request_id=str(kwargs.get("_log_request_id") or ""),
                  role=str(kwargs.get("_log_role") or ""), model=str(kwargs.get("model") or ""),
                  finish_reason=str(getattr(choices[0], "finish_reason", "") or "") if choices else None)
            return response
        except BaseException as exc:
            event("model_finish", request_id=str(kwargs.get("_log_request_id") or ""),
                  role=str(kwargs.get("_log_role") or ""), model=str(kwargs.get("model") or ""),
                  error_type=type(exc).__name__)
            raise

    client.acompletion_via_proxy = completion
    event("diag_installed", graph_compiled=builder._moleg_graph is not None)
    if builder._moleg_graph is not None:
        raise RuntimeError("diagnostic node installation was too late")
