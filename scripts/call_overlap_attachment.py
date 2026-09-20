from __future__ import annotations

import hashlib
import importlib
import json
import os
from functools import wraps

from cnu_rag_optimization.call_overlap import CallContract

PREPARATION = "search_preparation"
GRAPH_ENTRY = os.environ.get("CNU_GRAPH_ENTRY", "run_graph")
_PRIOR_BLOCK = "[이전 사용자 발화]"


def _digest(text):
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    except Exception:
        return None


def _messages_of(args, kwargs):
    return args[0] if args else kwargs.get("messages")


def install(*, adapter=None, emit=lambda event: None, graph_entry=None):
    """Wrap the orchestrator and preparation call sites in place.

    adapter: a CallOverlapAdapter, or None for tracing only.
    emit: metadata sink (hashes, ids and timings only).
    graph_entry: name of the request-level graph entry function on the runner and
    controller modules (default: CNU_GRAPH_ENTRY environment variable or "run_graph").
    Returns a record describing the attachment.
    """
    orchestrator = importlib.import_module("core.agent_orchestrator.orchestrator")
    validation = importlib.import_module("agent.validation_agent")
    prompts = importlib.import_module("config.prompts")
    settings = importlib.import_module("config.settings")
    countries = importlib.import_module("domain.country.service")
    runner = importlib.import_module("core.query_loop.runner")
    controller = importlib.import_module("api.controller.generate_controller")

    if getattr(orchestrator.call_llm, "__cnu_call_overlap__", False):
        raise RuntimeError("call overlap already attached")
    classify_system = orchestrator._CLASSIFY_AND_DECOMPOSE_SYSTEM
    original_orchestrator_call = orchestrator.call_llm
    original_preparation_call = validation.call_llm
    graph_entry = graph_entry or GRAPH_ENTRY
    original_graph = getattr(runner, graph_entry)

    def contract_audit(phase, messages, kwargs, request_id):
        """Component hashes only (no content), so a mismatch can be attributed to one input."""
        try:
            system = [m.get("content") for m in messages if m.get("role") == "system"]
            user = [m.get("content") for m in messages if m.get("role") == "user"]
            emit({"event": "call_overlap_contract", "phase": phase, "request_id": str(request_id),
                  "n_messages": len(messages), "system_hash": _digest(json.dumps(system, ensure_ascii=False)),
                  "user_hash": _digest(json.dumps(user, ensure_ascii=False)),
                  "user_chars": [len(u or "") for u in user], "user_has_prior_block": any(_PRIOR_BLOCK in (u or "") for u in user),
                  "user_stripped_equal": all((u or "") == (u or "").strip() for u in user),
                  "format_hash": _digest(json.dumps(kwargs.get("response_format"), sort_keys=True, ensure_ascii=False)),
                  "role": kwargs.get("role", "master"), "temperature": kwargs.get("temperature", 0),
                  "enable_thinking": kwargs.get("enable_thinking")})
        except Exception:
            pass

    def preparation_contract(messages, kwargs, request_id):
        return CallContract.build(
            PREPARATION, "agent.validation_agent.call_llm", str(request_id),
            {"messages": messages, "response_format": kwargs.get("response_format"),
             "temperature": kwargs.get("temperature", 0), "role": kwargs.get("role", "master"),
             "enable_thinking": kwargs.get("enable_thinking")},
            deterministic=kwargs.get("temperature", 0) == 0)

    def is_single_turn_classify(messages, kwargs):
        return (kwargs.get("role", "master") == "master" and isinstance(messages, list) and len(messages) == 2
                and messages[0].get("role") == "system" and messages[0].get("content") == classify_system
                and messages[1].get("role") == "user" and isinstance(messages[1].get("content"), str)
                and _PRIOR_BLOCK not in messages[1]["content"])

    def splits_into_sub_tasks(text):
        """The application rewrites the prompt per sub-task when the text names several
        countries; the preparation input then depends on the classification and cannot be
        prepared early. Uses the application's own rule-based detector (no model call)."""
        detector = getattr(runner, "_detect_multi_country_in_query", None)
        if detector is None:
            return False
        try:
            return bool(detector(text))
        except Exception:
            return False

    @wraps(original_orchestrator_call)
    async def orchestrator_call(*args, **kwargs):
        messages = _messages_of(args, kwargs)
        request_id = kwargs.get("request_id", "-")
        if adapter is not None and is_single_turn_classify(messages, kwargs) and splits_into_sub_tasks(messages[1]["content"]):
            emit({"event": "call_overlap", "operation": PREPARATION, "request_key": str(request_id),
                  "prepared": False, "reason": "multi_country_query"})
        elif adapter is not None and is_single_turn_classify(messages, kwargs):
            try:
                available = await countries.get_available_countries_async()
                prep_messages = prompts.get_preparation_system_prompt(available) + [
                    {"role": "user", "content": messages[1]["content"]}]
                prep_kwargs = {"response_format": prompts.get_preparation_response_format(),
                               "temperature": 0, "request_id": request_id,
                               "role": settings.PREPARATION_LLM_ROLE, "enable_thinking": False}
                contract = preparation_contract(prep_messages, prep_kwargs, request_id)
                if adapter.prepare(contract, lambda: original_preparation_call(messages=prep_messages, **prep_kwargs)):
                    contract_audit("prepared", prep_messages, prep_kwargs, request_id)
            except Exception:
                emit({"event": "call_overlap", "operation": PREPARATION, "request_key": str(request_id),
                      "prepared": False, "reason": "prepare_error"})
        text = await original_orchestrator_call(*args, **kwargs)
        if isinstance(messages, list) and messages and messages[0].get("content") == classify_system:
            emit({"event": "classify_result", "request_id": request_id, "output_hash": _digest(text)})
        return text

    @wraps(original_preparation_call)
    async def preparation_call(*args, **kwargs):
        messages = _messages_of(args, kwargs)
        request_id = kwargs.get("request_id", "-")
        if adapter is None:
            text = await original_preparation_call(*args, **kwargs)
        else:
            contract = preparation_contract(messages, kwargs, request_id)
            contract_audit("actual", messages, kwargs, request_id)

            async def fallback():
                return await original_preparation_call(*args, **kwargs)
            text = await adapter.consume(contract, fallback)
        emit({"event": "preparation_result", "request_id": request_id, "output_hash": _digest(text)})
        return text

    @wraps(original_graph)
    async def graph_wrapper(*args, **kwargs):
        try:
            return await original_graph(*args, **kwargs)
        finally:
            if adapter is not None:
                request_id = kwargs.get("request_id", args[0] if args else None)
                if request_id is not None:
                    try:
                        await adapter.close(str(request_id))
                    except Exception:
                        pass

    orchestrator_call.__cnu_call_overlap__ = True
    orchestrator.call_llm = orchestrator_call
    validation.call_llm = preparation_call
    setattr(runner, graph_entry, graph_wrapper)
    if getattr(controller, graph_entry, None) is original_graph:
        setattr(controller, graph_entry, graph_wrapper)
    record = {"enabled": adapter is not None, "operation": PREPARATION,
              "trigger": "single-turn classify call", "prompts_changed": False,
              "model_options_changed": False, "fallback": "original call on any contract difference"}
    emit({"event": "call_overlap_attachment", **record})
    return record
