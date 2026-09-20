from __future__ import annotations

import hashlib
import importlib
import json
from functools import wraps

from cnu_rag_optimization.selection_budget import budget_selection_input


def _digest(value):
    try:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        return None


def install(*, budget=None, emit=lambda event: None):
    agent = importlib.import_module("agent.law_analysis_agent")
    if getattr(agent._cap_search_tool_result, "__cnu_selection_budget__", False):
        raise RuntimeError("selection budget already attached")
    original_cap = agent._cap_search_tool_result
    original_select = agent.run_select_laws

    @wraps(original_cap)
    def capped(search_tool_result, *args, **kwargs):
        text = original_cap(search_tool_result, *args, **kwargs)
        request_id = kwargs.get("request_id", "-")
        if budget is None:
            emit({"event": "selection_input", "request_id": str(request_id), "applied": False,
                  "chars": len(text or ""), "input_hash": hashlib.sha256((text or "").encode("utf-8")).hexdigest()})
            return text
        out, record = budget_selection_input(text, budget)
        emit({"event": "selection_input", "request_id": str(request_id), **record,
              "input_hash": hashlib.sha256((out or "").encode("utf-8")).hexdigest()})
        return out

    @wraps(original_select)
    async def select(*args, **kwargs):
        result = await original_select(*args, **kwargs)
        try:
            ids = sorted(str(x) for x in (result.get("selected_ids") or []))
            emit({"event": "selection_result", "request_id": str(kwargs.get("request_id", "-")),
                  "has_relevant_laws": bool(result.get("has_relevant_laws")), "selected_count": len(ids),
                  "selected_hash": _digest(ids)})
        except Exception:
            pass
        return result

    capped.__cnu_selection_budget__ = True
    agent._cap_search_tool_result = capped
    agent.run_select_laws = select
    nodes = importlib.import_module("core.query_loop.law_search.nodes")
    if getattr(nodes, "law_analysis_agent", None) is agent:
        pass  # the node calls agent.run_select_laws through the module attribute; already rebound
    record = {"enabled": budget is not None, "stage": "law_selection",
              "keep_fields": list(budget.keep_fields) if budget else None,
              "text_max_chars": budget.text_max_chars if budget else None,
              "max_documents": budget.max_documents if budget else None,
              "candidate_set_changed": bool(budget and budget.max_documents is not None),
              "prompts_changed": False, "model_options_changed": False}
    emit({"event": "selection_budget_attachment", **record})
    return record
