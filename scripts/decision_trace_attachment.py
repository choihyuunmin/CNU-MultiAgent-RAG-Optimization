from __future__ import annotations

import hashlib
import importlib
import json
from functools import wraps

_PREP_KEYS = ("bad_word_detected", "pii_detected", "transformed_query", "keywords_from_original",
              "keywords_from_transformed", "country", "specific_article_number", "law_title_search_hint")


def _digest(value):
    try:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    except Exception:
        return None


def install(*, emit=lambda event: None):
    orchestrator = importlib.import_module("core.agent_orchestrator.orchestrator")
    validation = importlib.import_module("agent.validation_agent")
    analysis = importlib.import_module("agent.law_analysis_agent")
    if getattr(orchestrator.classify_and_decompose, "__cnu_decision_trace__", False):
        raise RuntimeError("decision trace already attached")
    original_classify, original_prepare, original_select = orchestrator.classify_and_decompose, validation.run_preparation, analysis.run_select_laws

    @wraps(original_classify)
    async def classify(*args, **kwargs):
        result = await original_classify(*args, **kwargs)
        try:
            c = result.classification
            emit({"event": "decision_classify", "request_id": str(kwargs.get("request_id", "-")),
                  "task": getattr(c, "task_type", None), "confidence": getattr(c, "confidence", None),
                  "sub_tasks": len(result.sub_tasks or []) if getattr(result, "sub_tasks", None) else 0,
                  "decision_hash": _digest({"task": getattr(c, "task_type", None), "confidence": getattr(c, "confidence", None),
                                            "countries": list(getattr(result, "mentioned_countries", []) or []),
                                            "sub_tasks": [(t.task_type, t.query) for t in (result.sub_tasks or [])]})})
        except Exception:
            pass
        return result

    @wraps(original_prepare)
    async def prepare(*args, **kwargs):
        result = await original_prepare(*args, **kwargs)
        try:
            subset = {k: result.get(k) for k in _PREP_KEYS if isinstance(result, dict)}
            emit({"event": "decision_prepare", "request_id": str(kwargs.get("request_id", "-")),
                  "early_exit": bool(isinstance(result, dict) and result.get("early_exit")),
                  "decision_hash": _digest(subset)})
        except Exception:
            pass
        return result

    @wraps(original_select)
    async def select(*args, **kwargs):
        result = await original_select(*args, **kwargs)
        try:
            ids = sorted(str(x) for x in (result.get("selected_ids") or []))
            emit({"event": "decision_select", "request_id": str(kwargs.get("request_id", "-")),
                  "has_relevant_laws": bool(result.get("has_relevant_laws")), "selected_count": len(ids),
                  "decision_hash": _digest(ids)})
        except Exception:
            pass
        return result

    classify.__cnu_decision_trace__ = True
    orchestrator.classify_and_decompose = classify
    validation.run_preparation = prepare
    analysis.run_select_laws = select
    # the runner imported classify_and_decompose by name; rebind that reference too
    runner = importlib.import_module("core.query_loop.runner")
    if getattr(runner, "classify_and_decompose", None) is original_classify:
        runner.classify_and_decompose = classify
    record = {"enabled": True, "decisions": ["classify", "prepare", "select"], "records_text": False}
    emit({"event": "decision_trace_attachment", **record})
    return record
