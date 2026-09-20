import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from cnu_rag_optimization.selection_budget import SelectionBudget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _install_fake_app(seen):
    def module(name, **attrs):
        m = types.ModuleType(name); m.__path__ = []; m.__dict__.update(attrs); sys.modules[name] = m; return m

    def _cap_search_tool_result(text, *, request_id="-", **kwargs):
        seen.append(("cap", len(text)))
        return text[:5000]  # the application's own safety cap

    async def run_select_laws(*, request_id, stage2_user_text, search_tool_result, laws_list):
        capped = agent._cap_search_tool_result(search_tool_result, request_id=request_id)
        seen.append(("select_input", capped))
        ids = [d["id"] for d in json.loads(capped)["laws"][:2]]
        return {"selected_ids": ids, "has_relevant_laws": bool(ids), "search_law_ids": ids}

    module("agent"); module("core"); module("core.query_loop"); module("core.query_loop.law_search")
    agent = module("agent.law_analysis_agent", _cap_search_tool_result=_cap_search_tool_result, run_select_laws=run_select_laws)
    module("core.query_loop.law_search.nodes", law_analysis_agent=agent)
    return agent


def _cleanup():
    for name in list(sys.modules):
        if name.split(".")[0] in {"agent", "core"}:
            sys.modules.pop(name, None)
    sys.modules.pop("selection_budget_attachment", None)


def payload(n=3):
    return json.dumps({"laws": [{"source": "s", "id": f"{i}", "title": "t", "content": "c" * 500, "score": 1.0} for i in range(n)]})


def test_budget_applies_after_the_app_cap_and_keeps_ids():
    seen, events = [], []
    try:
        agent = _install_fake_app(seen)
        attachment = importlib.import_module("selection_budget_attachment")
        record = attachment.install(budget=SelectionBudget(text_max_chars=50), emit=events.append)
        assert record["enabled"] and record["candidate_set_changed"] is False
        result = asyncio.run(agent.run_select_laws(request_id="r1", stage2_user_text="q", search_tool_result=payload(), laws_list=[]))
        assert result["selected_ids"] == ["0", "1"]
        assert seen[0][0] == "cap"  # application cap ran first
        compact = json.loads(next(v for k, v in seen if k == "select_input"))
        assert [d["id"] for d in compact["laws"]] == ["0", "1", "2"] and set(compact["laws"][0]) == {"id", "title", "content"}
        assert all(len(d["content"]) == 51 for d in compact["laws"])
        inp = next(e for e in events if e["event"] == "selection_input")
        assert inp["applied"] and inp["dropped_fields"] == ["score", "source"] and inp["chars_after"] < inp["chars_before"]
        sel = next(e for e in events if e["event"] == "selection_result")
        assert sel["selected_count"] == 2 and len(sel["selected_hash"]) == 64 and sel["request_id"] == "r1"
        with pytest.raises(RuntimeError, match="already attached"):
            attachment.install(budget=None, emit=events.append)
    finally:
        _cleanup()


def test_trace_only_mode_leaves_the_input_unchanged():
    seen, events = [], []
    try:
        agent = _install_fake_app(seen)
        attachment = importlib.import_module("selection_budget_attachment")
        record = attachment.install(budget=None, emit=events.append)
        assert record["enabled"] is False
        asyncio.run(agent.run_select_laws(request_id="r2", stage2_user_text="q", search_tool_result=payload(), laws_list=[]))
        assert next(v for k, v in seen if k == "select_input") == payload()[:5000]
        inp = next(e for e in events if e["event"] == "selection_input")
        assert inp["applied"] is False and inp["chars"] == len(payload()[:5000])
    finally:
        _cleanup()
