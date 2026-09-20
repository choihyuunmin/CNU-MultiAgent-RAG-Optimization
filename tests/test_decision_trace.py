import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _install_fake_app():
    def module(name, **attrs):
        m = types.ModuleType(name); m.__path__ = []; m.__dict__.update(attrs); sys.modules[name] = m; return m

    class Classification:
        def __init__(self, task_type, confidence): self.task_type, self.confidence = task_type, confidence

    class SubTask:
        def __init__(self, task_type, query): self.task_type, self.query = task_type, query

    class Result:
        def __init__(self, task, sub_tasks=None, countries=()):
            self.classification = Classification(task, "HIGH"); self.sub_tasks = sub_tasks; self.mentioned_countries = list(countries)

    async def classify_and_decompose(user_prompt, history=None, *, session_context=None, request_id="-"):
        return Result("composite", [SubTask("law_search", "q1"), SubTask("assistant", "q2")], ["A"]) if "and" in user_prompt else Result("law_search", None, ["A"])

    async def run_preparation(*, history, request_id, precomputed_text=None):
        return {"country": "A", "transformed_query": "t", "keywords_from_original": ["k"], "keywords_from_transformed": [], "bad_word_detected": False,
                "pii_detected": False, "specific_article_number": 0, "law_title_search_hint": "", "extra": "ignored"}

    async def run_select_laws(*, request_id, stage2_user_text, search_tool_result, laws_list):
        return {"selected_ids": ["b", "a"], "has_relevant_laws": True, "search_law_ids": ["b", "a"]}

    module("core"); module("core.agent_orchestrator"); module("core.query_loop"); module("agent")
    orchestrator = module("core.agent_orchestrator.orchestrator", classify_and_decompose=classify_and_decompose)
    runner = module("core.query_loop.runner", classify_and_decompose=classify_and_decompose)
    validation = module("agent.validation_agent", run_preparation=run_preparation)
    analysis = module("agent.law_analysis_agent", run_select_laws=run_select_laws)
    return orchestrator, runner, validation, analysis


def _cleanup():
    for name in list(sys.modules):
        if name.split(".")[0] in {"core", "agent"}:
            sys.modules.pop(name, None)
    sys.modules.pop("decision_trace_attachment", None)


def test_trace_emits_hashes_without_changing_results_and_rebinds_runner_reference():
    events = []
    try:
        orchestrator, runner, validation, analysis = _install_fake_app()
        attachment = importlib.import_module("decision_trace_attachment")
        record = attachment.install(emit=events.append)
        assert record["records_text"] is False and runner.classify_and_decompose is orchestrator.classify_and_decompose

        async def run():
            a = await runner.classify_and_decompose("law of A and B", request_id="r1")
            b = await orchestrator.classify_and_decompose("law of A", request_id="r2")
            p = await validation.run_preparation(history=[], request_id="r1")
            s = await analysis.run_select_laws(request_id="r1", stage2_user_text="q", search_tool_result="{}", laws_list=[])
            return a, b, p, s
        a, b, p, s = asyncio.run(run())
        assert a.classification.task_type == "composite" and b.classification.task_type == "law_search"
        assert p["country"] == "A" and s["selected_ids"] == ["b", "a"]  # results untouched
        kinds = [e["event"] for e in events]
        assert kinds.count("decision_classify") == 2 and kinds.count("decision_prepare") == 1 and kinds.count("decision_select") == 1
        c1, c2 = [e for e in events if e["event"] == "decision_classify"]
        assert c1["sub_tasks"] == 2 and c2["sub_tasks"] == 0 and c1["decision_hash"] != c2["decision_hash"]
        sel = next(e for e in events if e["event"] == "decision_select")
        assert sel["selected_count"] == 2 and len(sel["decision_hash"]) == 64
        for e in events:
            assert "q1" not in str(e) and "law of" not in str(e)  # no text recorded
        with pytest.raises(RuntimeError, match="already attached"):
            attachment.install(emit=events.append)
    finally:
        _cleanup()


def test_same_decisions_hash_equal_regardless_of_ignored_fields_and_id_order():
    events = []
    try:
        _, _, validation, analysis = _install_fake_app()
        attachment = importlib.import_module("decision_trace_attachment")
        attachment.install(emit=events.append)

        async def run():
            await validation.run_preparation(history=[], request_id="a")
            await validation.run_preparation(history=[{"role": "user", "content": "x"}], request_id="b")
            await analysis.run_select_laws(request_id="a", stage2_user_text="", search_tool_result="", laws_list=[])
        asyncio.run(run())
        prep = [e["decision_hash"] for e in events if e["event"] == "decision_prepare"]
        assert prep[0] == prep[1]
    finally:
        _cleanup()
