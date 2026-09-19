"""Offline guards for the call-overlap attachment, using fake application modules."""
import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

from cnu_rag_optimization.call_overlap import CallOverlapAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CLASSIFY = "CLASSIFY-SYSTEM-PROMPT"
PREP_SYSTEM = [{"role": "system", "content": "PREP-SYSTEM-PROMPT"}]
PREP_FORMAT = {"type": "json_schema", "json_schema": {"name": "search_preparation"}}


def _install_fake_app(calls, gate=None):
    async def call_llm(*args, **kwargs):
        messages = args[0] if args else kwargs["messages"]
        calls.append((messages[0]["content"], messages[-1]["content"], kwargs.get("role"), kwargs.get("request_id")))
        await asyncio.sleep(0)  # a real call is sent immediately and then awaits network I/O
        if gate is not None and messages[0]["content"] == "PREP-SYSTEM-PROMPT":
            await gate.wait()
        return '{"result": "%s"}' % messages[0]["content"]

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    async def available():
        return ["A", "B"]

    module("core"); module("core.agent_orchestrator"); module("core.query_loop"); module("agent"); module("config")
    module("domain"); module("domain.country"); module("api"); module("api.controller")
    orchestrator = module("core.agent_orchestrator.orchestrator", call_llm=call_llm, _CLASSIFY_AND_DECOMPOSE_SYSTEM=CLASSIFY)
    validation = module("agent.validation_agent", call_llm=call_llm)
    module("config.prompts", get_preparation_system_prompt=lambda countries: list(PREP_SYSTEM),
           get_preparation_response_format=lambda: dict(PREP_FORMAT))
    module("config.settings", PREPARATION_LLM_ROLE="master")
    module("domain.country.service", get_available_countries_async=available)

    async def run_moleg_graph(request_id, user_prompt, *, route="law_search"):
        # The real graph: classify first, then (on the law-search route) preparation.
        text = user_prompt.strip()
        await orchestrator.call_llm(messages=[{"role": "system", "content": CLASSIFY}, {"role": "user", "content": text}],
                                    response_format={"type": "json_object"}, temperature=0, request_id=request_id,
                                    role="master", enable_thinking=False)
        if route == "law_search":
            return await validation.call_llm(messages=list(PREP_SYSTEM) + [{"role": "user", "content": text}],
                                             response_format=dict(PREP_FORMAT), temperature=0,
                                             request_id=request_id, role="master", enable_thinking=False)
        return None
    runner = module("core.query_loop.runner", run_moleg_graph=run_moleg_graph,
                    _detect_multi_country_in_query=lambda text: (["A", "B"], text) if " and " in text else None)
    controller = module("api.controller.generate_controller", run_moleg_graph=run_moleg_graph)
    return orchestrator, validation, runner, controller


def _cleanup():
    for name in list(sys.modules):
        if name.split(".")[0] in {"core", "agent", "config", "domain", "api"}:
            sys.modules.pop(name, None)
    sys.modules.pop("moleg_call_overlap", None)


def test_preparation_starts_at_classify_and_is_reused_exactly_once():
    calls, events = [], []
    try:
        gate = asyncio.Event()
        _, _, runner, controller = _install_fake_app(calls, gate)
        attachment = importlib.import_module("moleg_call_overlap")
        adapter = CallOverlapAdapter(on_event=events.append)
        record = attachment.install(adapter=adapter, emit=events.append)
        assert record["enabled"] and record["prompts_changed"] is False

        async def run():
            gate.set()
            return await controller.run_moleg_graph("req-1", " question ")
        assert asyncio.run(run()) == '{"result": "PREP-SYSTEM-PROMPT"}'
        # exactly one classify call and one preparation call, preparation issued before classify returned
        assert [c[0] for c in calls] == [CLASSIFY, "PREP-SYSTEM-PROMPT"] or [c[0] for c in calls] == ["PREP-SYSTEM-PROMPT", CLASSIFY]
        assert all(c[1] == "question" and c[3] == "req-1" for c in calls)
        reused = [e for e in events if e.get("reused") is True]
        assert len(reused) == 1 and reused[0]["reason"] == "exact_contract"
        assert adapter.pending == 0
        kinds = {e["event"] for e in events}
        assert {"classify_result", "preparation_result", "call_overlap_attachment"} <= kinds
        assert all(len(e["output_hash"]) == 64 for e in events if e["event"].endswith("_result"))
        assert controller.run_moleg_graph is runner.run_moleg_graph
    finally:
        _cleanup()


def test_different_preparation_inputs_fall_back_to_the_original_call():
    calls, events = [], []
    try:
        orchestrator, validation, _, controller = _install_fake_app(calls)
        attachment = importlib.import_module("moleg_call_overlap")
        adapter = CallOverlapAdapter(on_event=events.append)
        attachment.install(adapter=adapter, emit=events.append)

        async def run():
            text = "question"
            await orchestrator.call_llm(messages=[{"role": "system", "content": CLASSIFY}, {"role": "user", "content": text}],
                                        temperature=0, request_id="req-2", role="master")
            # the application sends a multi-turn preparation message: not what was prepared
            return await validation.call_llm(messages=list(PREP_SYSTEM) + [{"role": "user", "content": "[이전 사용자 발화]\n- (1) x\n\n[현재 발화]\n" + text}],
                                             response_format=dict(PREP_FORMAT), temperature=0, request_id="req-2", role="master",
                                             enable_thinking=False)
        assert asyncio.run(run()) == '{"result": "PREP-SYSTEM-PROMPT"}'
        outcome = next(e for e in events if e.get("reused") is False)
        assert outcome["reason"] == "contract_mismatch"
        assert [c[0] for c in calls].count("PREP-SYSTEM-PROMPT") == 2  # speculative + original
    finally:
        _cleanup()


def test_multi_turn_classify_does_not_speculate_and_other_routes_discard_on_close():
    calls, events = [], []
    try:
        orchestrator, _, _, controller = _install_fake_app(calls)
        attachment = importlib.import_module("moleg_call_overlap")
        adapter = CallOverlapAdapter(on_event=events.append)
        attachment.install(adapter=adapter, emit=events.append)

        async def run():
            await orchestrator.call_llm(messages=[{"role": "system", "content": CLASSIFY},
                                                  {"role": "user", "content": "[이전 사용자 발화]\n- (1) a\n\n[현재 발화]\nb"}],
                                        temperature=0, request_id="req-3", role="master")
            assert adapter.pending == 0
            await controller.run_moleg_graph("req-4", "greeting", route="greeting")
            assert adapter.pending == 0
        asyncio.run(run())
        dropped = [e for e in events if e.get("discarded")]
        assert len(dropped) == 1 and dropped[0]["request_key"] == "req-4" and dropped[0]["reason"] == "request_closed"
        assert [c[0] for c in calls].count("PREP-SYSTEM-PROMPT") == 1  # the wasted speculative call
    finally:
        _cleanup()


def test_trace_only_mode_changes_nothing_but_records_hashes():
    calls, events = [], []
    try:
        _, _, _, controller = _install_fake_app(calls)
        attachment = importlib.import_module("moleg_call_overlap")
        record = attachment.install(adapter=None, emit=events.append)
        assert record["enabled"] is False
        asyncio.run(controller.run_moleg_graph("req-5", "question"))
        assert [c[0] for c in calls] == [CLASSIFY, "PREP-SYSTEM-PROMPT"]
        assert not [e for e in events if e["event"] == "call_overlap"]
        assert {e["event"] for e in events} == {"call_overlap_attachment", "classify_result", "preparation_result"}
        with pytest.raises(RuntimeError, match="already attached"):
            attachment.install(adapter=None, emit=events.append)
    finally:
        _cleanup()


def test_multi_country_text_is_not_speculated_because_the_app_rewrites_it():
    calls, events = [], []
    try:
        orchestrator, _, _, _ = _install_fake_app(calls)
        attachment = importlib.import_module("moleg_call_overlap")
        adapter = CallOverlapAdapter(on_event=events.append)
        attachment.install(adapter=adapter, emit=events.append)

        async def run():
            await orchestrator.call_llm(messages=[{"role": "system", "content": CLASSIFY}, {"role": "user", "content": "law of A and B"}],
                                        temperature=0, request_id="req-6", role="master")
            assert adapter.pending == 0
        asyncio.run(run())
        assert [c[0] for c in calls] == [CLASSIFY]
        assert [e["reason"] for e in events if e.get("prepared") is False] == ["multi_country_query"]
    finally:
        _cleanup()
