import asyncio

import pytest

from cnu_rag_optimization import (
    AsyncSingleFlight,
    ConfidenceRoutingFeatures,
    HTTPTransportPolicy,
    OptimizationPolicy,
    QueryFeatures,
    SelectorScope,
    SpeculativeDraft,
    build_async_client,
    cap_comparison_documents,
    compare_regression_records,
    compact_documents,
    decide_confident_route,
    llm_options,
    parallel_enrich,
    route_query,
    run_verified_speculation,
    select_ranked_documents,
    should_use_selector,
    try_typed_single_tool_dispatch,
)


def test_selector_preserves_rank_and_deduplicates() -> None:
    result = select_ranked_documents(
        [{"id": "a"}, {"id": "a"}, {"document_id": "b"}, {"id": "c"}],
        top_k=2,
    )
    assert result.selected_ids == ("a", "b")
    assert result.has_relevant_documents


def test_selector_applies_integrator_validator() -> None:
    result = select_ranked_documents(
        [{"id": "unsafe", "ok": False}, {"id": "safe", "ok": True}],
        validator=lambda row: bool(row.get("ok")),
    )
    assert result.selected_ids == ("safe",)


def test_selector_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        select_ranked_documents([], top_k=0)


def test_token_budgets_are_role_specific() -> None:
    policy = OptimizationPolicy(token_budget_enabled=True)
    assert llm_options("preparation", policy) == {"max_tokens": 384}
    assert llm_options("comparison", policy) == {"max_tokens": 2048}
    assert llm_options("unknown", policy) == {}


def test_disabled_token_budget_preserves_existing_provider_defaults() -> None:
    policy = OptimizationPolicy(token_budget_enabled=False)
    assert llm_options("comparison", policy) == {}


def test_comparison_context_is_capped_without_mutating_source() -> None:
    source = {"A": [{"id": "1", "content": "x" * 1000}]}
    capped = cap_comparison_documents(source, max_content_chars=20)
    assert len(capped["A"][0]["content"]) == 21
    assert len(source["A"][0]["content"]) == 1000


def test_document_compaction_caps_count_and_fields() -> None:
    docs = [{"id": str(i), "content": "x" * 100, "unused": "secret"} for i in range(5)]
    compact = compact_documents(docs, max_documents=2, field_max_chars=10)
    assert len(compact) == 2
    assert compact[0]["content"] == "x" * 10 + "…"
    assert "unused" not in compact[0]


def test_direct_lookup_uses_fast_path() -> None:
    decision = route_query(
        QueryFeatures(prompt_chars=40, source_count=1, direct_lookup=True),
        OptimizationPolicy(fast_path_enabled=True),
    )
    assert decision.complexity == "simple_lookup"
    assert decision.execution_path == "single_rag"


def test_analysis_stays_on_multi_agent_path() -> None:
    decision = route_query(
        QueryFeatures(
            prompt_chars=80,
            source_count=2,
            direct_lookup=True,
            analysis_requested=True,
        ),
        OptimizationPolicy(),
    )
    assert decision.complexity == "multi_source"
    assert decision.execution_path == "multi_agent"
    assert "analysis_requested" in decision.blockers


def test_multi_source_selector_scope() -> None:
    policy = OptimizationPolicy(
        selector_enabled=True,
        selector_scope=SelectorScope.MULTI_SOURCE,
    )
    assert not should_use_selector(1, policy)
    assert should_use_selector(2, policy)


def test_parallel_enrichment() -> None:
    async def value(result: str) -> str:
        await asyncio.sleep(0)
        return result

    assert asyncio.run(parallel_enrich(value("primary"), value("secondary"))) == (
        "primary",
        "secondary",
    )


def test_http_client_uses_bounded_pool() -> None:
    async def check() -> None:
        client = build_async_client(
            HTTPTransportPolicy(
                max_connections=8,
                max_keepalive_connections=4,
            )
        )
        try:
            assert client.timeout.connect == 5.0
        finally:
            await client.aclose()

    asyncio.run(check())


def test_verified_speculation_reuses_only_matching_draft() -> None:
    fallback_calls: list[tuple[str, ...]] = []

    async def selection() -> list[str]:
        return ["a", "b"]

    async def draft(ids: tuple[str, ...], value: str) -> SpeculativeDraft[str]:
        return SpeculativeDraft(value=value, selected_ids=ids)

    async def fallback(ids: tuple[str, ...]) -> str:
        fallback_calls.append(ids)
        return "baseline"

    hit = asyncio.run(
        run_verified_speculation(
            selection(),
            draft(("a", "b"), "draft"),
            fallback,
        )
    )
    assert hit.value == "draft"
    assert hit.reused_draft
    assert fallback_calls == []

    miss = asyncio.run(
        run_verified_speculation(
            selection(),
            draft(("a", "c"), "unsafe"),
            fallback,
        )
    )
    assert miss.value == "baseline"
    assert not miss.reused_draft
    assert fallback_calls == [("a", "b")]


def test_verified_speculation_can_compare_id_sets() -> None:
    async def fallback(_: tuple[str, ...]) -> str:
        return "baseline"

    result = asyncio.run(
        run_verified_speculation(
            _async_value(["a", "b"]),
            _async_value(SpeculativeDraft(value="draft", selected_ids=("b", "a"))),
            fallback,
            order_sensitive=False,
        )
    )
    assert result.reused_draft


async def _async_value(value):
    await asyncio.sleep(0)
    return value


def test_singleflight_coalesces_only_concurrent_requests() -> None:
    calls = 0

    async def exercise() -> tuple[list[str], list[bool], int]:
        nonlocal calls
        gate = asyncio.Event()
        flight: AsyncSingleFlight[str, str] = AsyncSingleFlight()

        async def factory() -> str:
            nonlocal calls
            calls += 1
            await gate.wait()
            return "same-output"

        first = asyncio.create_task(flight.do("same-key", factory))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.do("same-key", factory))
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(first, second)

        gate.set()
        third = await flight.do("same-key", factory)
        return (
            [item.value for item in (*results, third)],
            [item.shared for item in (*results, third)],
            calls,
        )

    values, shared, call_count = asyncio.run(exercise())
    assert values == ["same-output"] * 3
    assert shared == [False, True, False]
    assert call_count == 2


def test_regression_metrics_fail_closed_on_changed_answer() -> None:
    control = [
        {"status": "ok", "selected_ids": ["a", "b"], "response": {"x": 1}},
        {"status": "ok", "selected_ids": ["c"], "response": {"x": 2}},
    ]
    candidate = [
        {"status": "ok", "selected_ids": ["a", "b"], "response": {"x": 1}},
        {"status": "ok", "selected_ids": ["c"], "response": {"x": 3}},
    ]
    metrics = compare_regression_records(control, candidate)
    assert metrics.success_rate == 1.0
    assert metrics.id_recall == 1.0
    assert metrics.top1_agreement == 1.0
    assert metrics.exact_response_rate == 0.5


def test_accuracy_first_defaults_disable_rejected_ablations() -> None:
    policy = OptimizationPolicy()
    assert not policy.selector_enabled
    assert not policy.token_budget_enabled
    assert not policy.fast_path_enabled
    assert policy.confidence_routing_enabled
    assert policy.typed_dispatch_enabled


def test_confidence_router_accepts_only_complete_evidence() -> None:
    decision = decide_confident_route(
        ConfidenceRoutingFeatures(
            candidate_route="retrieval",
            candidate_count=1,
            entity_count=2,
            domain_term_matched=True,
            action_term_matched=True,
        )
    )
    assert decision.use_local_route
    assert decision.route == "retrieval"


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"candidate_count": 2}, "route_not_unique"),
        ({"entity_count": 0}, "entity_missing"),
        ({"domain_term_matched": False}, "domain_term_missing"),
        ({"action_term_matched": False}, "action_term_missing"),
        ({"excluded_intent": True}, "excluded_intent"),
        ({"ambiguous": True}, "ambiguous"),
        ({"has_prior_context": True}, "context_required"),
    ],
)
def test_confidence_router_falls_back_on_uncertainty(override, reason) -> None:
    values = {
        "candidate_route": "retrieval",
        "candidate_count": 1,
        "entity_count": 1,
        "domain_term_matched": True,
        "action_term_matched": True,
    }
    values.update(override)
    decision = decide_confident_route(ConfidenceRoutingFeatures(**values))
    assert not decision.use_local_route
    assert decision.route is None
    assert decision.reason == reason


def test_typed_dispatch_calls_only_valid_unique_tool() -> None:
    calls: list[dict[str, object]] = []

    async def search(arguments: dict[str, object]) -> dict[str, object]:
        calls.append(arguments)
        return {"ids": ["document-1"]}

    result = asyncio.run(
        try_typed_single_tool_dispatch(
            available_tools={"search": search},
            candidate_tool="search",
            arguments={"query": "explicit lookup"},
            argument_validator=lambda args: bool(args.get("query")),
        )
    )
    assert result.dispatched
    assert result.value == {"ids": ["document-1"]}
    assert calls == [{"query": "explicit lookup"}]


def test_typed_dispatch_falls_back_without_calling_tool() -> None:
    calls = 0

    async def tool(_: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        return "unexpected"

    result = asyncio.run(
        try_typed_single_tool_dispatch(
            available_tools={"search": tool, "stats": tool},
            candidate_tool="search",
            arguments={"query": "x"},
        )
    )
    assert not result.dispatched
    assert result.reason == "tool_not_unique"
    assert calls == 0
