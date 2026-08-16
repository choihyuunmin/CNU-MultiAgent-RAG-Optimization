import asyncio

import pytest

from cnu_rag_optimization import (
    OptimizationPolicy,
    QueryFeatures,
    SelectorScope,
    cap_comparison_documents,
    compact_documents,
    llm_options,
    parallel_enrich,
    route_query,
    select_ranked_documents,
    should_use_selector,
)


def test_selector_preserves_rank_and_deduplicates() -> None:
    result = select_ranked_documents(
        [{"id": "a"}, {"id": "a"}, {"law_id": "b"}, {"id": "c"}],
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
    policy = OptimizationPolicy()
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
        QueryFeatures(prompt_chars=40, country_count=1, direct_lookup=True),
        OptimizationPolicy(),
    )
    assert decision.complexity == "L1"
    assert decision.execution_path == "single_rag"


def test_analysis_stays_on_multi_agent_path() -> None:
    decision = route_query(
        QueryFeatures(
            prompt_chars=80,
            country_count=2,
            direct_lookup=True,
            analysis_requested=True,
        ),
        OptimizationPolicy(),
    )
    assert decision.complexity == "L3"
    assert decision.execution_path == "multi_agent"
    assert "analysis_requested" in decision.blockers


def test_multi_country_selector_scope() -> None:
    policy = OptimizationPolicy(selector_scope=SelectorScope.MULTI_COUNTRY)
    assert not should_use_selector(1, policy)
    assert should_use_selector(2, policy)


def test_parallel_enrichment() -> None:
    async def value(result: str) -> str:
        await asyncio.sleep(0)
        return result

    assert asyncio.run(parallel_enrich(value("translation"), value("origin"))) == (
        "translation",
        "origin",
    )
