"""Provider-neutral confidence routing and typed dispatch example."""

import asyncio

from cnu_rag_optimization import (
    ConfidenceRoutingFeatures,
    OptimizationPolicy,
    decide_evidence_route,
    decide_confident_route,
    try_typed_single_tool_dispatch,
)


policy = OptimizationPolicy()
decision = decide_confident_route(
    ConfidenceRoutingFeatures(
        candidate_route="retrieval",
        candidate_count=1,
        entity_count=1,
        domain_term_matched=True,
        action_term_matched=True,
    ),
    policy,
)

evidence_decision = decide_evidence_route(
    [
        {
            "id": "record-1_section-1",
            "parent_id": "record-1",
            "scope": "source-a",
            "score": 0.91,
            "content": "entity retention requirements",
        }
    ],
    expected_scope="source-a",
    query_terms=["retention", "requirements"],
)


async def search(arguments: dict[str, object]) -> dict[str, object]:
    return {"query": arguments["query"], "document_ids": ["document-1"]}


async def main() -> None:
    if not decision.use_local_route:
        print("fallback to existing LLM router")
        return
    result = await try_typed_single_tool_dispatch(
        available_tools={"search": search},
        candidate_tool="search",
        arguments={"query": "find entity records"},
        argument_validator=lambda args: bool(args.get("query")),
        policy=policy,
    )
    print(decision)
    print(evidence_decision)
    print(result)


asyncio.run(main())
