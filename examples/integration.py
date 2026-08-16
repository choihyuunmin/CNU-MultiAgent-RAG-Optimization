"""Provider-neutral confidence routing and typed dispatch example."""

import asyncio

from cnu_rag_optimization import (
    ConfidenceRoutingFeatures,
    OptimizationPolicy,
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
    print(result)


asyncio.run(main())
