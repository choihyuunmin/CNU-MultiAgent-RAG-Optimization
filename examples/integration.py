"""Provider-neutral integration example."""

from cnu_rag_optimization import (
    OptimizationPolicy,
    QueryFeatures,
    llm_options,
    route_query,
    select_ranked_documents,
    should_use_selector,
)


policy = OptimizationPolicy()
features = QueryFeatures(
    prompt_chars=42,
    source_count=2,
    direct_lookup=False,
    analysis_requested=True,
)
decision = route_query(features, policy)

ranked_documents = [
    {"id": "document-1", "title": "Rank 1", "score": 0.91},
    {"id": "document-2", "title": "Rank 2", "score": 0.87},
]

if should_use_selector(features.source_count, policy):
    selection = select_ranked_documents(
        ranked_documents,
        top_k=policy.selector_top_k,
    )
    selected_ids = selection.selected_ids
else:
    selected_ids = ()  # Integrator may invoke its existing LLM selector here.

comparison_options = llm_options("comparison", policy)

print(decision)
print(selected_ids)
print(comparison_options)
