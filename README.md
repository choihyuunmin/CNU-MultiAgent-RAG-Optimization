# CNU MOLEG RAG Optimization

Standalone research implementation of latency-reduction logic for a multi-agent legal RAG pipeline.

Repository contains only optimization primitives:

- deterministic top-k selection over ranked retrieval results;
- role-specific output-token budgets;
- comparison-context caps;
- conservative single-RAG fast-path routing;
- selector scope control for all or multi-country queries;
- parallel enrichment helper;
- prompt-free LLM timing metrics.

Repository intentionally excludes original service source, production prompts, API endpoints, credentials, legal documents, database schemas, and evaluation question text.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Minimal use

```python
from cnu_rag_optimization import OptimizationPolicy, select_ranked_documents

policy = OptimizationPolicy(selector_top_k=5)
selection = select_ranked_documents(reranked_documents, top_k=policy.selector_top_k)
```

Use `llm_options(role, policy)` when building provider requests. Use `route_query` only after an integrating application extracts provider-neutral query features such as country count and direct-lookup intent.

## Measured outcome

Same 400-question experiment, concurrency 4:

| Method | Mean latency | p50 | p95 | Pseudo-gold fidelity |
|---|---:|---:|---:|---:|
| Current control | 15.67 s | 17.20 s | 35.79 s | 0.975 self-repeat |
| Global selector + token budget | 11.27 s | 12.39 s | 24.62 s | 0.931 |
| Multi-country selector + token budget | 13.45 s | 15.72 s | 27.59 s | 0.958 |

Full aggregate methodology: [docs/EVALUATION.md](docs/EVALUATION.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and expert legal evaluation.
