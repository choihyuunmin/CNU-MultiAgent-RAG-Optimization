# CNU Multi-Agent RAG Optimization

Domain-independent latency optimization for multi-agent retrieval-augmented generation systems.

Repository contains only optimization primitives:

- persistent bounded HTTP connection pools;
- concurrent duplicate-request coalescing without completed-result caching;
- verified speculative execution with authoritative fallback;
- accuracy-gated vLLM server profiles;
- deterministic top-k selection over ranked retrieval results;
- role-specific output-token budgets;
- comparison-context caps;
- conservative single-RAG fast-path routing;
- selector scope control for all or multi-source queries;
- parallel enrichment helper;
- prompt-free LLM timing metrics.

Repository intentionally excludes original service source, production prompts, API endpoints, credentials, domain data, database schemas, and evaluation question text.

Optimization logic depends only on ranked documents, query features, pipeline roles, and source-group count. It can be integrated into legal, security, academic, enterprise, or general knowledge search systems.

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

Use `llm_options(role, policy)` when building provider requests. Use `route_query` after an integrating application extracts provider-neutral features such as source count, direct-lookup intent, and analysis intent.

## Accuracy-first rule

Speed gain is accepted only after same-query pseudo-gold regression passes. Default server gate requires 100% request success, document-ID recall, Top-1 agreement, and exact canonical response agreement. Domain-expert labels remain required to claim real correctness.

Methods that reduced latency but failed strict fidelity are retained as rejected ablations, not proposed results.

## Measured outcome

Validation used one large-domain retrieval system as a case study: 400 queries, concurrency 4, 3,678,686 indexed documents. Domain-specific fields and data are not required by this package.

| Method | Mean latency | p50 | p95 | Pseudo-gold fidelity | Decision |
|---|---:|---:|---:|---:|---|
| Current control | 15.67 s | 17.20 s | 35.79 s | 0.975 self-repeat | Reference |
| Global selector + token budget | 11.27 s | 12.39 s | 24.62 s | 0.931 | Rejected |
| Multi-source selector + token budget | 13.45 s | 15.72 s | 27.59 s | 0.958 | Rejected |

Transport and server optimizations keep request payloads and target model unchanged. Candidate profiles: prefix caching, chunked-prefill scheduling, and target-verified n-gram decoding.

## One-line method examples

| Method | Example |
|---|---|
| HTTP keep-alive | Reuse one bounded client for LLM, embedding, and reranker calls. |
| Single-flight | Two simultaneous identical questions share one in-flight request; no answer remains cached. |
| Verified speculation | Generate early, reuse only when authoritative document IDs match; otherwise run baseline generation. |
| Parallel enrichment | Fetch metadata while final answer generates, then merge unchanged results. |
| Prefix cache | Reuse static agent-system-prompt KV blocks across requests. |
| Chunked prefill | Tune prefill token budget while preserving model and prompts. |
| Speculative decoding | Draft tokens via n-gram and let target model verify every accepted token. |

Full aggregate methodology: [docs/EVALUATION.md](docs/EVALUATION.md).

Server-side accuracy gate and deployment matrix: [docs/ACCURACY_FIRST_SERVER_EXPERIMENTS.md](docs/ACCURACY_FIRST_SERVER_EXPERIMENTS.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules or a particular domain. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and domain-expert evaluation.
