# CNU Multi-Agent RAG Optimization

Domain-independent latency optimization for multi-agent retrieval-augmented generation systems.

Repository contains only optimization primitives:

- confidence-gated local routing with LLM fallback;
- typed single-tool dispatch with argument validation;
- persistent bounded HTTP connection pools;
- concurrent duplicate-request coalescing without completed-result caching;
- verified speculative execution with authoritative fallback;
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

See [examples/integration.py](examples/integration.py). Integrating application supplies domain-specific entity and intent matches; library applies conservative gates and falls back without executing a tool when evidence is incomplete.

## Accuracy-first rule

Speed gain is accepted only after same-query pseudo-gold regression passes. Default application gate requires 100% request success plus predefined document-rank and answer-similarity thresholds. Domain-expert labels remain required to claim real correctness.

Methods that reduced latency but failed strict fidelity are retained as rejected ablations, not proposed results.

## Measured outcome

Validation used one large-domain retrieval system as a case study: 400 queries, concurrency 4, 3,678,686 indexed documents. Domain-specific fields and data are not required by this package.

| Method | Mean latency | p50 | p95 | Pseudo-gold fidelity | Decision |
|---|---:|---:|---:|---:|---|
| Seeded control | 16.53 s | 16.15 s | 37.78 s | Reference | Control |
| Confidence route + typed dispatch | 14.85 s | 15.45 s | 35.62 s | 0.9763 | Accepted |

Remote inference engine remained unchanged. Candidate reduced LLM calls from 2,007 to 1,375 (-31.5%), mean latency 10.1%, p95 5.7%, and p99 16.2%. Document-ID Recall was 0.9804, nDCG@10 0.9778, Top-1 agreement 0.9650, and answer similarity 0.9756.

## One-line method examples

| Method | Example |
|---|---|
| Confidence routing | Route only when one intent plus explicit entity, domain, and action evidence all match; otherwise use existing LLM router. |
| Typed dispatch | Call sole selected search tool with validated structured arguments instead of asking LLM to echo them. |
| HTTP keep-alive | Reuse one bounded client for LLM, embedding, and reranker calls. |
| Single-flight | Two simultaneous identical questions share one in-flight request; no answer remains cached. |
| Verified speculation | Generate early, reuse only when authoritative document IDs match; otherwise run baseline generation. |
| Parallel enrichment | Fetch metadata while final answer generates, then merge unchanged results. |

Full aggregate methodology: [docs/EVALUATION.md](docs/EVALUATION.md).

Application-side experiment and rejected ablations: [docs/APPLICATION_SIDE_EXPERIMENTS.md](docs/APPLICATION_SIDE_EXPERIMENTS.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules or a particular domain. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and domain-expert evaluation.
