# CNU Multi-Agent RAG Optimization

Domain-independent latency optimization for multi-agent retrieval-augmented generation systems.

Repository contains only optimization primitives:

- confidence-gated local routing with LLM fallback;
- contract-constrained link-state routing with EWMA latency and load cost;
- trace-compiled procedure reuse with typed fail-closed contracts;
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
- prompt-free LLM timing metrics;
- fail-closed evidence-convergence routing;
- bounded first-token hedging for ablation experiments.

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

A later one-pass ablation found larger raw latency reductions for compiled
procedure reuse, but its 0.9102 pseudo-gold fidelity missed the 0.97 acceptance
gate. It is reported as rejected evidence, not a new best result. See
[docs/CASE_STUDY_400Q_20260903.md](docs/CASE_STUDY_400Q_20260903.md).

## Measured outcome

Accepted validation used one large-domain retrieval system as a case study: 400 queries, concurrency 4, and 3,679,496 indexed records. Domain-specific fields and data are not required by this package.

| Method | LLM calls | Mean | p50 | p95 | p99 | Pseudo-gold fidelity | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Current control | 1,657 | 22.36 s | 22.85 s | 51.77 s | 107.57 s | Reference | Control |
| Safe hybrid | 1,650 | 19.94 s | 21.20 s | 47.85 s | 59.83 s | 0.9769 | Accepted |
| Safe hybrid + tail hedge | 1,668 | 19.76 s | 21.52 s | 47.00 s | 56.16 s | 0.9766 | Rejected as default |

Remote inference engine remained unchanged. Safe hybrid combined bounded connection reuse, parallel independent enrichment, and evidence-constrained critical-path routing. It reduced mean latency 10.8%, p95 7.6%, and p99 44.4%. Document-ID Recall was 0.9827, nDCG@10 0.9798, Top-1 agreement 0.9700, and answer similarity 0.9750. Tail hedging fired on 17/400 requests and produced only marginal extra latency reduction while increasing calls, so it remains an ablation rather than default behavior.

## One-line method examples

| Method | Example |
|---|---|
| Confidence routing | Route only when one intent plus explicit entity, domain, and action evidence all match; otherwise use existing LLM router. |
| Typed dispatch | Call sole selected search tool with validated structured arguments instead of asking LLM to echo them. |
| HTTP keep-alive | Reuse one bounded client for LLM, embedding, and reranker calls. |
| Single-flight | Two simultaneous identical questions share one in-flight request; no answer remains cached. |
| Verified speculation | Generate early, reuse only when authoritative document IDs match; otherwise run baseline generation. |
| Parallel enrichment | Fetch metadata while final answer generates, then merge unchanged results. |
| Evidence convergence | Skip LLM selection only when ranked evidence has one parent, exact scope, sufficient score, and query-term coverage. |
| Contract link-state routing | Choose the lowest-delay semantically equivalent path after contract, load, and failure checks. |
| Compiled procedure reuse | Reuse a verified stage-to-action procedure, never a prior answer; fall back on every contract miss. |
| Tail hedge | After delayed first token, race one duplicate under a strict concurrency budget and cancel loser. |

Full aggregate methodology: [docs/EVALUATION.md](docs/EVALUATION.md).

Application-side experiment and rejected ablations: [docs/APPLICATION_SIDE_EXPERIMENTS.md](docs/APPLICATION_SIDE_EXPERIMENTS.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules or a particular domain. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and domain-expert evaluation.
