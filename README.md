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

## 2025-moleg-search live experiment

The 2026-09-03 live-server study used the real MOLEG hybrid search stack
(OpenSearch translation/origin indices, BGE-M3 embeddings, and the multi-agent
query loop).  Removing one discarded LLM tool-echo step reduced the 50-query
search-stage mean from 4.263 s to 0.250 s with identical ranked document IDs.
Coalescing the duplicate query embeddings and preparing the OpenSearch pipeline
once reduced the remaining hybrid-search mean by another 15.8%, also with exact
rank preservation.  A 10-query `/api/generate` integration check reduced mean
latency from 11.356 s to 6.486 s; document Recall versus control was 0.975.
That small end-to-end check is provisional because Top-1 agreement was 0.900;
the exact-rank-preserving search-stage candidate is accepted, while full answer
pipeline rollout remains gated on a larger repeated evaluation.

Protocol, accepted/rejected ablations, reranker finding, scripts, and raw metric
files: [docs/MOLEG_SEARCH_COT_OPTIMIZATION_20260903.md](docs/MOLEG_SEARCH_COT_OPTIMIZATION_20260903.md).

## 2025-moleg-search orchestrator latency (2026-09-04)

A follow-up study aggregated the observable agent/LLM traces of 252 live
`/api/generate` requests. It found the latency is dominated by the orchestrator
LLM, not search: the orchestrator (gemma-4-31B-it) was **88.2%** of non-streamed
LLM time, while the search tool call was only 5.2%. The 2026-09-03 work had
optimized that small search path. Each law-search request runs the orchestrator
as several sequential calls (classify, preparation, select, generation), each
1–4.5 s even for small JSON outputs; server prefix caching was already on (~54%
token hit rate).

Merging the two query-analysis calls (classify + preparation, which both read the
same user query) into one orchestrator call reduced that stage from 3.691 s to
2.645 s (**28.4%, 1.047 s/request**) over 50 fixed questions, with completion
tokens dropping 122→87. Against the current two-call output as pseudo-gold, task,
country, and guardrail decisions matched 100%, keyword Jaccard was 0.99/0.97, and
the free-text `transformed_query` matched exactly on 43/50 (wording-only
differences). Full rollout stays gated on a search-stage Recall/Top-1 regression.

Protocol, role-by-role breakdown, scripts, and raw metrics:
[docs/MOLEG_ORCHESTRATOR_LATENCY_20260904.md](docs/MOLEG_ORCHESTRATOR_LATENCY_20260904.md).

## 2025-moleg-search long-CoT and fan-out contention (2026-09-04)

Two follow-up experiments separated the latency causes further.

**Long CoT is real but localized.** Aggregating all logged calls, the slowest
role is the translation agent (ollama gpt-oss:20b): 562 calls, mean 6.9 s, p95
19.2 s, max 57.0 s, often with empty output. It emits long reasoning the app
discards. Reducing its reasoning (ollama `think=low`) cut warm per-clause
translation 56.6% (2.3x; 19x on a cold call) over 12 clauses with semantically
equivalent output. The vLLM gpt-oss worker showed ~0 reasoning at default; gemma
emits no separate reasoning at all. So "long CoT" is the translation agent, not
the orchestrator.

**Fan-out contention is localized too.** A concurrency sweep on a shared server
showed the gpt-oss worker (`--max-num-seqs 4`) inflates per-call latency 2.75x at
K=16 with throughput plateauing ~17.5/s (admission-control queueing), while gemma
scales cleanly (no inflation to K=16 short / K=8 generation; near-linear
throughput). The orchestrator lever is therefore fewer sequential calls, not
concurrency; the worker lever is its concurrency cap.

Protocol, per-model reasoning check, sweeps, scripts, and raw metrics:
[docs/MOLEG_COT_AND_CONTENTION_20260904.md](docs/MOLEG_COT_AND_CONTENTION_20260904.md).

## 2025-moleg-search acceleration ablation (2026-09-04)

Combining paper-oriented techniques on the query-analysis stage (call fusion,
fast-model right-sizing, compact schema, grammar-constrained decoding) and
measuring both latency and *real retrieval fidelity*, re-run over **400 queries**
(concurrency 4) with an added accuracy-preserving parallel arm:

| Method | Mean | Speedup | Recall | Top-1 | kw Jaccard |
|---|---:|---:|---:|---:|---:|
| Baseline (2 calls, gemma-31B) | 3.73 s | 1.00x | 1.000 | 1.000 | 1.000 |
| Parallel classify∥prep (accuracy-preserving) | 3.43 s | 1.09x | 0.980 | 0.976 | 0.994 |
| + call fusion (gemma) | 3.04 s | 1.23x | 0.917 | 0.878 | 0.955 |
| + fast model (gpt-oss) | 0.48 s | 7.86x | 0.446 | 0.372 | 0.556 |
| + compact schema | 0.54 s | 6.89x | 0.331 | 0.229 | 0.603 |
| + guided decoding | 0.54 s | 6.94x | 0.345 | 0.223 | 0.588 |

App-level acceleration is a clear latency-accuracy Pareto: stacking techniques
reaches ~7x but at 400 queries retrieval Recall collapses to 0.33-0.45, because search amplifies
small extraction differences. Accuracy-preserving speedup therefore needs
serving-level decode acceleration (fp8 / tensor-parallel / speculative decoding)
that keeps the orchestrator's output identical; the orchestrator decodes at only
~53 tok/s vs 239 (gpt-oss-20b) and 191 (gemma4-e4b). Accuracy is retrieval
fidelity vs the current baseline (pseudo-gold), not expert-judged.

Protocol, technique definitions, serving-level recommendation, script, and raw
metrics: [docs/MOLEG_ACCELERATION_ABLATION_20260904.md](docs/MOLEG_ACCELERATION_ABLATION_20260904.md).

## 2025-moleg-search serving-level acceleration (measured 2026-09-04)

Serving-level decode acceleration was measured on real hardware by serving a
same-family benchmark model (gemma-4-E4B) only in a spare GPU's free memory and
tearing it down; production was not restarted. fp8 quantization vs bf16 gave
**+21-23% decode throughput** (186->225 tok/s extract, 191->236 gen; latency
-17%), but the outputs were **not** preserved (0/8 identical, 0.569 similarity),
so fp8 needs the same regression validation as app-level tricks. The truly
output-lossless techniques (speculative decoding, tensor parallelism) could not
be measured here: the venv's n-gram proposer is broken (numba vs NumPy 2.4, and
the shared production venv was left untouched) and GPU0 is full (no second GPU
for tensor-parallel). Net: serving-level acceleration is real, but "serving-level
= accuracy-preserving" is not automatic; only the lossless methods guarantee it,
and they need a maintenance window / spare GPU to measure on gemma-31B.

Details: [docs/MOLEG_ACCELERATION_ABLATION_20260904.md](docs/MOLEG_ACCELERATION_ABLATION_20260904.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules or a particular domain. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and domain-expert evaluation.
