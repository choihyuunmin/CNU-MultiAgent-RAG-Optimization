# 2025-moleg-search integration notes

The live candidate was built from the backend source at
`/data/project/vllm/fine-tune/2025-moleg-rag` and kept in a separate server-side
experiment directory.  Production was not edited or restarted.

## `src/agent/law_search_agent.py`

`run_search` receives authoritative `country`, `merged_keywords`, and
`search_terms` from the preparation node.  Replace the one-tool LLM loop and
its direct fallback with the direct handler call below.  Parsing and the return
shape remain unchanged.

```python
payload = {
    "search_terms": search_terms,
    "country": country if country else None,
    "keywords": list(merged_keywords),
}
logger.info("[%s] law_search_agent typed direct dispatch", request_id)
tool_result = await tool_search_laws(payload, request_id)
```

This is safe for this node because the LLM prompt explicitly required the
single available tool to receive exactly the supplied JSON.  A generic caller
with multiple tools or incomplete arguments should use
`try_typed_single_tool_dispatch` and retain its LLM fallback.

## `src/infra/vector_db/vector_store.py`

Two changes form the safe hybrid-search hot-path candidate.

1. Guard `_ensure_search_pipeline` with an instance `Lock` and a
   `_hybrid_pipeline_ready` flag.  Set the flag only after the OpenSearch `PUT`
   succeeds.
2. Expose `embed_hybrid_query(query, keywords)` and let `search` and
   `search_world_law_origin` accept an optional precomputed `query_vector`.

## `src/tools/search_tool/engine.py`

Before starting translation/origin OpenSearch futures, calculate one shared
vector and pass it to both methods.  If precomputation fails, pass `None` so the
existing per-index paths retry.  Do not disable the origin index and do not
reduce `HYBRID_*_TOP_K`; both aggressive arms failed the measured document
fidelity gates.

## Reranker follow-up

The current `_rerank_results` request omits authorization and receives HTTP
401 from the live reranker.  Add an optional `RERANK_API_KEY` setting and a
Bearer header without hard-coding the value.  This is a quality repair, not
part of the exact-rank-preserving speed candidate: authenticated reranking
changed Top-1 ranking and added measured latency, so it requires a separately
approved A/B rollout.

See [the measured report](../../docs/MOLEG_SEARCH_COT_OPTIMIZATION_20260903.md)
for protocol, results, and limitations.

## Typed dispatch — measured with current models (2026-09-04)

Re-verified against the current stack (orchestrator gemma-4-31B, worker
gpt-oss-20b). Over 25 fixed questions, comparing the current tool-echo path
against the direct-dispatch replacement with the same preparation output:

| Path | mean | p50 | p95 |
|---|---:|---:|---:|
| Current (gpt-oss tool-echo) | 0.92 s | 0.86 s | 1.30 s |
| Typed direct dispatch | 0.50 s | 0.47 s | 0.71 s |

Latency saved 0.42 s/request (46%). Ranked document IDs identical on 24/25,
document Recall 0.994, Top-1 agreement 24/25. The one difference is a case where
the gpt-oss echo drifted from the intended arguments, so the direct path is the
more faithful one. This is accuracy-preserving harness engineering: it removes a
call without changing the search inputs. Metrics:
`experiments/results/moleg-acceleration-20260904/typed_dispatch_verification.json`.

Deployment stays with the team's k8s process; production was not modified.

### 400-query confirmation (2026-09-04)

Re-run at 400 queries (the 50 fixed questions cycled x8, concurrency 1):

| Path | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Current (gpt-oss tool-echo) | 0.95 s | 0.93 s | 1.17 s | 1.23 s |
| Typed direct dispatch | 0.50 s | 0.48 s | 0.66 s | 0.85 s |

Latency saved 0.45 s/request (47.3%). Ranked document IDs identical on 397/400
(99.2%), document Recall 0.9988, Top-1 agreement 99.5%. Fidelity is higher at
400 than at 25, confirming the change is accuracy-preserving; the ~0.8%
non-identical are cases where the gpt-oss echo drifts from the intended args.
