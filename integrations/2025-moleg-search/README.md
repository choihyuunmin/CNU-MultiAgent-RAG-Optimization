# 2025-moleg-search integration notes

## 참조 ID 축약·복원 어댑터

모델에 긴 법령 ID `918273_54` 대신 짧은 번호 `0`을 전달하고, 선택 결과의 `0`을
실제 법령 ID로 되돌린다. 본문·제목·점수는 유지한다. 후보가 16개 이상일 때만 적용하며,
중복 ID·다른 필드에서 언급한 ID·복원 오류는 별도로 처리한다.
[입력·출력 예시와 동작 설명](../../docs/REFERENCE_HARNESS_20260915.md).

### Adaptive selection IDs (2026-09-15)

The [structured adapter configuration](structured-adaptive.json) enables short,
request-local candidate IDs only when at least 16 candidates are present. It
restores original IDs before the existing selection parser, preserves every
evidence field and the input layout, and falls back on invalid returned IDs.
It does not cache answers. Source fingerprints must match the verified app
revision; inspect and revalidate a new revision before changing those pins.

Add these options to the existing isolated launcher command (retain the required
`--app-root`, `--app-env`, `--serving-env`, `--proxy-config`, `--trace` and `--port`):

```sh
--adapter-config integrations/2025-moleg-search/workflow-combined.json \
--structured-harness integrations/2025-moleg-search/structured-adaptive.json
```

The workflow configuration controls worker reasoning separately. Use
`workflow-combined-baseline.json` to evaluate the ID adapter without low reasoning.
Raw capture is disabled in the reusable configuration. The study's private
configuration enables capture for replay; aggregate metadata is recorded in the
ordinary private application trace. No inference engine restart is needed.

The [study protocol](../../docs/STRUCTURED_HARNESS_PROTOCOL_20260915.md) separates
baseline, prior low reasoning and the new adapter. JSON whitespace minification,
typed search dispatch and compact output grammar are exploratory options and are
disabled in the selected configuration. Reversible representation changes can
still change model decisions; fidelity and answer support require evaluation.

### Portable compiler binding

[structured-portable.json](structured-portable.json) selects the generic
`ReferenceContract` compiler through a thin MOLEG envelope binding. It retains
the separately measured 16-candidate policy. In an offline audit of all 575
captured selection calls, the 192 eligible calls produced the same encoded input
bytes, mappings and restored output values as the measured specialized adapter.
Unseen references in other prompt fields are conservatively bypassed.

Use the portable file as `--structured-harness` to select this binding. The
1,800-request timings used `structured-adaptive.json`; this offline equality
audit is not an additional end-to-end timing experiment. Calibration from the
synthetic topology study is not reused for legal retrieval. See
[generic design, evaluation and limits](../../docs/REFERENCE_HARNESS_20260915.md).

## Current execution (2026-09-11)

The app launcher now removes global HTTP/pipeline admission limits and uses an
observe-only workflow adapter. Its default `workflow-reasoning.json` applies low
effort and an answer-free deadline to the configured gpt-oss worker. Legacy
capacity flags and budget configurations are rejected. See the
[current implementation and validation scope](../../docs/UNRESTRICTED_REASONING_20260911.md).
The dated experiments below describe their archived code, not this revision.

## 2026-09-08 workflow adapter

The isolated scaling launcher now accepts `--adapter-config` and
`--adapter-trace` to layer observable stage timing and per-model token admission
on the legacy application adapter. Start with [observe-only configuration](workflow-observe.json).
See [contracts, historical evidence and combined evaluation plan](../../docs/WORKFLOW_ADAPTER_20260908.md).
The [live comparison](../../docs/WORKFLOW_GPU_RESULTS_20260908.md) stopped after
192 requests because the budget arm timed out once. KV pressure decreased but
incremental acceleration/reliability was not established. No production rollout.
With an explicit `serving_meter` configuration the MOLEG bridge now obtains
orchestrator token counts and pressure from the current serving instance.
Generic verified overlap is pressure-gated but is not enabled in this campaign;
legacy ungated overlap is disabled in combined mode. The historical notes below describe earlier experiments,
not current reranker health or proof of final-answer equivalence.

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
