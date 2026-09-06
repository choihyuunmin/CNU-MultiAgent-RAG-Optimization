# Multi-Agent RAG Optimization

## 2026-09-06 (2차): 무손실 추측 디코딩으로 오케스트레이터 CoT 시간 2.7배 단축

같은 모델·프롬프트·탐욕 디코딩을 유지하면서 오케스트레이터(gemma-4-31B-it)의 구조화
호출을 가속했습니다. 실제 로그로 제안 정책을 오프라인에서 고르는 오라클 시뮬레이션과,
단계별 응답 이력을 프롬프트 조회와 결합하는 vLLM 플러그인 제안기가 핵심입니다.
400문항·동시성 4에서 분류/준비 호출이 운영 인스턴스 대비 2.68배(62.7% 단축) 빨라졌고,
출력 변동은 같은 서버 반복 호출의 자연 변동과 같은 크기이며 출처 라벨 기준 검색 지표는
모든 조건에서 동일했습니다. 단일 사용자 종단은 1.38배(27.5%), 동시성 4 종단은 워커 병목
이동으로 2.9%에 그쳤습니다. KV 용량 축소로 인한 대기열 선두 차단, 복제본 부하분산과
지연 인지 라우팅, ingress·오버레이 스트리밍 경로 계측도 포함합니다.
[프로토콜](docs/MOLEG_SPECDEC_PROTOCOL_20260906.md) ·
[결과 보고서](docs/MOLEG_SPECDEC_RESULTS_20260906.md) ·
플러그인 `integrations/2025-moleg-search/vllm_plugin/` ·
시뮬레이터 `scripts/simulate_moleg_specdec_policies.py`.

## 현재 연구 범위 — 검색기는 유지하고 바깥에서 가속하기

이 저장소의 목적은 운영 중인 `moleg-search`를 고치는 것이 아니라, 검색기 바깥의
중계 서버·네트워크·GPU 처리 방식으로 지연을 줄이는 논문 주제를 실험하는 것입니다.
운영 Kubernetes 배포와 모델 서버는 그대로 두고 별도 프로세스에서 비교합니다.
앱 내부 호출 제거·검색어 변경 등의 과거 실험은 참고 자료이며 현재 운영 적용안이 아닙니다.

2026-09-06 실험은 같은 모델 요청의 전달 경로, 동시 처리 제한, 짧은 요청 우선 처리,
실제 요청 본문의 압축 전송을 구분합니다. 방법은
[실험 프로토콜](docs/MOLEG_INFRA_PROTOCOL_20260906.md)에 기록합니다.
400문항 × 4조건 × 2회(3,200개 모델 요청)를 측정한 결과, 가벼운 외부 중계는
기존 통로보다 평균 7.34% 단축했고, 8개 동시 처리 제한은 제한 없는 중계보다
16.42% 느렸습니다. [방법·결과·논문 기여 정리](docs/MOLEG_INFRA_RESULTS_20260906.md)를
참고하세요. 이는 모델 요청 구간의 결과이며 최종 답변 정확도 개선을 입증하지 않습니다.
동시 요청 4개·48문항 보조 실험에서는 단축률이 2.06%였고, 요청 본문 압축은
59.80%의 바이트 절감에도 전송 처리 시간이 0.470밀리초 늘었습니다.
모델 호출 구간의 단축률과 전체 검색 시간의 단축률을 혼동하지 않습니다.

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

The measured app-level variants show a latency-fidelity tradeoff: stacking techniques
reaches ~7x but at 400 queries retrieval Recall collapses to 0.33-0.45, because search amplifies
small extraction differences. This motivates evaluating serving-level decode
acceleration (quantization / tensor parallelism / speculative decoding), with
separate output and quality checks rather than assumed identity. The orchestrator decodes at only
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
so fp8 needs the same regression validation as app-level tricks. Speculative
decoding and tensor parallelism could not be measured in that run: the venv's
n-gram proposer is broken (numba vs NumPy 2.4, and
the shared production venv was left untouched) and GPU0 is full (no second GPU
for tensor-parallel). Serving-level acceleration does not automatically preserve
observed outputs. Speculative distribution guarantees have assumptions, while
floating-point and batching changes can affect bitwise identity. These methods
need a maintenance window / spare GPU and explicit gemma-31B validation.

Details: [docs/MOLEG_ACCELERATION_ABLATION_20260904.md](docs/MOLEG_ACCELERATION_ABLATION_20260904.md).

## 2025-moleg-search harness methods, described (2026-09-04)

A prose write-up of each method (not code names) with an end-to-end 400-query
comparison (concurrency 4, search included), and a newly devised harness method:

| Method | Mean | Speedup | Recall | Top-1 | Verdict |
|---|---:|---:|---:|---:|---|
| Current production (sequential 2 calls + tool-echo search) | 4.67 s | 1.00x | 1.000 | 1.000 | reference |
| Parallel analysis + typed dispatch | 3.75 s | 1.24x | 0.995 | 0.994 | accepted |
| Verified fast-path cascade (new) | 1.38 s | 3.37x | 0.422 | 0.329 | rejected |

The accuracy-preserving winner is running the two independent analysis calls
concurrently and calling the search handler directly (removing the redundant
tool-echo): 24% faster end-to-end at Recall 0.995. The newly devised speculative
cascade (fast small model + deterministic grounding gate, deferring to the large
model on failure) reaches 3.37x but a cheap gate cannot preserve retrieval
fidelity (Recall 0.422) — the fast model diverges in phrasing even when grounded.
CoT reduction preserves accuracy only where reasoning is genuinely wasted (the
translation agent), not on the orchestrator (which emits none) or as an
extraction substitute.

Full prose method descriptions and data:
[docs/MOLEG_HARNESS_METHODS_20260904.md](docs/MOLEG_HARNESS_METHODS_20260904.md).

## 2025-moleg-search all-methods comparison, 400 distinct questions (2026-09-05)

Built a 400 distinct-question set and compared every method against the existing
baseline (time + retrieval accuracy).

| Method | Mean | Speedup | Recall | Verdict |
|---|---:|---:|---:|---|
| Baseline (existing) | 3.73 s | 1.00x | 1.000 | reference |
| typed dispatch | 3.17 s | 1.18x | 0.994 | accuracy-preserving |
| parallel analysis | 2.92 s | 1.28x | 0.995 | accuracy-preserving |
| merge | 2.68 s | 1.39x | 0.944 | small loss |
| fast model | 1.02 s | 3.64x | 0.377 | rejected |
| compact + guided | 1.00 s | 3.74x | 0.282 | rejected |
| cascade | 1.07 s | 3.49x | 0.305 | rejected |

Plus a serving-level row: n-gram speculative decoding cuts the gemma-4-31B
analysis stage 47% (2.89->1.53 s, 1.89x), with 392/400 identical observed outputs.
This is an analysis-stage result, not proof of bitwise losslessness or measured
end-to-end composability. A repeated same-serving control is needed to separate
ordinary output variation from speculative-decoding differences. The table's
Recall measures agreement with baseline documents, not expert-verified accuracy.
Parallel + typed dispatch + an n-gram-served orchestrator remains a combination
to validate end to end; aggressive model substitution collapses baseline
retrieval agreement (Recall 0.28-0.38).

Data: [docs/MOLEG_HARNESS_METHODS_20260904.md](docs/MOLEG_HARNESS_METHODS_20260904.md).

## Paper-oriented follow-up: retrieval and full streaming API (2026-09-05)

A new 400-distinct-question benchmark includes source-grounded law questions,
scenarios, comparisons, translation, and conversation. The completed streaming
experiment uses 400 questions × 3 conditions × 2 repetitions (2,400 requests),
with concurrency 4 and rotated condition blocks on the existing two H200 GPUs.
No production model restart or deployment was performed.

| Condition | Mean API latency | p95 | Mean reduction, question-bootstrap 95% CI |
|---|---:|---:|---:|
| Frozen-source baseline | 13.772 s | 30.852 s | Reference |
| Same-model execution/transport bundle | 12.213 s | 27.603 s | 11.32% [5.20%, 16.70%] |
| Bundle + authenticated reranking + low-reasoning streamed answers | 10.619 s | 23.681 s | 22.89% [19.54%, 26.92%] |

In the separate repeated retrieval experiment, authenticated reranking improves
source-clause Hit@1 from 107/131 (81.68%) to 118/131 (90.08%), with Holm-adjusted
McNemar p=0.03545. These are source-derived **silver known-item** labels, not
expert relevance gold. The gain is primarily ordering clauses within a law;
it does not establish improved final-answer correctness. Full-API source-clause
inclusion is 91.60% for both baseline and the balanced bundle. Neither bundle
has proven quality equivalence or bitwise losslessness. Two approximately
240-second application timeouts remain included in the latency statistics.

The separate JSON API experiment completes 800 requests (400 × 2 conditions):
13.000→11.733 s, a 9.74% reduction [4.12%, 15.57%]. A fixed-input generation
control (60 questions × 3 conditions × 2 repetitions) measures a 41.84% reduction
[23.95%, 53.17%] from lower reasoning on the directly connected worker. This is
upstream generation latency, not another full-API speedup. Blind phi-4 assessment
of 131 source-grounded first-repeat answers per condition does **not** establish
improved answer quality or equivalence (support: 1.908→1.901 on a 0–2 scale).
Human expert labels remain unset; a 400-question blinded review packet is ready.

A four-arm **search-only** experiment also finds an important counterexample to
agreement-based evaluation: keeping the original question without generative
extraction takes 0.426 s versus 3.425 s (8.03×), while source-clause Hit@1 is
120/131 versus 118/131—even though document Recall against the old output is
only 0.345 across all 400 queries (0.512 on the same 131 source-QA queries).
On those same 131, the small extractor has higher agreement (0.693) but lower
source Hit@1 (110/131), reversing the candidate ordering between the two metrics.
Quality equivalence is not established, and the synthetic questions
often name the jurisdiction explicitly. This is not an 8× full-service result.
Combining both candidate pools covers 131/131 designated source clauses, not all
relevant legal evidence. See the report for paired tests and limitations.

Methods, limitations, additional control results, and reproducibility artifacts:
[Korean study report](docs/MOLEG_PAPER_RESULTS_20260905.md) and
[protocol](docs/MOLEG_PAPER_PROTOCOL_20260905.md).

## Integration boundary

Inputs and outputs use standard Python mappings and dataclasses. No dependency on original application modules or a particular domain. Integrators remain responsible for retrieval, reranking, document safety filtering, LLM calls, and domain-expert evaluation.
