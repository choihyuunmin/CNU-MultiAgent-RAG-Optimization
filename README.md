# Multi-Agent RAG Optimization

## 2026-09-15: 방법 탐색 + 동시성 용량 램프(하드웨어 한계)

지연 개선 방법을 먼저 비교한 뒤 최적안으로 동시성을 1→500 올려 하드웨어 한계를 판정했다
(조건당 1회, 운영·모델 변경 없음). **방법 탐색(C=4)**: worker `reasoning=low`가 −7.9%로
최적이며 재현율 1.00 유지. 온톨로지 검색 제거는 이득이 없어(임계경로 아님) 유지했다.
**용량 램프**: 30초 SLO 유효 용량의 무릎은 **동시성 ~5–10**(goodput 최대 ~0.38 rps@C10),
원시 처리량 포화는 **~50–100**(~0.6–0.66 rps, 오케 KV 20부터 99–100%·50부터 선점),
경성 붕괴는 **동시성 300**(성공률 40–44%, 240초 타임아웃 다수, 선점 ~130)으로 자동 중단
(C=500 미실행). 재현율은 C≤100에서 ≥0.985, C=300 급락은 타임아웃 부작용. 개선 조건은
C≤100 전 구간에서 지연 같거나 짧고(C1 16%, C100 8.3%) 고부하에선 이득이 잡음으로 수렴한다.
**결론**: 앱 계층 가속은 오케스트레이터 KV·prefill 용량에 상한되며, 그 이상은 KV 증설·복제·
프리필 가속이 필요하다.
[결과](docs/CAPACITY_RAMP_RESULTS_20260915.md) · [공개 자료](experiments/results/combined-sweep-20260915/README.md).

## 2026-09-14: 결합 하네스 어댑터 — 200문항 실측(시간↓·정확도 유지)

원격 `agent/generalize-multi-agent-rag`의 최신 하네스(ProgramHarness, CompletionRouter,
CoflowAdmission, NetworkHarness, InferenceOverlapAdapter, ApplicationAdapter)를 실험
브랜치로 통합하고, worker `reasoning_effort=low` 제어와 결합한 개선 어댑터를 만들었다.
200문항 × baseline/combined × 사용자 {1,4,16} × 2반복 = **2,400요청**을 실제 서빙으로
실행했다(SSE 정상 2,398, combined 실패 0). 질문 단위 paired bootstrap 단축률은
**U1 19.3%(12.5~29.0), U4 11.5%(8.3~14.8), U16 5.4%(1.5~9.9)** 로 세 부하 모두 유의하며,
검색 소스 재현율(Δ≈0)·근거 ID 일치(0.94~0.97)·blinded 판정(relevance 1.815→1.831,
support 1.80→1.831, 구간이 0 포함)은 저하 없이 유지됐다. 이득의 원천은 worker 답변 전
reasoning 제거(문자수 1,939→164, 첫 출력 2.39→0.35초)이며, 부하가 커질수록 오케스트
레이터 selection(7.25초)·preparation(6.62초) prefill이 종단을 지배해 이득이 5%로
압축된다. **정량화된 경계**: 앱 계층의 worker·검색 가속은 오케스트레이터 prepare+select
천장에 상한되며, 다음 레버는 오케스트레이터 prefill의 무손실 가속이다. 운영 배포·모델
변경 없음(모델 argv·포트 실험 전후 동일).
[결과](docs/COMBINED_HARNESS_RESULTS_20260914.md) ·
[프로토콜](docs/COMBINED_HARNESS_PROTOCOL_20260914.md) ·
[공개 자료](experiments/results/combined-20260914/README.md).

## 2026-09-11: 호출 상한 제거와 reasoning 병목 수정

현재 기본 실행에서 HTTP·생성·모델 예산 대기열을 제거했습니다. 일반 async 호출과
SDK 스트리밍을 사용하며, 누락되던 생성 옵션 전달과 취소 시 실제 스트림 종료를 수정했습니다.
워커에 낮은 추론량을 요청하고, 답변 없이 reasoning만 지속되는 요청에 첫 출력 기한과
명시적 오류 처리를 적용합니다. 정상 답변·도구 호출 및 첫 답변 지연은 별도로 계측합니다.

이 변경의 GPU 성능·답변 품질은 아직 미측정입니다. 아래 날짜별 결과는 당시 코드의
기록이며 현재 변경의 개선율이 아닙니다. 기존 `mode=budget`과 호출 상한 CLI는 제거됐습니다.
[원인·구현·실행 방법·검증 범위](docs/UNRESTRICTED_REASONING_20260911.md).

## 2026-09-10: 조건별300문항 실험 완료

기존400문항 실험은 서버 재부팅으로 8,100개 응답 기록을 남기고 중단됐습니다.
사용자 요청에 따라 재개분은 동일한300문항 × A/B/C × 사용자1/2/4/8/16/32 ×
남은2반복, 총 **10,800요청**으로 줄였습니다. 중단된 비교 묶음은 세 조건 모두
300문항으로 다시 측정하며, 기존400문항 자료와 재개분을 단순 합산하지 않습니다.
복구 후 검색 문서 수와 일부 모델 실행 명령이 달라진 점을 함께 기록합니다.
10,800요청과 자동 평가1,764건을 완료했습니다(응답 스트림 비정상14건, 판정 오류2건 보존).
U32에서 고정16 조건은 평균53.19초, 예산 조건은60.19초로 예산의 추가 가속을
확인하지 못했습니다. KV 압력 감소와 속도 개선은 별개이며 정확도 비열등성은 미입증입니다.
[재개 계획](docs/WORKFLOW300_RECOVERY_PROTOCOL_20260909.md) ·
[완료 결과](docs/WORKFLOW300_RESULTS_20260910.md) ·
[공개용 원자료·감사](experiments/results/workflow300-20260909/README.md) ·
[기존400문항 중단 기록](experiments/results/workflow400-20260908/README.md).

## 2026-09-08: 관측 가능한 CoT·워크플로 기반 통합 어댑터

연구 목표를 **에이전트 실행 흐름 관측 → 병목 판단 → 의미를 보존한 실행 가속 →
답변 정확도 검증**으로 통합했습니다. `WorkflowAdapter`는 단계별 모델 호출·reasoning
통계·대기를 관측하고, 모델별 동시 호출/토큰 예산과 전체 입력 일치 기반 선실행을 제공합니다.
출력 토큰 한도를 줄이는 기능이 아닙니다. 기존 앱 실험 launcher에 관측/예산 모드를
연결했으며, 운영 배포는 변경하지 않았습니다.

과거 추측 디코딩과 16·32명 상한 조정은 설계 근거로 연결하되 개선율을 합산하지 않습니다.
새 통합판은 CPU 합성 요청 768회의 payload·스트림 보존 검사를 통과했습니다.
후속 실제 GPU 비교에서는 U16 첫 반복 192회 중 191회가 완료됐고, 새 예산 조건의
240초 실패로 나머지 U32·둘째 반복은 중단했습니다. 기존/고정16/예산의 평균 관측시간은
70.49/37.60/42.13초입니다(마지막 수치는 실패 기한 포함). 예산은 KV 최대를
83.62%→56.78%로 낮췄지만 추가 가속을 보이지 못했습니다. 실패 요청에서는 워커
reasoning 126,142문자·답변 0문자가 관측됐습니다. 새 정책은 배포하지 않습니다.
오케스트레이터의 실제 tokenizer/압력 계측은 연결했으며, 새 선실행은 비활성입니다.
전문가 답변 동등성은 미검증이고 품질 게이트는 release를 거절합니다.

[통합 설계·기존 실험 대응·후속 비교](docs/WORKFLOW_ADAPTER_20260908.md) ·
[어댑터](src/cnu_rag_optimization/adaptive.py) ·
[실행 예제](examples/workflow_adapter.py) ·
[CPU 계약 검증](experiments/results/workflow-adapter-20260908/offline-contracts.json) ·
[실제 GPU 결과·실패 원인·후속 판단](docs/WORKFLOW_GPU_RESULTS_20260908.md).

## 2026-09-07 후속: 이중 상한 대조와 GPU 메모리 직접 계측

SSH 접속 후 현재 K8s 소스·설정을 확인했습니다. 생성 상한 4 외에 SSE 종료까지
유지되는 HTTP 상한 4가 별도로 있었습니다. 운영 서빙은 유지하고 별도 앱의 두 상한과
전송 방식을 바꾼 320회 대조를 완료했습니다. 16명에서 평균은 기준 58.18초,
상한 8은 43.43초, 상한 16+즉시 전송은 36.05초였습니다. 마지막 조건을 고정해
16·32명에서 순서를 교차한 별도 **512회 검증도 모두 정상 완료**했습니다.
최종 평균은 16명 **60.36→34.04초(43.60% 단축)**, 32명 **104.16→52.61초(49.49% 단축)**입니다.
질문 단위 95% CI는 각각 38.84~48.55%, 46.55~52.34%입니다. 출처 포함 지표는 같았지만
전문가 답변 품질 동등성을 입증한 것은 아닙니다. 선택용 결과는 검증과 합산하지 않습니다.

가용 RAM은 460GiB 이상이었고 swap·OOM 증가는 없었습니다. Nsight 장치 지표에서
지속 DRAM 포화는 관측하지 못했지만, 상한 16의 오케스트레이터 KV 최대는 99.73%로
용량 압력이 높았습니다. 따라서 GPU 활동률·HBM 활동·KV 용량·host RAM을 구분합니다.
반복 검증에서는 KV 최대 98.93%, 선점 0이었으며 전력 제한 누적 증가도 관측했습니다.
운영 설정은 유지했고 실험 앱·계측 종료와 임시 인증 설정 삭제까지 완료했습니다.

[후속 결과·기술 아이디어](docs/MOLEG_SCALING_ISOLATED_RESULTS_20260907.md) ·
[조건·보정 기록](docs/MOLEG_SCALING_ISOLATED_PROTOCOL_20260907.md) ·
[선택용 대조 그림](experiments/results/moleg-scaling-isolated-20260907/factorial/ablation.pdf) ·
[16·32명 반복 검증 그림](experiments/results/moleg-scaling-isolated-20260907/validation-counterbalanced/scaling.pdf) ·
[보완 논문](docs/MOLEG_PAPER_DRAFT_20260905.md).

## 2026-09-07 초기: 사용자 규모별 실측과 앱 대기열 병목

아래 접속 제한·미측정 상태는 초기 실험 당시의 기록입니다. 후속 직접 계측과는 별도 실행입니다.

64개 고정 질문을 외부 사용자 1·2·4·8·16·32명에서 실행했습니다. 첫 스윕 384회 중
383회가 정상 완료했고, 32명에서 180초 타임아웃이 발생해 계획한 두 번째 반복은
중단했습니다. 평균 관측 지연은 1명 13.86초 → 16명 59.78초 → 32명 108.08초였습니다.
현재 경로는 내부 생성 슬롯 4개 제한과 일치하는 응답 순서를 보였으며, 1→16명
지연 증가의 약 94%는 첫 진행 이벤트 이전 구간에서 발생했습니다. 이는 GPU에
16·32개의 파이프라인을 동시에 실행한 결과가 아닙니다.

오케스트레이터 KV 관측 최대는 58.34%, 선점은 0으로, 이번 지연 증가를 GPU 메모리
용량 부족으로 단정할 근거는 없었습니다. HBM 대역폭·추론 호스트 RAM/swap은
접근 제한으로 미측정입니다. 장문 비교 답변에서는 인위적 전송 대기가 의심되어
보관된 helper의 CPU 재현도 수행했지만, 이를 운영 API의 가속 결과로 주장하지 않습니다.
기존 **2.9%는 동시성 4의 종단 단축률**이며 신뢰구간이 0을 포함합니다.

U=16의 외부 고정·적응형 상한 대조 192회도 완료했습니다. 총 지연 단축은 각각
1.88%·0.67%로 신뢰구간이 0을 포함했습니다. HTTP 구간에서 줄어든 시간의 대부분이
외부 대기로 옮겨졌으므로 기본 개선안으로 채택하지 않았습니다. 예비 실험 포함 새 API
요청은 총 592회입니다. 내부 슬롯·전송을 독립 조절하는 다음 대조는 코드 준비까지
완료했으며, 추론 호스트 접속과 앱 설정 접근이 필요합니다.

[추가 결과·기술 제안](docs/MOLEG_SCALING_RESULTS_20260907.md) ·
[실험 프로토콜·독립 대조 계획](docs/MOLEG_SCALING_PROTOCOL_20260907.md) ·
[보완 논문](docs/MOLEG_PAPER_DRAFT_20260905.md) ·
[사용자 규모 곡선](experiments/results/moleg-scaling-20260907/sweep/scaling.pdf) ·
[회신 초안](docs/MOLEG_SCALING_REPLY_20260907.md).

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
