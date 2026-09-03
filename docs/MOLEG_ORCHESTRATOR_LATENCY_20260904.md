# 2025-moleg-search 오케스트레이터 지연 분석·단축 실험

실험일: 2026-09-04 (Asia/Seoul)

## 결론

세계법제정보 지능형 검색시스템의 멀티에이전트 응답 지연은 **검색 단계가 아니라
오케스트레이터(orchestrator) LLM 호출**에 집중돼 있었다. 실제 운영 로그(252건의
`/api/generate` 요청)의 관찰 가능한 에이전트·LLM 추적을 집계한 결과, 비스트리밍
LLM 호출 시간의 **88.2%가 오케스트레이터(gemma-4-31B-it) 호출**이었고 검색 도구
호출은 5.2%에 그쳤다. 2026-09-03 실험이 최적화한 검색 dispatch 경로는 전체 지연의
작은 부분이었다.

오케스트레이터는 한 요청에서 **분류(classify) → 준비(preparation) → 선별(select)
→ 생성**을 각각 별도 gemma 호출로 순차 수행한다. 이 중 같은 사용자 질문을 두 번
읽는 **분류 + 준비 두 호출을 한 번의 호출로 병합**하면, 50개 고정 질문에서 분석
단계 평균이 3.691초에서 2.645초로 **28.4% 감소(요청당 1.047초 절감)**했다. 병합
결과는 task 100%, country 100%, 가드레일 100% 일치했고 키워드 Jaccard는 0.99/0.97,
completion 토큰은 122→87로 줄었다. 운영 배포는 변경하지 않았다.

## 실제 대상과 환경

- 검색 UI/기능: `2025-moleg-search`
- 애플리케이션(멀티에이전트 오케스트레이션): k8s 노드 `moleg-app`
- 추론 백엔드: `moleg-gpu`, NVIDIA H200 NVL 2장, vLLM + LiteLLM 프록시
- LLM 역할 라우팅(LiteLLM):
  - `orchestrator`(분류·분석·선별·생성): **google/gemma-4-31B-it**
  - `worker agent` / `retrieval tool`(도구 호출·서술): **openai/gpt-oss-20b**
  - `comparison_sft`(비교 전용 SFT): gemma4-e4b
  - `guideline agent`: kanana-safeguard-8b
  - `translate agent1~4`: ollama(gpt-oss:20b, translategemma:4b, llama3.2:3b, exaone3.5:7.8b)
- 임베딩 bge-m3, 리랭커 bge-reranker-v2-m3-ko, OpenSearch hybrid 검색

운영 배포는 중단·재기동하지 않았다. 실측은 운영과 동일한 vLLM 오케스트레이터
엔드포인트에 대해 순차(concurrency=1)로 수행했고 warm-up은 집계에서 제외했다.

## 관찰 가능한 추론 추적 분석 (운영 로그 집계)

`REQ recv` / `LLM sent` / `LLM recv | role=… model=… elapsed=…ms resp_len=…` /
`tool_loop` / `Tool done` 로그를 252개 `/api/generate` 요청에서 집계했다. 이는 모델의
비공개 내부 사고 추정이 아니라, 애플리케이션이 실제로 주고받은 호출의 벽시계
시간이다.

### 역할별 LLM 시간 점유 (비스트리밍 호출)

| 역할 / 모델 | 호출 수 | 평균 | p95 | 총합 | 점유 |
|---|---:|---:|---:|---:|---:|
| master / orchestrator (gemma-4-31B) | 749 | 2,336ms | 6,164ms | 1,749.6s | **88.2%** |
| tool / retrieval (gpt-oss-20b) | 233 | 444ms | 841ms | 103.4s | 5.2% |
| comparison_sft (gemma4-e4b) | 9 | 5,674ms | 11,248ms | 51.1s | 2.6% |
| tool_synthesis / worker (gpt-oss-20b) | 15 | 2,895ms | 6,912ms | 43.4s | 2.2% |
| translate (ollama) | 2 | 17,732ms | — | 35.5s | 1.8% |

요청당 총 비스트리밍 LLM 시간은 평균 8.03초, p95 16.78초, 최대 47.39초였고, 요청당
평균 4.1회(p95 8회, 최대 17회) 호출했다. 검색 도구 실행 자체는 평균 537ms로 빨랐다.

### 오케스트레이터 호출의 출력 크기별 분해

| 출력 길이(문자) | 성격 | 호출 수 | 평균 | 총합 |
|---|---|---:|---:|---:|
| < 120 | 분류·라우팅·선별 결과 | 369 | ~1,080ms | 398s |
| 120–300 | **준비(preparation) JSON** | 237 | 2,975ms | 705s |
| ≥ 300 | 국가별 생성·서술 | 143 | 4,521ms | 647s |

가장 큰 단일 덩어리는 **준비 단계(705s, 40%)**였다. 준비 프롬프트는 약 13KB(가드레일
지침 + 58개 지원 국가 목록 포함)로 프리필이 크고, 출력은 300자 내외인데도 평균 3초가
걸린다. 오케스트레이터 서버의 prefix caching은 켜져 있었으나(토큰 히트율 약 54%)
남은 프리필과 순차 호출 수가 지연을 지배했다.

### 병목 원인

1. **오케스트레이터 순차 호출 수.** law_search 한 건이 분류·준비·선별·생성 등
   여러 gemma 호출을 순차로 실행한다. 각 호출은 작은 JSON 출력에도 1~4.5초가 걸린다.
2. **동시 팬아웃 시 경합.** 다국가 질문은 국가별 준비·선별을 gemma에 동시 전송하는데,
   단일 gemma 서버에서 배치 경합이 생겨 호출당 지연이 2.3초→3.4~6.7초로 늘었다.
3. 검색·임베딩·리랭크·도구 호출은 상대적으로 작았다(합계 ~5%대).

## 실험: 분류 + 준비 호출 병합

분류(classify)와 준비(preparation)는 **같은 사용자 질문을 두 번 읽어** 각각
`{task}`와 `{bad_word/pii, transformed_query, keywords×2, country, article, title_hint}`를
낸다. 두 지침을 하나의 시스템 프롬프트로 합쳐 **한 번의 gemma 호출**로 두 구조를
동시에 산출하도록 했다. 50개 고정 질문(행위형 25 + 법령명형 25)에 대해 현재 방식(2회
순차 호출)과 병합 방식(1회 호출)을 각각 실측했다.

| 변형 | 평균 | p50 | p95 | 최대 | completion 토큰 |
|---|---:|---:|---:|---:|---:|
| 현재: classify + preparation (2회) | 3,691ms | 3,351ms | 5,317ms | 6,364ms | 122 |
| 병합: 1회 호출 | 2,645ms | 2,269ms | 4,531ms | 5,065ms | 87 |

- 분석 단계 평균 지연 감소: **28.4% (요청당 1,047ms)**
- 요청당 총 LLM 시간(평균 8.03초) 대비 약 13% 절감에 해당

### 정확도(현재 2회 출력 대비 회귀)

| 항목 | 일치 |
|---|---|
| task 분류 | 50/50 = 100% |
| country 추출 | 50/50 = 100% |
| 가드레일(bad_word/pii) | 50/50 = 100% |
| transformed_query 완전 일치 | 43/50 = 86% |
| keywords_from_original Jaccard | 0.990 |
| keywords_from_transformed Jaccard | 0.970 |

여기서 정확도는 **현재 2회 호출 출력을 pseudo-gold로 둔 회귀 지표**이며 법률 전문가의
절대 정답 판정이 아니다. task·country·가드레일은 완전 일치했다. transformed_query는
자유 서술형 검색 문구라 7/50이 표현만 달랐고(키워드 Jaccard 0.99로 사실상 동일),
이 필드는 하이브리드 검색에 키워드와 함께 쓰이므로 순위 영향은 제한적이다.

## 적용 결정

1. `classify_intent_with_confidence`와 `validation_agent.run_preparation`를 **하나의
   오케스트레이터 호출**로 병합해 law_search 진입부의 순차 gemma 호출을 1회 줄인다.
   출력 스키마는 `{task, bad_word_detected, pii_detected, transformed_query,
   keywords_from_original, keywords_from_transformed, country, specific_article_number,
   law_title_search_hint}` 단일 JSON.
2. 정규식 fast-path 분류가 먼저 성립하면 기존처럼 LLM 병합 호출을 건너뛴다.
3. 배포 전 게이트: 50문항 이상에서 검색 문서 Recall·Top-1·답변 유사도 회귀를 통과할 것.
   특히 transformed_query 표현 차이가 검색 순위에 주는 영향을 검색 단계 회귀로 확인한다.

### 추가 후보(미측정, 후속)

- 선별(select) + 생성(comment) 병합 또는 선별 경량화.
- 추천/대안 질문 생성을 응답 스트리밍 이후로 이동해 임계 경로에서 제외.
- 다국가 팬아웃 시 오케스트레이터 동시 호출 수 제한(경합 완화).

## 재현

- 스크립트: `scripts/evaluate_moleg_orchestrator_merge.py`
  (프롬프트·질문·국가 목록·자격증명을 저장소에 저장하지 않고 `--rag-root`의 운영
  소스에서 실행 시점에 재구성한다.)
- 집계 결과: `experiments/results/moleg-orchestrator-20260904/`

모든 결과 파일은 비밀값, 프롬프트 원문, 질문 원문, 엔드포인트 주소를 저장하지 않는다.
