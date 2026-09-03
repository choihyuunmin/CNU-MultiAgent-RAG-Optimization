# 2025-moleg-search 멀티에이전트 검색 최적화 실험

실험일: 2026-09-03 (Asia/Seoul)

## 결론

검색 인자가 이미 확정된 뒤 LLM에게 동일 JSON을 다시 쓰게 하는 `law_search_agent`
단계를 제거하는 것이 가장 큰 개선이었다. 50개 고정 질문에서 검색 단계 평균은
4.263초에서 0.250초로 94.1% 감소했고, 반환 문서 ID의 전체 순서, Recall, Top-1이
모두 1.000이었다. 이 변경에 search pipeline 1회 준비와 동일 질의 임베딩
single-flight를 결합하면 순수 hybrid 검색도 평균 15.8% 더 짧아졌다.

운영 배포는 변경하지 않았다. 후보 코드는 서버의 별도 실험 복사본에서 검증했다.

## 실제 대상과 환경

- 검색 UI/기능: `2025-moleg-search`
- 검색 백엔드 소스: `/data/project/vllm/fine-tune/2025-moleg-rag`
- 후보 복사본: `/data/project/vllm/fine-tune/experiments/2025-moleg-rag-typed-dispatch`
- GPU 서버: `192.168.100.53`, NVIDIA H200 NVL 2장
- 추론: vLLM 0.19.0, 검색 tool 역할은 `microsoft/phi-4`(포트 8002)
- 임베딩: `BAAI/bge-m3`(포트 8005)
- 리랭커: `dragonkue/bge-reranker-v2-m3-ko`(포트 8001)
- 검색: OpenSearch의 번역·원문 인덱스 hybrid 검색

저장소의 `main`과 `agent/generalize-multi-agent-rag`를 모두 확인했다. 실험 작업은
일반화 브랜치의 typed dispatch/single-flight 원칙을 실제 MOLEG 검색 코드에 적용한
`experiment/moleg-search-cot-optimization` 브랜치에 기록했다.

## 관찰 가능한 추론 추적 분석

검색 노드에 진입할 때 `country`, `merged_keywords`, `search_terms`는 이미 준비돼 있다.
그런데 기존 `run_search`는 다음 작업을 추가로 했다.

1. 유일한 도구 `search_laws`와 완성된 인자 JSON을 Phi-4에 전송한다.
2. “JSON과 동일한 인자로 반드시 도구를 호출하라”고 지시한다.
3. LLM의 설명 텍스트는 사용하지 않고 tool trace만 찾는다.
4. tool trace가 없으면 원래 JSON으로 `search_laws`를 직접 호출한다.

실측 50회 모두에서 LLM은 도구를 선택하지 않았고, 시스템은 생성 결과를 버린 뒤
직접 호출 폴백을 수행했다. 단일 관찰 예에서는 655자 설명 생성에 5.156초가 들었다.
이는 숨겨진 내부 CoT를 추정한 것이 아니라, 애플리케이션이 실제로 받은 모델 출력과
tool trace를 분석한 결과다.

후보는 유일 도구·완성 인자라는 계약이 성립할 때 동일 handler를 즉시 호출한다.
검색, 시소러스, 임베딩, OpenSearch, 결과 포맷은 바꾸지 않는다.

## 실험 1: LLM dispatch 제거

서버에 포함된 고정 50문항을 사용했다. 각 문항에서 기준/후보 순서를 번갈아 실행했고,
각 변형 1회 warm-up은 집계에서 제외했다.

| 변형 | LLM 호출 | 평균 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|---:|
| 현재 LLM auto + direct fallback | 50 | 4.263초 | 3.370초 | 8.745초 | 18.197초 |
| typed direct dispatch | 0 | 0.250초 | 0.224초 | 0.440초 | 0.711초 |

- 평균 절감: 4.012초, 94.1%
- 전체 ranked ID 완전 일치율: 1.000
- 문서 ID Recall: 1.000
- Top-1 일치율: 1.000
- 국가 precision: 1.000
- 명시 법령 25문항의 title Hit@20: 양쪽 0.480
- 명시 법령 MRR@20: 양쪽 0.414

여기서 Recall은 현재 결과를 pseudo-gold로 둔 회귀 지표이고, Hit/MRR은 질문에 명시된
법령명 정규화 문자열(4개 명시 alias 포함)을 사용한 자동 지표다. 법률 전문가의 정답
판정으로 해석하면 안 된다.

## 실험 2: hybrid 검색 hot path

코드 감사에서 번역·원문 검색이 같은 질의를 병렬 처리하면서 임베딩 API를 두 번
호출하고, 두 스레드가 같은 OpenSearch search pipeline을 매 요청 `PUT`하는 것을
확인했다. 50문항, 6개 변형을 회전 순서로 비교했다.

| 변형 | 평균 | p95 | 기준 대비 | Recall | Top-1 | 판단 |
|---|---:|---:|---:|---:|---:|---|
| 현재 검색 | 0.225초 | 0.436초 | — | 1.000 | 1.000 | 기준 |
| pipeline 준비 1회 | 0.210초 | 0.392초 | -6.5% | 1.000 | 1.000 | 보존형 |
| 임베딩 single-flight | 0.207초 | 0.439초 | -7.9% | 1.000 | 1.000 | 보존형 |
| 두 방법 결합 | 0.190초 | 0.368초 | -15.8% | 1.000 | 1.000 | 채택 |
| 원문 인덱스 제거 | 0.118초 | 0.200초 | -47.5% | 0.637 | 0.980 | 탈락 |
| vector k=40, size=24 | 0.198초 | 0.339초 | -12.2% | 0.696 | 0.820 | 탈락 |

보존형 결합은 전체 ranked ID 완전 일치율도 1.000이었다. 공격적 두 방법은 빨랐지만
검색 누락 때문에 채택하지 않았다.

## 실험 3: 리랭커 인증 상태

현재 소스는 리랭커 URL에 Authorization 헤더를 보내지 않아 실제 서버에서 HTTP 401을
받고 원래 hybrid 순서를 그대로 반환한다. 같은 25개 명시 법령 질문에 인증을 주입한
독립 실험 결과는 다음과 같다.

| 상태 | 평균 | p95 | Hit@20 | MRR@20 |
|---|---:|---:|---:|---:|
| 현재 401 fallback | 0.221초 | 0.375초 | 0.480 | 0.414 |
| 인증된 rerank | 0.290초 | 0.435초 | 0.480 | 0.480 |

인증된 리랭크는 맞은 12개 질문을 모두 1위로 올렸지만 평균 검색시간은 31.3% 늘었다.
이는 속도 최적화와 섞지 않은 별도의 품질 복구 항목이다. 키는 코드에 넣지 않고
환경변수로 주입해야 한다.

## 실험 4: 멀티에이전트 end-to-end

동일 Phi-4 endpoint를 쓰는 기준 서버(28010)와 별도 후보 서버(28011)를 띄워 행동형
5개, 명시 법령형 5개를 `/api/generate`로 교차 실행했다. 서버별 warm-up 1회는 제외했다.

| 변형 | 평균 | p50 | p95 |
|---|---:|---:|---:|
| 기준 | 11.356초 | 11.666초 | 15.673초 |
| typed dispatch + safe hot path | 6.486초 | 5.599초 | 10.840초 |

- 평균 지연 감소: 42.9%
- 문서 Recall: 0.975
- 전체 ranked ID 완전 일치: 0.800
- Top-1 일치: 0.900
- 답변 문자 bigram Jaccard: 0.718

검색 stage 단독에서는 결과가 완전히 같지만, end-to-end에서는 독립 실행된 상위 LLM의
검색어 준비와 답변 생성 변동이 포함된다. 10문항 결과는 연결 검증으로 보고, 최종
운영 판정에는 전체 50문항 이상과 동일 기준 반복 실행이 필요하다.

같은 기준 서버를 양쪽 arm으로 둔 10문항 반복 대조에서는 평균 지연 차이가 2.5%,
ranked ID/Recall/Top-1이 모두 1.000, 답변 bigram Jaccard가 0.832였다. 따라서 후보의
검색 Recall 0.975는 높지만 Top-1 0.900과 답변 Jaccard 0.718은 이 소규모 기준 반복보다
낮다. 검색 단계 채택 판단과 별개로, 전체 답변 파이프라인 배포는 보류하고 더 큰 반복
평가를 통과시켜야 한다.

## 적용 결정

1. `law_search_agent.run_search`의 LLM tool echo를 typed direct dispatch로 교체한 검색
   단계 후보를 채택한다.
2. hybrid pipeline 생성은 프로세스 단위 lock/once로 보호한다.
3. 번역·원문 검색이 동일한 query vector를 공유한다.
4. 원문 검색 제거와 k 축소는 적용하지 않는다.
5. 리랭커 인증 복구는 품질 변경으로 별도 feature flag/A-B 검증 후 적용한다.
6. 전체 `/api/generate` 배포는 50문항 이상 반복에서 Top-1 및 답변 회귀 gate를 다시
   통과할 때까지 보류한다.

## 재현 파일

- `scripts/evaluate_moleg_search_dispatch.py`
- `scripts/evaluate_moleg_search_hotpath.py`
- `scripts/evaluate_moleg_search_rerank.py`
- `scripts/evaluate_moleg_search_e2e.py`
- `experiments/results/moleg-search-20260903/`

모든 결과 JSON은 비밀값과 질문 원문을 저장하지 않는다.
