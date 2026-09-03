# 작업 및 대화 요약

작성일: 2026-09-04 (실험 수행일: 2026-09-03)

## 1. 요청 내용과 범위 확정

최초 요청은 다음과 같았다.

- `CNU-MultiAgent-RAG-Optimization` 저장소를 작업 폴더에 받는다.
- `main`뿐 아니라 다른 브랜치도 확인한다.
- LLM의 관찰 가능한 추론·도구 호출 과정을 분석해 불필요한 부분을 줄인다.
- 실제 GPU/vLLM 서버에서 시간과 정확도를 측정한다.

초기에는 일반 CoT 단축 실험을 검토했지만, 사용자가 실험 대상을
`2025-moleg-search`의 멀티에이전트 검색 성능 개선으로 명확히 정정했다. 이후 모든
본실험은 MOLEG 검색 기능을 대상으로 다시 설계했다. 정정 전 GSM8K 실험 브랜치와
생성 산출물은 최종 작업 브랜치에서 제거했다.

## 2. 저장소와 브랜치

- 로컬 저장소: `/data/project/choihm/CNU-MultiAgent-RAG-Optimization`
- 확인한 원격 브랜치:
  - `main`
  - `agent/generalize-multi-agent-rag`
- 최종 작업 브랜치: `experiment/moleg-search-performance`
- 실험 커밋: `46d1657`
- 원격 저장소에는 push하지 않았다.

`main`의 기본 최적화 코드와 일반화 브랜치의 typed dispatch, single-flight,
accuracy-first 회귀 평가 방식을 비교했다. 실제 MOLEG 실험은 일반화 브랜치를 기반으로
진행했다.

## 3. 실제 서버와 검색 구성 확인

GPU 서버에서 다음 환경을 확인했다.

- NVIDIA H200 NVL GPU 2장
- vLLM 0.19.0
- 검색 tool 모델: `microsoft/phi-4`
- 임베딩 모델: `BAAI/bge-m3`
- 리랭커: `dragonkue/bge-reranker-v2-m3-ko`
- 번역·원문 OpenSearch 인덱스를 이용한 hybrid 검색

`2025-moleg-search`가 사용하는 실제 검색 백엔드 소스는 서버의
`/data/project/vllm/fine-tune/2025-moleg-rag`에서 확인했다. 핵심 검색 경로는 다음과
같았다.

1. 상위 에이전트가 국가, 키워드, 검색문을 준비한다.
2. `law_search_agent`가 LLM을 호출해 `search_laws` 도구 선택을 다시 요청한다.
3. 검색 도구가 임베딩, OpenSearch 번역/원문 검색, 결과 병합, 리랭크를 수행한다.
4. 후속 에이전트가 검색 결과를 사용해 답변을 만든다.

운영 배포는 수정하지 않았다. 후보 구현은 다음 별도 복사본에서 실행했다.

`/data/project/vllm/fine-tune/experiments/2025-moleg-rag-typed-dispatch`

## 4. 추론 및 코드 분석 결과

### 불필요한 LLM tool echo

검색 단계에 들어올 때 `country`, `merged_keywords`, `search_terms`가 이미 확정돼
있었다. 그런데 기존 코드는 유일한 `search_laws` 도구와 해당 JSON을 Phi-4에 보내
“같은 인자로 반드시 호출하라”고 다시 판단시켰다.

50회 실측에서 LLM은 한 번도 tool call을 만들지 않았다. 대신 법률 설명 텍스트를
생성했고, 애플리케이션은 이 결과를 버린 다음 원래 인자로 직접 검색하는 폴백을
수행했다. 관찰 사례 한 건에서는 사용하지 않는 655자 설명을 생성하는 데 5.156초가
소요됐다.

이 분석은 모델의 비공개 내부 사고를 추측한 것이 아니라 애플리케이션에 반환된 출력,
tool trace, 실제 호출시간을 분석한 것이다.

### 검색 hot path 중복

검색 엔진에서는 다음 중복 작업을 확인했다.

- 번역·원문 인덱스가 같은 질의를 사용하면서 임베딩 API를 각각 호출함
- 두 검색 스레드가 같은 OpenSearch search pipeline을 매 요청마다 `PUT`함

이에 따라 동일 query vector 공유와 pipeline 1회 준비를 보존형 후보로 설계했다.

## 5. 구현한 후보

### Typed direct dispatch

이미 완성된 검색 인자를 LLM에 다시 전달하지 않고 유일한 `search_laws` handler에
직접 전달한다. 검색, 시소러스, 임베딩, 인덱스, 결과 포맷은 변경하지 않는다.

### Safe hybrid hot path

- OpenSearch pipeline 준비를 프로세스 단위 lock/once로 보호한다.
- 번역·원문 검색에 동일한 query vector를 전달한다.
- 임베딩 실패 시 기존 인덱스별 호출 경로가 재시도하도록 폴백한다.

### 공격적 대조군

속도와 정확도의 경계를 확인하기 위해 다음 방법도 측정했다.

- 원문 인덱스 검색 제거
- vector top-k와 request size 축소
- 인증된 리랭커 복구

## 6. 실험 결과

### 검색 dispatch 50문항

| 변형 | LLM 호출 | 평균 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|---:|
| 현재 LLM auto + direct fallback | 50 | 4.263초 | 3.370초 | 8.745초 | 18.197초 |
| Typed direct dispatch | 0 | 0.250초 | 0.224초 | 0.440초 | 0.711초 |

- 평균 지연 감소: 94.1%
- 평균 절감시간: 4.012초
- ranked ID 완전 일치율: 1.000
- 문서 ID Recall: 1.000
- Top-1 일치율: 1.000
- 국가 precision: 1.000
- 명시 법령 Hit@20: 양쪽 0.480
- 명시 법령 MRR@20: 양쪽 0.414

### Hybrid 검색 hot path 50문항

| 변형 | 평균 | 기준 대비 | Recall | Top-1 | 판단 |
|---|---:|---:|---:|---:|---|
| 현재 검색 | 0.225초 | 기준 | 1.000 | 1.000 | 기준 |
| Pipeline 준비 1회 | 0.210초 | -6.5% | 1.000 | 1.000 | 보존형 |
| 임베딩 single-flight | 0.207초 | -7.9% | 1.000 | 1.000 | 보존형 |
| 두 방법 결합 | 0.190초 | -15.8% | 1.000 | 1.000 | 채택 |
| 원문 인덱스 제거 | 0.118초 | -47.5% | 0.637 | 0.980 | 탈락 |
| vector k=40, size=24 | 0.198초 | -12.2% | 0.696 | 0.820 | 탈락 |

### 리랭커 상태 25문항

현재 소스는 리랭커 요청에 인증 헤더를 넣지 않아 HTTP 401을 받고 있었다.

| 변형 | 평균 | Hit@20 | MRR@20 |
|---|---:|---:|---:|
| 현재 401 fallback | 0.221초 | 0.480 | 0.414 |
| 인증된 rerank | 0.290초 | 0.480 | 0.480 |

인증 복구는 검색된 정답 법령의 순위를 개선했지만 평균 검색시간이 31.3% 늘었다.
따라서 exact-rank 보존형 속도 개선과 분리한 품질 개선 항목으로 남겼다.

### 멀티에이전트 end-to-end 10문항

기준 서버와 별도 후보 서버를 띄워 `/api/generate`를 교차 호출했다.

| 변형 | 평균 | p50 | p95 |
|---|---:|---:|---:|
| 기준 | 11.356초 | 11.666초 | 15.673초 |
| Typed dispatch + safe hot path | 6.486초 | 5.599초 | 10.840초 |

- 평균 지연 감소: 42.9%
- 문서 Recall: 0.975
- ranked ID 완전 일치율: 0.800
- Top-1 일치율: 0.900
- 답변 문자 bigram Jaccard: 0.718

동일한 기준 서버를 양쪽 arm으로 사용한 반복 대조에서는 문서 Recall과 Top-1이
1.000, 답변 Jaccard가 0.832였다. 후보의 end-to-end Top-1 0.900은 기존 0.95 gate에
못 미치므로 전체 서비스 적용은 보류했다.

## 7. 최종 판단

### 채택 후보

- 검색 단계의 LLM tool echo 제거
- 유일 도구에 대한 typed direct dispatch
- OpenSearch pipeline의 lock/once 준비
- 번역·원문 검색 query embedding 공유

### 탈락 또는 보류

- 원문 인덱스 제거: Recall 손실로 탈락
- 검색 후보 수 축소: Recall과 Top-1 손실로 탈락
- 인증 리랭커: 품질 개선 가능성이 있으나 지연 증가와 순위 변경 때문에 별도 A/B 필요
- 전체 `/api/generate` 배포: 50문항 이상 반복 평가 전까지 보류

검색 단계 후보는 50문항에서 정확한 순위 보존을 확인했지만, 이는 현재 결과를
pseudo-gold로 사용한 회귀 평가다. 법률 전문가가 판정한 절대 정확도로 해석할 수 없다.

## 8. 산출물과 검증

- 상세 보고서: `docs/MOLEG_SEARCH_COT_OPTIMIZATION_20260903.md`
- 적용 안내: `integrations/2025-moleg-search/README.md`
- 재현 스크립트: `scripts/evaluate_moleg_search_*.py`
- 원시 측정 결과: `experiments/results/moleg-search-20260903/`
- 테스트: 35개 전체 통과

실험용 서버는 종료했고 임시 네트워크 경로도 제거했다. 운영 vLLM과 운영 MOLEG
배포는 중단하거나 재시작하지 않았다.

## 9. 보안 관련 정리

대화에서 GitHub 토큰과 SSH 비밀번호가 제공됐으나 문서, 코드, 결과 JSON에는 값을
기록하지 않았다. 로컬 작업 중 임시로 복사된 환경설정 파일도 제거했다. 대화에 노출된
자격증명은 사용 후 교체하는 것이 권장된다.
