# 결합 하네스 어댑터 — 200문항 실험 결과 (2026-09-14)

원격 `agent/generalize-multi-agent-rag`의 최신 하네스를 실험 브랜치로 통합하고 기존
worker reasoning 제어와 결합한 개선 어댑터를, 200문항 실제 부하 실험으로 검증했다.
운영 배포·모델 서버 변경은 없다. 두 조건 모두 격리 loopback 앱으로 공유 백엔드를
읽기 전용으로 사용했으며, 실험 종료 후 모델 프로세스 argv·포트가 실험 전과 동일함을
확인했다(8개 vLLM + LiteLLM 유지).

## 실행 요약

- 200문항(kind 비례 층화, 참조 답변 65문항) × baseline/combined × 사용자 {1,4,16} ×
  2반복 = **2,400요청**. (load, repeat)마다 두 조건에 동일 질문 순서, 조건 순서는 회전.
- SSE 정상 완료 **2,398/2,400**. 실패 2건은 모두 baseline의 240초 tail timeout
  (U16 q271, U1 q113). combined 실패 0건. 실패는 삭제·재시도 없이 지연 통계에 포함.
- `combined` = worker `reasoning_effort=low`(답변 전 reasoning 지연 종료 포함) +
  ProgramHarness 검색 겹치기 + 즉시 전송. baseline = 계측만.

## 지연 (질문 단위 paired bootstrap, 반복 평균)

| 사용자 | baseline 평균 | combined 평균 | 단축률 | 95% 구간 | p95(base→comb) | 30초 내 성공률(base→comb) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.68초 | 8.61초 | **19.3%** | 12.5 ~ 29.0% | 17.5 → 15.1초 | 0.993 → 1.000 |
| 4 | 13.99초 | 12.38초 | **11.5%** | 8.3 ~ 14.8% | 24.8 → 23.4초 | 0.975 → 1.000 |
| 16 | 32.26초 | 30.52초 | **5.4%** | 1.5 ~ 9.9% | 56.0 → 59.4초 | 0.460 → 0.520 |

세 부하 모두 95% 구간이 0을 포함하지 않는 유의한 단축이다. 단축률은 부하가 커질수록
19→11.5→5.4%로 단조 감소한다. U16의 p95는 combined가 소폭 높다(꼬리 지연은 오케스트
레이터 경합이 지배하며 두 조건이 유사, 표본 변동 포함). 30초 내 성공률은 모든 부하에서
combined가 같거나 높다.

## 정확도 — 비열등 확인

- **검색 소스 재현율(silver known-item)**: 변화 없음. law 재현율 delta U1 +0.004
  (95% 구간 [0, 0.0125]), U4·U16 0.000. 두 조건의 근거 ID 집합 완전 일치율은
  U1 0.967 / U4 0.935 / U16 0.952이며, 불일치분은 공유 엔진의 비결정성 범위다.
- **blinded 판정(phi-4, 참조 답변 65문항, 조건명 숨김)**: relevance
  1.815→1.831(Δ +0.015, 95% [−0.031, +0.062]), support 1.80→1.831(Δ +0.031,
  95% [−0.046, +0.108]), insufficient_evidence 0.092(양쪽 동일). 두 지표 모두
  구간이 0을 포함해 저하 없음(자동 silver 지표이며 전문가 정확도는 아니다).
- 요청 성공률은 combined가 baseline 이상(combined 실패 0).

즉 combined는 모든 부하에서 유의하게 지연을 줄이면서 소스 재현율·근거 충실성·성공률을
저하시키지 않았다.

## 메커니즘과 상한(오케스트레이터 천장)

- combined의 이득 원천은 worker의 답변 전 reasoning 제거다. 관측된 worker reasoning
  문자수 평균 **1,939 → 164**, worker 첫 가시 출력 시각 **2.39초 → 0.35초**. 단일
  사용자 종단 단축(10.68→8.61초, 약 2.07초)의 대부분이 이 구간이다.
- 그러나 부하가 커지면 오케스트레이터(gemma) 단계가 종단을 지배한다. combined에서
  관측한 단계별 RPC 평균: **selection 7.25초(p95 22.9)**, **preparation 6.62초
  (p95 18.4)**, classification 1.69초 — worker answer는 1.24초, 검색 도구는 0.45초.
  worker에서 아낄 수 있는 약 2초는 오케스트레이터 prepare+select 벽 앞에서 상대적으로
  작아지고, 그래서 U16 이득이 약 5%로 압축된다.
- ProgramHarness 검색 겹치기는 이 워크로드에서 사실상 0 기여였다(온톨로지/시나리오
  후속 노드가 조기 반환·비활성, 검색 RPC 0.45초). 시작 시점만 앞당기는 안전한 추가일
  뿐 별도 가속 근거로 제시하지 않는다.

**정량화된 경계(비자명 결과):** 앱 계층의 worker·검색 가속은 오케스트레이터의
prepare+select prefill 비용에 의해 상한된다. 이 비용은 동시성에 따라 커지고 앱 계층
하네스로는 넘을 수 없다. 다음 실질 레버는 worker나 검색이 아니라 오케스트레이터
selection/preparation prefill의 무손실 가속(추측 디코딩·호출 수 감소)이다.
[[MOLEG_SPECDEC_RESULTS_20260906]]의 오케스트레이터 단계 2.68배 결과와 연결된다.

## 산출물

- 코드: `docs/COMBINED_HARNESS_PROTOCOL_20260914.md`, `src/cnu_rag_optimization/`
  (program_harness, inference_overlap, completion, coflow, network_harness,
  application_adapter), `scripts/moleg_program_overlay.py`,
  `scripts/{prepare,supervise,analyze}_moleg_combined*.py`,
  `integrations/2025-moleg-search/workflow-combined*.json`.
- 공개 자료(수치·해시): `experiments/results/combined-20260914/`.
- 원시 응답·질문·답변 원문은 서버의 비공개 실험 디렉터리에만 보존(git 제외).

## 한계

기존 회귀 데이터의 200문항 부분집합이며 독립 전문가 holdout이 아니다. 자동 판정은
전문가 정확도 동등성을 증명하지 않는다. 단계별 RPC 합은 관측 구간이며 배타적 GPU
계산시간이 아니다. 반복 2회·공유 엔진 변동이 있어 절대치는 시간대 효과를 포함할 수
있다. 과거 실험 수치와 합산하지 않는다.
