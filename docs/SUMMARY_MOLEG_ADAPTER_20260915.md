# 멀티에이전트 RAG 어댑터 개선 — 종합 정리 (2026-09-14 ~ 15)

원격 최신 하네스를 결합해 어댑터 성능을 개선하고, 실제 부하 실험으로 시간·정확도·
하드웨어 한계를 측정한 작업의 요약이다. 운영 배포·모델 서버 변경은 없었다(전 구간
읽기 전용, 모델 argv·포트 실험 전후 동일). 브랜치 `experiment/combined-harness-20260914`.

## 1. 목표

멀티에이전트 법령 검색(MOLEG)에서 **소요시간을 줄이고 정확도를 유지**하는 어댑터를,
원격 `agent/generalize-multi-agent-rag`의 최신 하네스와 기존 reasoning 제어를 결합해
만들고 실측한다. 공유 백엔드: 2×H200 NVL, 오케스트레이터 gemma-4-31B-it, worker gpt-oss-20b.

## 2. 시도한 방법과 채택 여부

| 방법 | 출처 | 결과 | 채택 |
|---|---|---|---|
| worker `reasoning_effort=low` + 답변 전 stall 종료 | 기존(09-11) | 답변 전 reasoning 1,939→164자, 첫 출력 2.39→0.35초 | **채택** |
| ProgramHarness 검색 겹치기(준비된 후속 즉시 실행) | 원격 신규 | 출력 안전하나 이 워크로드 기여 ~0(검색은 임계경로 아님) | 코드 보존, 미사용 |
| 온톨로지 검색 제거 | 사용자 허용 | 지연 이득 없음(이미 병렬 겹침), 재현율 불변 | 미채택(유지) |
| CompletionRouter·Coflow·NetworkHarness 스케줄링 | 원격 신규 | 오케 경합이 병목일 때 클라이언트 스케줄링 무이득(과거 부하 실험) | 코드 보존, 미사용 |

**결론**: 실질 효과는 worker reasoning 제거 한 가지에서 나온다. 새로 결합한 하네스·검색
겹치기·온톨로지 제거는 이 스택에서 이득이 없었다(정직한 음성 결과).

## 3. 성능 — 200문항 A/B (2026-09-14, 2,400요청)

질문 단위 paired bootstrap 지연 단축(모두 95% 구간이 0 제외):

| 동시 사용자 | baseline | combined | 단축률 | 95% 구간 |
|---:|---:|---:|---:|---:|
| 1 | 10.68초 | 8.61초 | **19.3%** | 12.5~29.0% |
| 4 | 13.99초 | 12.38초 | **11.5%** | 8.3~14.8% |
| 16 | 32.26초 | 30.52초 | **5.4%** | 1.5~9.9% |

정확도 유지: 소스 재현율 Δ≈0, 근거 ID 일치 0.94~0.97, blinded phi-4 판정 relevance
1.815→1.831·support 1.80→1.831(구간이 0 포함). combined 실패 0건.
[상세](COMBINED_HARNESS_RESULTS_20260914.md).

## 4. 하드웨어 한계 — 동시성 1→500 램프 (2026-09-15, 조건당 1회)

| 경계 | 동시성 | 근거 |
|---|---|---|
| SLO(30초) 유효 용량 무릎 | **~5–10** | 30초 goodput 최대 ~0.38 rps@C10, C20부터 급락 |
| 원시 처리량 포화 | **~50–100** | ~0.6–0.66 rps, 오케 KV C20부터 99–100%, C50부터 선점 |
| 경성 붕괴 | **300** | 성공률 40–44%, 240초 타임아웃 다수, 선점 ~130 → 자동 중단 |

재현율은 C≤100에서 ≥0.985 유지(C=300 급락은 타임아웃 부작용). 개선 조건은 C≤100 전
구간에서 지연이 같거나 짧고, 고부하에선 KV 포화가 지배해 이득이 잡음으로 수렴한다.
[상세](CAPACITY_RAMP_RESULTS_20260915.md).

## 5. 핵심 결론 (비자명 결과)

**앱 계층의 worker·검색 가속은 오케스트레이터의 KV·prefill 용량에 상한된다.** 단일
사용자에서 19% 단축은 실제이고 정확도도 안전하지만, 동시성이 오르면 오케스트레이터
selection(7.25초)·preparation(6.62초) prefill이 종단을 지배해 이득이 ~5%로 압축되고,
KV 포화(C≥20) 이후에는 사라진다. 실질 SLO 동시성은 ~10, 절대 처리 상한은 ~0.6 rps다.

## 6. 다음 단계

클라이언트 동시성 증가로는 상한을 넘을 수 없다. 실질 개선은 오케스트레이터 자체:
(1) selection/preparation prefill의 무손실 가속(추측 디코딩 — 09-06 단계 2.68배와 연결),
(2) 오케스트레이터 호출 수 감소(classify+preparation 병합 — 09-04에서 단계 28% 단축),
(3) KV 용량 증설·인스턴스 복제. 이들은 서버·엔진 변경이 필요하므로 별도 승인 실험 대상.

## 7. 산출물

- 어댑터 코드: `src/cnu_rag_optimization/`(program_harness, inference_overlap, completion,
  coflow, network_harness, application_adapter), `scripts/moleg_program_overlay.py`
- 실험 하네스: `scripts/{prepare,supervise,analyze}_moleg_combined*.py`,
  `scripts/supervise_moleg_sweep.py`, 런처 `--program-overlap`·`--no-ontology`
- 문서: `docs/COMBINED_HARNESS_{PROTOCOL,RESULTS}_20260914.md`,
  `docs/CAPACITY_RAMP_RESULTS_20260915.md`, 본 요약
- 공개 자료(수치·해시): `experiments/results/combined-20260914/`,
  `experiments/results/combined-sweep-20260915/`
- 오프라인 테스트 237개 통과. 원시 질문·답변은 서버 비공개 디렉터리에만 보존(git 제외).
