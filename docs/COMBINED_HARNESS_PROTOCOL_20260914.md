# 결합 하네스 어댑터 — 개선·실험 프로토콜 (2026-09-14)

원격 `agent/generalize-multi-agent-rag` 브랜치의 최신 하네스 모듈을 실험 인프라가
있는 브랜치로 통합하고, 이를 기존 reasoning 제어와 결합한 개선 어댑터를 만든 뒤
실제 200문항 실험으로 소요시간과 정확도를 함께 측정한다.

## 결합한 기술

- `ProgramHarness` (프로그램 의존성 하네스): 검색 노드에서 선행 작업이 끝난 후속
  작업(온톨로지 스코프 검색, 시나리오 distill)을 원래 `asyncio.gather` 장벽 이전에
  즉시 시작한다. 합류 지점은 같은 결과를 기다리고 원래 병합 코드를 그대로 실행한다.
  모델·프롬프트·검색 인자·반환 구조·vLLM 설정·timeout을 변경하지 않는다.
- worker(gpt-oss) `reasoning_effort=low` + 답변 전 reasoning 지연 종료(첫 유효 출력
  60초, reasoning 32,768자). 프록시 `extra_body.reasoning_effort` 경유, 프롬프트 불변.
- 즉시 SSE 전송(인위적 UI 지연 제거). 두 조건 공통이므로 개선율로 계산하지 않는다.
- (코드로 통합, 이번 실험 미사용) `CompletionRouter`·`CoflowAdmission`·`NetworkHarness`·
  `InferenceOverlapAdapter`. 앞선 부하 실험에서 오케스트레이터 경합이 병목일 때 클라이언트
  측 스케줄링이 이득을 주지 못했으므로 기본 실행 경로에는 넣지 않는다.

## 조건

| 조건 | 실제 변경 | 공통 |
|---|---|---|
| baseline | 계측만(관측 어댑터, 6단계 hook) | 호출 대기열 제거, 즉시 전송, 모델·검색·프롬프트 |
| combined | worker `reasoning_effort=low`+지연 종료 + ProgramHarness 검색 겹치기 | 위와 동일 |

두 조건은 동일 모델·검색·프롬프트·최대 출력 토큰을 사용한다. 온톨로지 겹치기가
검색 결과를 바꾸는지는 검색 소스 재현율과 근거 ID 일치율로 함께 검증한다.

## 데이터

기존 회귀셋(2026-09-09 재개 300문항)에서 kind 비례 층화로 **200문항**을 선정한다.
독립 전문가 holdout이 아니다. 과거 지연 수치와 합산하지 않는다. 참조 답변이 있는
문항은 blinded 판정에 사용한다.

## 실행

1. 프리플라이트: 8개 모델 프로세스 health, 검색 노드 SHA-256 지문, 겹치기 transform
   컴파일을 사전 확인한다. 지문 불일치 시 겹치기는 fail-closed.
2. 스모크: 8문항 × 2조건 U1(16요청). SSE 완료·내부 오류(LiteLLM `reasoning_effort`
   호환 포함)를 점검한 뒤에만 본 실험을 시작한다.
3. 본 실험: 200문항 × 2조건 × 사용자 {1,4,16} × 2반복 = **2,400요청**. (load, repeat)마다
   질문 순서를 두 조건에 동일하게, 조건 순서는 회전한다. 완료 또는 240초 기한까지의
   관측시간을 보존한다. 실패는 삭제·재시도하지 않으며 실패 발생 시 부하 상향을 멈춘다.
4. 판정: 첫 반복의 참조 답변 문항을 조건명을 숨긴 별도 판정 모델(phi-4)로 평가한다.

## 분석

같은 질문 반복을 묶은 paired bootstrap으로 조건별 평균·p95 지연 단축과 95% 구간을
보고한다. 단계별(분류·준비·검색·선택·답변) 오케스트레이터/워커/검색 기여를 분해해
가속의 상한(오케스트레이터 천장)을 정량화한다. 검색 소스 재현율, 근거 ID 일치율,
판정 relevance·support로 정확도 비열등을 확인한다. 성공률·재현율·판정이 나빠지면
단순 지연 감소를 개선으로 채택하지 않는다. 운영 배포는 하지 않는다.

## 산출물(코드)

- `src/cnu_rag_optimization/{program_harness,inference_overlap,completion,coflow,network_harness,application_adapter}.py`
- `scripts/moleg_program_overlay.py` — 검색 노드에 지문 검증 후 겹치기 부착.
- `integrations/2025-moleg-search/workflow-combined.json`, `…-baseline.json`
- `scripts/launch_moleg_scaling_app.py` `--program-overlap`
- `scripts/prepare_moleg_combined_experiment.py`, `scripts/supervise_moleg_combined.py`
