# 호출 대기열 제거와 답변 전 reasoning 제어

2026-09-11. 현재 코드의 기본 실행은 일반 async 모델 호출이다. HTTP·생성·모델 예산
대기열을 없애고, 워커의 답변 전 reasoning 장기화에 직접 대응한다.

## 확인한 병목과 코드 원인

1. [300문항 실측](WORKFLOW300_RESULTS_20260910.md)에서 U32의 고정16 대비 예산 정책은
   평균 53.19→60.19초로 느려졌다. KV 압력 감소는 지연 감소를 보장하지 않았다.
2. [실패 요청 q191](WORKFLOW_GPU_RESULTS_20260908.md)의 워커는 196.62초 동안
   reasoning 126,142자를 생성하고 답변은 내지 못했다. 이는 서빙 API의 관측값이며
   숨겨진 사고 내용의 반복이나 GPU 계산 시간만을 분리한 증거는 아니다.
3. 보관된 앱의 `infra/llm/client.py`는 전달할 인자를 수동 선별하면서
   `reasoning_effort`를 누락했다. 이 저장소의 `moleg_paper_runtime.direct_kwargs`는
   `max_tokens`, `seed`, `stop` 등도 누락했다. 생성 옵션이 서버에 도달하지 않는 문제다.
4. 보관된 앱의 스트리밍 함수는 `finally`에서 시간만 기록하고 SDK 스트림을 닫지 않았다.
   상위 iterator를 닫는 것만으로 실제 HTTP 응답까지 닫는다고 보장할 수 없었다.
5. 첫 청크는 reasoning일 수 있다. 첫 청크 지연과 첫 답변 지연을 분리하고, reasoning이
   계속 들어오는 동안에도 갱신되지 않는 첫 유효 출력 기한이 필요하다.

## 변경

- `WorkflowAdapter`에서 `ModelAdmission`, `ModelBudget`, FIFO 대기·독점 실행·예약량
  기반 실행 제한을 삭제했다. `call`은 바로 `await`, `stream`은 바로 iterator를 소비한다.
  토큰 추정치는 계측으로만 남는다. 추정치 누락도 요청을 직렬화하지 않는다.
- 앱 launcher는 `GlobalWaitQueueMiddleware`를 제거하고 생성 전역 제한기를 활성 수
  계측으로 교체한다. 같은 세션의 상태 갱신 직렬화와 인증 middleware는 유지한다.
  `--max-pipelines`, `--max-http`, 어댑터 `mode=budget`은 더 이상 지원하지 않는다.
- SDK의 실제 함수 signature로 생성 인자를 전달한다. 기존 프록시·모델 경로를 재사용하며
  입력 messages·tools를 변경하지 않는다. 정상 종료·예외·취소 모두 실제 SDK 스트림을 닫는다.
- 기본 [설정](../integrations/2025-moleg-search/workflow-reasoning.json)은 현재 모델 목록의
  `worker agent`(gpt-oss-20b)에 낮은 추론량을 요청한다. 개별 호출에 명시된 추론량은 우선한다.
  기존 프록시가 전달하는 `extra_body.reasoning_effort`로 정규화한다.
  실제 서버 점검에서 발견한 LiteLLM 호환성 문제는 프록시 요청에
  `allowed_openai_params=["reasoning_effort"]`를 추가해 해결한다. 옵션을 버리거나
  운영 프록시 설정을 변경하지 않는다.
  gpt-oss의 effort 설정은 [vLLM 공식 recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md)를 참고했다.
- 첫 답변 또는 도구 호출 전 **60초**, 또는 **reasoning 32,768자 초과**를 감지하면
  `ReasoningStallError`로 종료한다. 이 수치는 운영 실측으로 최적화된 값이 아닌 수정 가능한
  공학적 기본값이다. 문자를 토큰으로 환산하지 않는다. 첫 도구 호출도 유효 출력으로 처리한다.
  이 기한은 모델 호출 시작부터 측정하며 서버 대기·네트워크 시간도 포함한다.
- 빈 답변·토큰 한도 종료·완료 신호 없는 스트림은 `IncompleteGenerationError`로 전파한다.
  이미 전송된 답변을 재생하거나 자동 재시도하지 않는다. 상위 앱의 기존 오류 응답 경로가
  이를 처리한다. 종료 처리는 정상 답변을 생성하는 대체 수단이 아니다.
- 첫 청크·첫 reasoning·첫 답변·첫 도구 호출 시간, 문자 수, 정상 완료·잘림·장기화 여부를
  원문 없이 기록한다. usage가 제공되지 않으면 reasoning 토큰은 미관측으로 남긴다.

`reasoning_effort=low`도 유한한 토큰 상한이나 정확도 보존을 보장하지 않는다.
오케스트레이터는 별도 reasoning이 관측되지 않은 단계가 있으므로 같은 정책을 적용하지 않는다.
출력 필드를 숨기는 것과 추론 계산량을 줄이는 것은 구분한다.

## 실행과 이전 실험

기존 앱 Python 환경에서 다음과 같이 실행한다. 기본값은 일반 프록시 호출,
즉시 전송, reasoning 설정, 호출 상한 없음이다. 인증정보는 기존 파일에서 읽는다.

```bash
python scripts/launch_moleg_scaling_app.py \
  --app-root /operator/app-source \
  --app-env /operator/app.env \
  --serving-env /operator/serving.env \
  --proxy-config /operator/proxy.yaml \
  --trace /operator/new-run/model.jsonl
```

순수 관측 대조는 `--adapter-config integrations/2025-moleg-search/workflow-observe.json`을
지정한다. 이 경우에도 호출 대기열은 없다. 모델 옵션 전달·종료 처리는 동일하므로
낮은 추론량·장기화 제어의 추가 영향을 비교할 수 있다. 기본 워크플로 trace는
위 예에서 `model.workflow.jsonl`이며 모델 trace와 별도 파일이다.

과거 상한 실험·예산 동결용 스크립트와 결과는 역사적 자료다. 옛 budget 설정이나 숫자
`slots` manifest를 현재 launcher에 넣으면 실패한다. 이전 실험을 재현하려면 당시 동결
코드/커밋을 사용한다. 부하 생성기의 사용자 수와 선택적 과거 실험 모듈의 상한은
현재 앱의 호출 실행 경로에 연결되지 않는다. vLLM 자체의 GPU scheduler는 변경하지 않았다.

## 검증 범위

CPU 테스트는 64개 요청의 동시 진입, 미상·대형 토큰 추정치의 비차단 실행,
생성 옵션 전달, 실제 SDK stream close, 취소 전파, 첫 답변 기한, 정상 답변·도구 호출,
빈 출력·잘림 처리, reasoning 원문 비기록을 검증했다. 기존 테스트를 포함해 **149개가 통과**했다.
보관된 실제 앱 `GenerateQueue` 소스를 사용하는 추가 호환성 검사에서도 64개 세션의 즉시
진입, 같은 세션의 직렬화, 종료 후 상태 정리를 확인했다.

현재 변경의 GPU 종단 성능과 답변 품질은 새로 측정하지 않았다. 앞선 실험의 수치를
이 변경의 성능 개선율로 사용하지 않는다. 특히 낮은 추론량 적용 후 정답·근거 충실성의
동등성은 아직 미검증이다. 이 작업은 저장소 코드 수정이며 운영 배포는 수행하지 않았다.
