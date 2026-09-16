# 기능 보완 실험 자료

현재 회수·검증을 마친 새 실험은 **1,064요청**이다. 200개 동시 요청 3조건·2반복의
1,200요청 추가 실험은 서버에 독립 실행했으나, 내부망 접속 시간 초과로 전체 결과를
아직 회수하지 못했다. 마지막 확인 상태는 `unretrieved-main-status.json`에 기록한다.
이 폴더의 완료된 자료를 추가 실험의 최종 결과로 해석하지 않는다.

| 폴더 | 내용 | 새 요청 수 |
|---|---|---:|
| `pilot-auth` | 재정렬 인증 오류 상태 vs 정상, C4/32, 2회 | 256 |
| `pilot-dispatch` | 인증 정상, 제한 없음 vs 선착순 16 vs 후속 우선 16, C4/64, 2회 | 408 |
| `pilot-window32` | 인증 정상, 제한 없음 vs 후속 우선 32, C100, 2회 | 400 |
| `prior-correction` | 이미 공개한 2,896건의 단계 실패 재검사. 새 실험이 아님 | — |

## 읽는 순서

- `stage-outcomes.json`: 추적 기록으로 검증한 단계·외부 기능 상태와 응답시간.
- `audited-requests.jsonl`: 요청별 새 판정. 추적이 없는 완료는 성공으로 추정하지 않는다.
- `requests.jsonl`, `summary.json`: 기존 수집기의 원시 수치와 응답 완료 판정. `pipeline_ok`는 내부 실패를 놓칠 수 있으므로 새 성공률로 쓰지 않는다.
- `evidence-comparison.json`: 반환 근거 ID의 대조군 일치 정도와 대조군 자체 반복 변동. 정확도 지표가 아니다.
- `completion-comparison.json`(있는 경우): 질문별 반복을 묶은 완료율 차이와 bootstrap 구간.
- `plan.json`, `audit.json`: 조건, 문항 해시, 코드·설정 동결과 감사. 인증 예비 계획의 설정 해시 누락은 감사 파일에 명시했다.
- `artifact-sha256.json`: 공개 파일 무결성. 질문·응답 원문과 인증 값은 포함하지 않는다.

C100은 평균 88.97→81.82초(8.04% 감소)였지만 단축률 95% 구간 −1.45~16.01%가
0을 포함한다. 두 조건 모두 내부·외부 기능 오류 없이 200/200건 완료했고 원천 법령
포함률은 각각 116/120이었다. 전문가 정확도 유지나 지속적인 사용자 용량의 증거는 아니다.

![완료한 비교와 미회수 상태](comparisons.png)

## 재계산

```bash
python scripts/summarize_moleg_continuation_evidence.py --directory PATH_TO_ONE_STUDY
python scripts/summarize_moleg_completion.py --directory PATH_TO_ONE_STUDY
```

[구현과 적용 범위](../../../docs/CONTINUATION_HARNESS_20260916.md) ·
[실험 방법](../../../docs/CONTINUATION_PROTOCOL_20260916.md) ·
[이전 완료율 정정](../../../docs/STAGE_OUTCOME_CORRECTION_20260916.md)
