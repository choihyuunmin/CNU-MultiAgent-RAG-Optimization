# 2026-09-07 접속 복구 후 실험 산출물

- `factorial/`: 선택용 U16 A/B/C/D/E, 320/320 완료. `selection.json`은 새 검증 전에 E를 고정한 기록이다.
- `validation-counterbalanced/`: U16·32 A/E 각 2회, 512/512 완료. 주 효과 추정은 이 폴더의 `aggregate.json`이다.
- 각 주 대조의 `audit.json`: 질문·조건·반복·해시·시간 합계 감사. 검증은 manifest에 저장한 실행 순서도 확인한다.
- 각 주 대조의 `resources-audited.json`: UTC 구간에 맞춘 host/앱/Nsight 통계와 표본 범위·간격 점검.
- `factorial/resources.json`, `host-interim.json`: 더 이른 집계. 범위 감사에는 `resources-audited.json`을 사용한다.
- `smoke-live-config/`: 현재 pod 설정을 사용한 예비 4회. 주 효과에 합산하지 않는다.
- `smoke-config-excluded/`: 예전 설정의 리랭커 401 fallback으로 제외한 예비 4회. HTTP 200만으로 정상 파이프라인을 판단하지 않은 근거다.
- `validation-excluded/`: 순서 보정 전에 중단한 캠페인. 기준 64행과 후보 부분 3행을 보존하며 주 효과에 합산하지 않는다. 전체 시도 횟수는 67보다 많다.
- `environment.json`: 진행 중의 현재 pod 소스 해시·장비·스크립트 상태. 실행별 runner 해시는 각 manifest가 기준이며 이후 순서 수정은 새 manifest에 반영돼 있다.
- `environment-final.json`: 모델 PID·시작 시점·health, 종료 상태, 비공개 Nsight 원본의 해시, 전력 제한 카운터의 부분 구간 관측.
- `host_resources.jsonl`: 추론 호스트 숫자 지표. 장치 전체 값이지 요청별 귀속 값이 아니다.
- `app-health.json`: 비공개 실행 trace에서 허용한 상태·오류 개수만 추출한 결과. 원문·세션·엔드포인트는 포함하지 않는다.

## 내부 trace 그룹과 캠페인 대응

그룹 번호는 해당 앱에서 trial별 세션 접두사가 처음 관측된 순서이며 0부터 시작한다.
모든 주 대조 그룹은 index 0~63을 정확히 한 번씩 포함하고 execute가 64회 반환했다.

| 캠페인/trial | 앱 | trace 그룹 번호 |
|---|---|---:|
| 현재 설정 smoke | baseline / emit | 각각 0 |
| 선택용 A / B | baseline / emit | 각각 1 |
| 선택용 C / D / E | slots8 / emit8 / emit16 | 각각 0 |
| 제외 검증 기준 / 부분 후보 | baseline / emit16 | 2 / 1 |
| 검증 0: U16 기준, 반복 0 | baseline | 3 |
| 검증 1: U16 개선, 반복 0 | emit16 | 2 |
| 검증 2: U32 개선, 반복 0 | emit16 | 3 |
| 검증 3: U32 기준, 반복 0 | baseline | 4 |
| 검증 4: U32 기준, 반복 1 | baseline | 5 |
| 검증 5: U32 개선, 반복 1 | emit16 | 4 |
| 검증 6: U16 개선, 반복 1 | emit16 | 5 |
| 검증 7: U16 기준, 반복 1 | baseline | 6 |

제외 후보 그룹은 19개 실행 trace 중 3개만 execute가 반환했다. 취소된 요청을 정상 완료로 세지 않는다.
주 대조에는 선택용 320회·검증 512회만 포함한다. 원본 app trace와 Nsight 파일은 서버의
보호된 실험 디렉터리에 남겼으며, 인증 설정 임시 사본 3개와 실험 프로세스는 정리했다.

공개 그림은 `factorial/ablation.pdf`와 `validation-counterbalanced/scaling.pdf`다.
`server_times.pdf`의 모델 지표는 공유 서버의 호출당 값이며 API 단계의 인과 분해가 아니다.
질문·답변 정확성, 전문가 품질 동등성, 100명 이상 일반화는 산출물 감사의 범위 밖이다.
