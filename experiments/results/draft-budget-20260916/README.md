# 고정-k 실제 GPU·검색 확대 실험

- `pilot-k4/`: 완료한 도구 호출 96건. `rows.jsonl`은 질문·입출력 해시와 시간·토큰 수만 포함한다.
  `blocks.jsonl`은 부하별 경과시간과 실제 모델 계수다. `summary.json`은 실패 포함 집계다.
- `e2e-k4/`: 별도 8문항의 종단 비교 64건과 자동 근거 판정 16건. 단계 오류·호출 수·근거 ID를 함께 공개한다.
  `runner.py`는 이 첫 확인에 사용한 실행기의 동결 사본이다.
- `confirmation32-k4/`: 27개국 별도 32문항, 동시 4/16/32, 3반복 576건. 자동 근거 판정 84건.
  `decision.json`은 승격 조건 미충족을 기록하고, `failure-audit.json`과 `evidence-path-audit.json`은 원인 확인의 범위를 명시한다.
- `cleanup-audit.json`: 기존 모델 8개 정상 응답, 실험 포트 종료, 메모리 복구, 측정 코드 불변 확인.
- `publication-audit.json`: 전체 요청·판정 수와 공개 파일 검증.
- `isolation-audit.json`: 역할별 실제 모델과 격리 서빙 설정의 확인 범위.
- `runtime-recovered.json`: 연결 복구 후 실제 GPU 서버의 버전·소스 해시 점검.
- `preflight*.json`: 연결 장애 당시의 기록으로 보존한다. 여기에 나온 로컬 GPU는 실험 GPU가 아니다.
- 질문 본문, 준비 출력, 모델 응답, 서버 로그, 인증정보는 공개하지 않는다.

실행 순서, 같은 조건의 대조군, 적용 범위와 한계는 [결과 보고서](../../../docs/WORKER_DRAFT_RESULTS_20260916.md)에 있다.

재집계:

```bash
.venv/bin/python scripts/summarize_worker_draft_pilot.py \
  --directory experiments/results/draft-budget-20260916/pilot-k4
```

[최종 확대 결과와 승격 보류 판단](../../../docs/WORKER_DRAFT_CONFIRMATION_20260916.md).

공개한 요청 수는 도구 호출 재생 96건과 전체 검색 640건이다. 서로 다른 단위이므로 하나의
종단 성능 표본으로 합치지 않는다. 자동 평가는 100판정이다. 준비·워밍업은 별도다.
