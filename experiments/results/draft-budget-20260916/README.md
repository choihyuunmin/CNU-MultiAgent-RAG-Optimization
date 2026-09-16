# 고정-k 실제 GPU 예비 실험

- `pilot-k4/`: 완료한 도구 호출 96건. `rows.jsonl`은 질문·입출력 해시와 시간·토큰 수만 포함한다.
  `blocks.jsonl`은 부하별 경과시간과 실제 모델 계수다. `summary.json`은 실패 포함 집계다.
- `e2e-k4/`: 별도 8문항의 종단 비교 64건과 자동 근거 판정 16건. 단계 오류·호출 수·근거 ID를 함께 공개한다.
  `runner.py`는 이 첫 확인에 사용한 실행기의 동결 사본이다.
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
