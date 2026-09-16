# 실제 검색기: 적응형 ID 하네스 실험

1,800요청, 100고유 문항, 3조건, 동시 요청 4/20/100, 2반복.
질문·응답 원문 및 인증 정보는 포함하지 않는다.

- `plan.json`: 고정 조건·문항 해시·실행 코드 해시.
- `analysis-vs-original.json`: 기존 방식 대비 개선.
- `analysis-vs-prior.json`: 이전 low-reasoning 설정 대비 새 기능 기여.
- `analysis-prior-vs-original.json`: 이전 설정 자체의 효과.
- `stage-metrics.json`: 모델 단계·입출력 토큰·복원 재시도 횟수.
- `replay-*.json`: 소규모 선택 실험, 본 실험과 합산하지 않음.
- `judge-c*-summary.json`: 성능 측정 이후 자동 근거 평가, 전문가 평가 아님.
- `configurations.json`: 실제 적용 설정.
- `model-inventory-*.json`: 모델 프로세스 전후 목록.
- `live-source-manifest.json`: 실행 중인 앱 소스 124개 파일의 해시.
- `audit.json`: 1,800쌍·코드 동결·소스 일치·모델 유지 확인 및 수치 파일 해시.
- `performance.png` / `performance.pdf`: 수치에서 생성한 그림. 최초 감사 해시 목록에는 그림이 포함되지 않음.

[해석과 한계](../../../docs/STRUCTURED_HARNESS_RESULTS_20260915.md).
