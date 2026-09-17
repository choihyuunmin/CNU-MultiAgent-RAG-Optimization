# dispatch-review-20260916 — 원본 / 미서명 직접 디스패치 / 서명 직접 디스패치, 3라운드 회전, 동시성 4

- 실행: 2026-09-16 (moleg k8s, 파드 3+1, 동결 이미지 `localhost/moleg-rag@sha256:3fb707…`), 1,800요청.
- `protocol.json`, `status.json`: 실행 프로토콜·셀 완료 기록(코드 해시 포함, 원시 텍스트 없음).
- `review-summary.json`: 실행 직후 표준 평가기(`analyze_review_trial.py`)의 셀·짝 비교 집계.
- `supplement.json`: 보완 분석(`docs/paper-draft-20260917/analysis/supplement.py`) — 원본-대-원본 무효 대조군, 절사평균·기하평균,
  위치 균형 추정치의 문항 부트스트랩 구간, 정확 집합·재현율 격차 구간, 엔진 카운터(래핑/네이티브 요청·토큰), 브랜치 검사,
  중간 종점(요청 시작→마지막 검색 완료).
- `self_repeat.json`: arm별 자기 반복 일치율 대 arm 간 일치율 — arm과 인스턴스가 교란됨을 보이는 근거.
- 원시 질문·응답·앱 로그·vLLM 카운터 원본은 서버(`~/cnu-review-20260916/`)에만 있다.
