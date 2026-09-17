# dispatch-sweep-exploratory-20260914 — 원본 / 서명 직접 디스패치 단일 패스 동시성 스윕

- 실행: 2026-09-13/14. `cnu-scale-main`(C=1,2 완료, 4 중단), `cnu-scale-revised`(C=5,10,20 완료, 50에서 안전 정지),
  `cnu-high-main`(C=50,100, 개정 고부하 안전 정책). 동결 이미지 동일, 앱 대기열 상한 100, 타임아웃 600초, 수준마다 arm 순서 교대.
- `*/protocol.json`, `*/status.json`: 프로토콜과 셀 완료 기록.
- `sweep_existing.json`, `sweep_existing.txt`: 짝 비교(평균·구간·부호 검정), 처리량, 60초 SLO 달성률, 기준 일치도, 셀별 엔진
  카운터(워커/오케스트레이터 요청·큐 시간·최대 대기·KV·선점). 스크립트: `docs/paper-draft-20260917/analysis/sweep_existing.py`.
- 단일 패스이며 C≥50에서 먼저 실행된 arm이 빨라 순서 교란이 있다. 확증 실험은 `dispatch-sweep-20260918`.
- 원시 응답·카운터 원본(82 MB)은 서버(`~/cnu-scale-evidence-20260914/`, `~/cnu-high-evidence-20260914/`)에만 있다.
