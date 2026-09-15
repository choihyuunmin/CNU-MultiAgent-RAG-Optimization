# 2026-09-08 실제 GPU 비교 — 중단된 캠페인

계획 768회 중 U16 첫 반복 A/B/C 192회 실행, SSE 191회 완료. C에서 240초 실패가
발생해 사전 규칙으로 중단했다. U32 및 둘째 반복은 미실행이다. `complete=false`를
완료로 고치거나 실패 행을 삭제하지 않는다.

- [결과와 병목 해석](../../../docs/WORKFLOW_GPU_RESULTS_20260908.md)
- `manifest.json`, `summary.json`, `requests.jsonl`: 계획·실행된 trial·원 요청 수치/해시.
- `workflow-analysis.json`: 실패 포함 기술 통계, 단계 work, 토큰 계측, 논리 예약.
- `failure-audit-v2.json`: 누락/중복 및 실패 질문의 원문 없는 비교. 초기 v1은 비공개 보존.
- `resources.json`: 각 trial 시간창의 host/NVML 관측. Nsight 계측 없음.
- `app-health.json`: 리랭커·가드레일 HTTP 상태와 내부 오류/반환 수.
- `answer_judge_summary.json`: 21문항×3조건, 자동 근거 평가. 전문가 정답 아님.
- `quality-gate.json`: 새 정책 배포 거절. 미완결·실패·전문가/안전 근거 부족.
- `calibration.json`, `split-plan.json`, `observe-final.json`, `budget-final.json`: 개발과 고정 정책.
- `preflight.json`, `provenance.json`: 소스 해시·모델 설정·운영 PID/health 확인.

정책 C 평균에는 240초 실패가 포함된다. 완전한 응답 완료시간 평균이 아니며, 자료의
bootstrap CI는 조기 중단·순서 미균형을 교정하지 않는다. 원문·인증·내부 endpoint는
이 폴더에 포함하지 않는다. CPU 합성 768회 산출물과 다른 실험이다.
