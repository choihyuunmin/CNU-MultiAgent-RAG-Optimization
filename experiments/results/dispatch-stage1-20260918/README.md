# dispatch-stage1-20260918 — 검증 프로토콜 1단계: 공개 구현(커밋 5e5c990)으로 원본 대 미서명 직접 실행, 동시성 4

- 실행: 2026-09-18 17:19–18:32 KST, moleg k8s(파드는 GPU 노드), 동결 이미지 `localhost/moleg-rag@sha256:3fb707…`.
  번들 `adapter.zip`은 커밋 5e5c990의 공개 패키지 `src/cnu_rag_optimization`에서 새로 만들었다(sha256 4bd45ebd…, `review-inventory.json`에 파일별 해시).
  직접 실행은 `typed_dispatch.try_typed_single_tool_dispatch`(validator 필수, 무손실 JSON 스냅샷) 경로다. 서명(capability) arm은 포함하지 않았다.
- 순서: 스모크 20문항 × {original, direct}(40요청, 모두 정상) → 라운드 1 original 200, direct 200 → 라운드 2 original 200(반복). 총 640요청, 전송 실패 0.
  타임아웃 600초, 유휴 게이트 엄격(실행 0·대기 0, 300초), 프로덕션 엔진 공유.
- `stage1-analysis.json`(`docs/paper-draft-20260917/analysis/stage1_analysis.py`): 셀 통계, 짝 지연(원본 r1 대 직접, 원본 r1 대 원본 r2 무효 대조군, 원본 r2 대 직접),
  중간 종점 E/S, 브랜치 경계 시간·검증, 분기 ID 기준 출력 해시 감사(`trace_equivalence`), 엔진 카운터, 단계별 LLM 호출 수·토큰.
- `stage1-trace-audit.json`: 커밋의 `analyze_followup.py`(request_finished 호출 목록 매핑·라운드별 arm 누락 허용 패치) 출력.
- `stage1-gate.json`: `performance_gate.evaluate_release_gate` 평가. 1단계 증거로는 통과하지 않는다(부하 1개, 라운드 1, 기준 미사전선언, 독립 품질 평가 없음).
- 원시 질문·응답·앱 로그는 서버(`~/cnu-stage1-20260918/`)에만 있다. 결과 문서: `docs/DISPATCH_STAGE1_RESULTS_20260918.md`.
