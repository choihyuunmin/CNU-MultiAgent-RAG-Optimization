# 후속 실험: 인스턴스 효과 분리 + 핸들러 출력 해시 (cnu-review-20260916 후속, 2026-09-17)

## 목적
1. 09-16 3라운드 실험에서 arm 간 법령 집합 일치율(89–92%)이 각 arm의 자기 반복 일치율(95%)보다 낮고,
   같은 메커니즘인 unsigned/signed 사이도 91%였다. 각 arm이 파드 하나씩이었으므로 **arm 효과와
   인스턴스 효과가 분리되지 않는다.** 같은 arm을 파드 두 개(original-a/b, capability-a/b)로 돌려
   "같은 arm·다른 인스턴스" 일치율을 직접 잰다.
2. 핸들러 출력·다음 단계 객체 해시(`output_hash`, `result_hash`, `law_ids_hash`)를 기록해
   준비 인자가 같은 브랜치의 검색 결과가 arm/인스턴스 사이에 동일한지 확인한다.

동결 이미지(digest 동일), 모델·프롬프트·질문·자원 제한 모두 09-16과 같다. 바뀐 것은 attachment의
해시 필드 추가와 파드 수뿐이다. 프로덕션·모델 서버는 건드리지 않는다.

## 실행 (moleg-app, axops 계정, sudo 필요)
```sh
cd ~/cnu-review-followup-20260917
mkdir -p ~/cnu-review-followup-20260917-results            # client 파드 hostPath
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f followup.private.json
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf get pods -n moleg -l cnu-trial=cnu-review-followup-20260917 -o wide
#   다섯 파드가 모두 Running(앱 파드 readiness 통과)이 될 때까지 기다린다.
sudo python3 launch_followup_remote.py --folder ~/cnu-review-followup-20260917            # 1라운드: 800요청, 약 85분
#   두 번째 라운드를 역순으로 더 돌리려면: --rounds 2  (1,600요청, 약 170분)
```
launcher는 smoke(질문 1개 × 4 arm) → 본 실행 → 앱 로그 보존(`*-application.log`) → `/results/main` 복사 →
`analyze_followup.py` 순으로 진행하고 `followup-summary.json`을 남긴다.

## 정리
```sh
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf delete pods,configmap -n moleg -l cnu-trial=cnu-review-followup-20260917
```

## 결과 해석
- 같은 방식 인스턴스 간 차이와 다른 방식 간 차이를 비교한다. 1회 순차 실행만으로
  인스턴스/시간/방식 효과를 분리하거나 디스패치의 무영향을 입증할 수 없다.
- 2026-09-18 수정: 이벤트를 분기 ID로 연결한다. 이전 요약기는 병렬 완료 순서가
  뒤집히면 입력/출력을 잘못 연결했다. 과거 조건부 해시 통계는 원로그 재분석 전 사용하지 않는다.
- `hashes.audit`: 누락·실패는 `unverifiable`, 동일 인자 다중 분기는
  `ambiguous_repeated_inputs`로 분리한다. 이를 불일치나 일치로 대신하지 않는다.
- 준비 인자 동일·출력 차이는 검색 이후 경로를 조사할 근거지만, 인덱스/설정/시간을
  통제하지 않은 상태에서 핸들러 자체의 비결정성으로 단정하지 않는다.
- 수치는 워크스테이션의 `docs/paper-draft-20260917/analysis/` 스크립트와 같은 규약(중복 제거 법령 ID,
  nonempty 참조 재현율)으로 계산된다.

## 파일
- `followup.private.json` — ConfigMap + 파드 5개 (09-16 매니페스트에서 `make_followup_manifest.py`로 생성)
- `review_dispatch_attachment.py` — 해시 필드가 추가된 attachment (오프라인 단위 시험 통과)
- `run_followup_trial.py` — 임의 arm 이름·순서를 받는 클라이언트 러너 (안전 점검 동일)
- `launch_followup_remote.py` — 감독 스크립트
- `analyze_followup.py` — 표준 라이브러리만 쓰는 요약기

새 검증 계획: [전체 응답 시간·품질 동시 검증](../../DISPATCH_VALIDATION_PROTOCOL_20260918.md).
요약기는 이제 공개 패키지의 `trace_equivalence`를 사용한다. `pip install -e .`로
패키지를 준비한다. `run_followup_trial.py --concurrency N --timeout 600`으로 부하와
타임아웃을 명시할 수 있으며 기본값(4, 180초)은 과거 실행 재현용으로 유지한다.
