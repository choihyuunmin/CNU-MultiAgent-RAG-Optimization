# 부하 스윕 확증 실험 (cnu-sweep-20260917)

## 목적
09-13/14 단일 패스 스윕(원본 대 서명 어댑터, C=1·2·5·10·20·50·100)은 C=20에서 5.4% 단축(구간 0 제외),
C=50에서 −8.6%(어댑터가 느림), C=100에서 +4.7%로 엇갈렸고, 고부하 두 셀 모두 **나중에 실행된 arm이 느렸다**
(순서 교란). 엔진 카운터는 워커 엔진 큐 시간이 원본 대비 어댑터에서 C=10/20/50/100에서 각각
448→133, 1125→400, 5394→1925, 8843→4261초(배치당)로 줄어드는 메커니즘을 보였지만, C≥50에서는
오케스트레이터 엔진(KV 100%, 선점, 큐 11,000–23,000초)이 양쪽 모두를 지배했다.
이 실험은 같은 수준을 **회전 순서·반복 라운드**로 다시 재어 순서 교란을 제거한다.

## 설계
- arm: `original`, `capability`(서명 직접 디스패치). `direct` 파드도 떠 있으므로 `--arms original,direct,capability`로 확장 가능.
- 수준: `--levels 8,16,32,64,100` (기본). 라운드: `--rounds 2` (원본→어댑터, 어댑터→원본). 시간이 되면 `--rounds 4`.
- 셀당 200문항, 닫힌 루프, 요청 타임아웃 600초(09-13/14 스윕과 동일), 앱 대기열 용량 100(`serve_sweep.py`가
  `GlobalWaitQueueMiddleware` 상한을 100으로, 환경변수 `MAX_CONCURRENT_GENERATE_REQUESTS=100`).
- 안전 정지: 엔진 텔레메트리 손실/대기 >256 30초, 최근 20건 중 4건 실패, 무진행, 셀 3,600초. 정지 시 전체 중단.
- 동결 이미지·모델·프롬프트·질문은 09-16 리뷰 실험과 동일. attachment는 출력 해시 필드가 추가된 버전.

예상 소요: 2 arm × 5 수준 × 2 라운드 = 20 배치, 배치당 7–12분 + 유휴 대기 → 약 3–3.5시간.
C=64/100 배치 동안 공유 GPU 스택에 높은 부하가 걸리므로 **야간 등 한산한 시간**에 실행한다.

## 실행 (moleg-app, axops, sudo 필요)
```sh
cd ~/cnu-sweep-20260917
mkdir -p ~/cnu-sweep-20260917-results
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f sweep.private.json
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf get pods -n moleg -l cnu-trial=cnu-sweep-20260917 -o wide   # 4개 Running 확인
sudo python3 launch_sweep_remote.py --folder ~/cnu-sweep-20260917                    # smoke → main → 로그 보존 → 수집 → 분석
#   예: sudo python3 launch_sweep_remote.py --folder ~/cnu-sweep-20260917 --levels 8,16,32 --rounds 4
```
결과: `~/cnu-sweep-20260917/main/level-XXX/round-N/…`, `*-application.log`, `sweep-summary.json`.

## 정리
```sh
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf delete pods,configmap -n moleg -l cnu-trial=cnu-sweep-20260917
```

## 분석
`analyze_sweep.py --root main --logs . --output sweep-summary.json` 은 수준·라운드별 짝 비교(평균·중앙값·부트스트랩 구간·
부호 검정), 라운드 동일 가중 평균, 처리량·SLO(60초) 달성률·실패, 재현율·정확 집합 일치, 엔진 카운터(워커 큐·대기·
오케스트레이터 KV·선점), 앱 로그의 경계 비-핸들러 시간을 출력한다. 워크스테이션의 `docs/paper-draft-20260917/analysis/`에
같은 스크립트가 있다.
