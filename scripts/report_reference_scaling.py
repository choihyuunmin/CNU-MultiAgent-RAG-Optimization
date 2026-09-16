"""Write the Korean result report from the audited public artifacts."""
import argparse
import json
from pathlib import Path


def number(value, decimals=2):
    return '—' if value is None else f'{value:.{decimals}f}'


def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();root=args.directory
    data=json.loads((root/'analysis.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    audit=json.loads((root/'audit.json').read_text())
    stages=json.loads((root/'stage-metrics.json').read_text())
    if not data['complete'] or data['n']!=audit['planned_requests']:
        raise ValueError('refuse to describe an unfinished study as complete')
    lines=['# 참조 ID 축약·복원 어댑터: 동시 요청 1~200 실측', '',
           '> **완료율 정정:** 이 보고서의 원래 완료 수치는 내부 실패를 일부 누락했다. [재검사·수정 지표](STAGE_OUTCOME_CORRECTION_20260916.md)를 함께 확인한다.', '',
           '## 측정한 기능', '',
           '**모델에는 긴 문서 ID 대신 `0`, `1` 같은 짧은 번호를 전달하고, 모델이 고른 번호를 실제 문서 ID로 복원한다.** '
           '문서 본문·제목·점수는 줄이지 않는다. 후보가 16개 이상일 때만 적용하고, ID가 다른 곳에도 쓰이거나 복원이 불가능하면 원래 입력을 사용한다.', '',
           '이번 비교의 개선 구성에는 worker의 낮은 reasoning 설정도 포함된다. 표의 개선율은 두 기능을 합친 효과다. '
           '새 ID 기능만의 추가 효과는 [이전 3조건 실험](STRUCTURED_HARNESS_RESULTS_20260915.md)에 따로 있다.', '',
           '## 실험 규모', '',
           f"- 동시 요청: {', '.join(map(str,plan['levels']))}. 대표 {len(plan['levels'])}개 지점이며 1~200의 모든 정수를 측정한 것은 아니다.",
           f"- 총 **{data['n']:,}요청**, 기존/개선 각 2회. 각 부하 안에서 질문·개수가 동일하고 비교 순서를 뒤집었다.",
           '- 조건당 질문 수는 `max(32, 동시 요청 수)`. 기존 200문항 회귀셋에서 종류별 비율을 맞춰 선택했다.',
           '- 요청 기한 240초. 실패도 평균·p95에 포함하며 성공 요청의 시간은 원자료에 별도로 제공한다.',
           '- 모델·서빙 설정을 유지한 독립 loopback 앱 실험이다. 실제 검색 소스 124개 파일을 다시 확인했다.',
           '- 검색 도구·설정은 같지만 모델의 검색 인자와 반환 후보는 실행마다 달라질 수 있다. ID 변환 자체는 받은 후보를 삭제하지 않는다.', '',
           '## 응답시간과 완료율', '',
           '각 조건의 두 반복을 합친 관측시간의 평균·p95다. 처리량은 기존 완료 판정(정정 전) 수를 두 반복의 전체 실행 시간 합으로 나눈 값이다.', '',
           '| 동시 요청 | 질문 수/회 | 평균 기존→개선(초) | 평균 단축 | p95 기존→개선(초) | 응답 완료율 기존→개선(정정 전) |',
           '|---:|---:|---:|---:|---:|---:|']
    for load in data['loads']:
        a,b=load['arms']['baseline'],load['arms']['improved']
        lines.append(f"| {load['concurrency']} | {load['unique_questions']} | {number(a['latency_including_failures']['mean'])}→{number(b['latency_including_failures']['mean'])} | {number(load['latency']['reduction_pct'])}% | {number(a['latency_including_failures']['p95'])}→{number(b['latency_including_failures']['p95'])} | {percent(a['pipeline_success']/a['n'])}→{percent(b['pipeline_success']/b['n'])} |")
    lines += ['', '단축률이 음수인 조건은 개선 구성이 더 느렸다는 뜻이다. 시간 초과의 240초는 실제 완료 시간이 아니라 관측을 끝낸 시점이다. '
              '시간 초과가 있는 조건의 평균·p95와 단축률은 이 관측 상한의 영향을 받으므로 완료율과 함께 읽어야 한다.', '',
              '### 200개 동시 요청의 반복별 결과', '',
              '| 반복 | 조건 | 정상 완료 | 평균 관측시간(초) | p95(초) |',
              '|---:|---|---:|---:|---:|']
    for trial in data['trials']:
        if trial['level']!=200:
            continue
        label={'baseline':'기존','improved':'개선'}[trial['arm']]
        times=trial['metrics']['elapsed_s']
        lines.append(f"| {trial['repeat']+1} | {label} | {trial['pipeline_success']}/{trial['n']} | {number(times['mean'])} | {number(times['p95'])} |")
    lines += ['', '실행 순서대로 표시했다. 공유 서버에서 두 번 측정한 결과이므로, 합산 평균뿐 아니라 반복 간 완료율·시간 변동도 확인한다.', '',
              '## 처리량과 통계 구간', '',
              '| 동시 요청 | 이전 판정 처리량 기존→개선(req/s) | 30초 이내 이전 판정 처리량 기존→개선(req/s) | 평균 단축률 95% 구간 |',
              '|---:|---:|---:|---:|']
    for load in data['loads']:
        a,b=load['arms']['baseline'],load['arms']['improved'];ci=load['latency']['reduction_95ci_pct']
        lines.append(f"| {load['concurrency']} | {number(a['throughput_rps'],3)}→{number(b['throughput_rps'],3)} | {number(a['goodput_30s_rps'],3)}→{number(b['goodput_30s_rps'],3)} | {number(ci[0])}–{number(ci[1])}% |")
    lines += ['', '질문별 반복 평균을 단위로 paired bootstrap 4,000회를 수행했다. 이 구간은 문항별 변동을 나타내며, '
              '두 번의 실행만으로 공유 대기열과 서버 상태의 실행 간 변동을 충분히 추정한 것은 아니다. 작은 표본의 구간과 p95를 정밀한 서버 용량 추정으로 해석하지 않는다. '
              '30초 이내 처리량은 내부 오류 없는 완료 기준이며 전문가 정답 기준의 처리량이 아니다.', '',
              '![동시 요청별 실측](../experiments/results/reference-scaling-20260916/scaling.png)', '',
              '## 검색 결과 변동', '',
              '| 동시 요청 | 개선의 기존 응답 대비 근거 ID 재현율 | 재현율 산정 쌍 | 기존 방식 자체 반복 재현율 | ID 목록 완전 일치 | 원천 법령 포함률 기존→개선 |',
              '|---:|---:|---:|---:|---:|---:|']
    for load in data['loads']:
        a,b=load['arms']['baseline'],load['arms']['improved']
        lines.append(f"| {load['concurrency']} | {percent(load['candidate_recall_vs_baseline'])} | {load['scorable_recall_pairs']}/{a['n']} | {percent(load['baseline_repeat_recall'])} | {percent(load['exact_evidence_fraction'])} | {percent(a['source_law_hit'])}→{percent(b['source_law_hit'])} |")
    lines += ['', '반환 근거 ID 재현율은 기존 응답을 기준으로 한 일치 정도다. 기존 응답이 정답이라는 뜻은 아니다. '
              '원천 법령 포함률은 라벨이 있는 질문의 known-item 지표이며 전문가 답변 정확도와 다르다. '
              '기존 응답에 법령이 없으면 재현율을 계산할 수 없어 그 쌍을 제외한다. 기존 응답에는 법령이 있는데 개선 응답이 비어 있으면 0점이다. '
              '고부하에서는 기존 방식의 실패로 산정 대상이 줄어들므로 재현율을 전체 요청의 정확도로 읽지 않는다. 완료율과 산정 쌍 수를 함께 제시한다.', '',
              '## 자동 답변 근거 평가', '',
              '성능 측정 종료 후 첫 반복의 참조 근거가 있는 질문을 별도 모델 `microsoft/phi-4`로 평가했다. 조건명은 평가 모델에 전달하지 않았다.', '',
              '| 동시 요청 | 유효 평가 기존/개선 | 관련성 기존→개선(0–2) | 근거 지지 기존→개선(0–2) | 근거 잘림 기존/개선 |',
              '|---:|---:|---:|---:|---:|']
    for level in [1,20,100,200]:
        path=root/f'judge-c{level}-summary.json'
        if not path.exists():
            lines.append(f'| {level} | 미실행 | — | — | — |');continue
        judge=json.loads(path.read_text());a,b=judge['variants']['baseline'],judge['variants']['improved']
        lines.append(f"| {level} | {a['valid']}/{a['n']} · {b['valid']}/{b['n']} | {number(a['mean_relevance_0_to_2'],3)}→{number(b['mean_relevance_0_to_2'],3)} | {number(a['mean_support_0_to_2'],3)}→{number(b['mean_support_0_to_2'],3)} | {a['evidence_truncated_count']}/{b['evidence_truncated_count']} |")
    lines += ['', '자동 평가·소표본이며 전문가 정확도 유지 또는 비열등성을 입증하지 않는다. 구간이 0을 포함한다는 이유로 동등하다고 판정하지 않는다. '
              '답변이 없는 요청은 0점으로 포함한다.', '']
    supplement=root/'judge-c200-repeat1-summary.json'
    if supplement.exists():
        judge=json.loads(supplement.read_text());a,b=judge['variants']['baseline'],judge['variants']['improved']
        lines += ['### 사후 확인: 200개 구간의 두 번째 반복', '',
                  '200개 구간에서 시간·완료율의 반복 차이와 첫 반복의 낮은 근거 점수를 확인한 뒤, '
                  '이미 저장된 두 번째 반복 답변을 같은 판정기로 추가 평가했다. 성능 요청을 새로 실행하지 않았으며, 위 사전 지정 평가를 대체하지 않는다.', '',
                  f"- 유효 평가: 기존 {a['valid']}/{a['n']}, 개선 {b['valid']}/{b['n']}. 근거 잘림: {a['evidence_truncated_count']}/{b['evidence_truncated_count']}건.",
                  f"- 관련성(0–2): {number(a['mean_relevance_0_to_2'],3)}→{number(b['mean_relevance_0_to_2'],3)}. 근거 지지(0–2): {number(a['mean_support_0_to_2'],3)}→{number(b['mean_support_0_to_2'],3)}.",
                  '- 두 반복의 방향과 점수 차이가 크다. 좋은 반복만 선택해 정확도 유지나 안정적인 200개 처리 성능으로 제시하지 않는다.', '',
                  '실행 스크립트: `scripts/evaluate_reference_scaling_repeat.py --level 200 --repeat 1`과 프로토콜의 디렉터리·운영 파일 경로 인자.', '']
    lines += ['## 어댑터 실행 기록', '']
    for arm,label in [('baseline','기존'),('improved','개선')]:
        stage=stages[arm];selection=stage['stages'].get('law_selection_response',{})
        lines.append(f"- {label}: 선택 호출 {selection.get('n',0):,}회, ID 축약 적용 {selection.get('short_id_calls',0):,}회, 복원 실패 재호출 {stage['fallbacks']:,}회. 선택 호출 평균 출력 토큰 {number(selection.get('mean_output_tokens'))}.")
    lines += ['', '전체 집계는 실제로 달라질 수 있는 검색 경로·후보 구성의 영향도 포함한다. 입력 토큰 감소나 특정 단계만의 인과 효과를 이 집계에서 단정하지 않는다.', '',
              '## 조정 모델의 캐시와 대기', '',
              '두 반복에서 관측한 KV 캐시 최대 점유와 최대 대기 요청 수, 선점 횟수 증가분의 합이다. '
              'KV 점유는 모델이 유지하는 토큰 캐시의 사용 비율이다. 공유 인스턴스 관측이므로 다른 요청도 수치에 기여할 수 있다.', '',
              '| 동시 요청 | KV 최대 기존→개선 | 최대 대기 요청 기존→개선 | 선점 증가 기존→개선 |',
              '|---:|---:|---:|---:|']
    def resource(level,arm):
        values=[t.get('telemetry',{}).get('orchestrator',{}) for t in data['trials'] if t['level']==level and t['arm']==arm]
        kv=[v.get('gauges',{}).get(key,{}).get('max') for v in values for key in ['kv_cache_usage_perc','gpu_cache_usage_perc']]
        waiting=[v.get('gauges',{}).get('num_requests_waiting',{}).get('max') for v in values]
        preemptions=[v.get('counters',{}).get('num_preemptions_total') for v in values]
        def maximum(xs):
            return max([x for x in xs if x is not None],default=None)
        return maximum(kv), maximum(waiting), sum(preemptions) if preemptions and all(x is not None for x in preemptions) else None
    for level in plan['levels']:
        a,b=resource(level,'baseline'),resource(level,'improved')
        lines.append(f'| {level} | {percent(a[0])}→{percent(b[0])} | {number(a[1],0)}→{number(b[1],0)} | {number(a[2],0)}→{number(b[2],0)} |')
    lines += ['',
              '## 관측 범위와 감사', '',
              '- 32명 이상은 한꺼번에 제출한 유한 요청 묶음이다. 완료가 늘수록 실제 진행 중 요청이 줄어든다. 장시간 유지되는 동시 사용자 용량을 보장하지 않는다.',
              '- 낮은 부하와 높은 부하의 문항 수가 다르다. 개선율은 같은 부하 안에서 비교하며, 부하 곡선을 순수한 동시성 효과만으로 해석하지 않는다.',
              '- 모델 서버는 공유한다. 외부 부하·prefix cache 변동을 완전히 제거한 전용 환경은 아니다.',
              f"- 전체 {audit['observed_requests']:,}건의 문항 쌍, 실행 코드 동결, 지정 동시성 도달을 확인했다. 모델 프로세스 전후 일치: {audit['model_inventory_unchanged']}.",
              '- 운영 서비스에 배포하지 않았다. 비공개 질문·응답과 인증 값은 공개 결과에서 제외했다.', '',
              '[측정 계획](REFERENCE_ID_SCALING_PROTOCOL_20260916.md) · [기능과 예시](REFERENCE_HARNESS_20260915.md) · '
              '[수치·설정·감사 자료](../experiments/results/reference-scaling-20260916/README.md)', '']
    completed=sum(arm['pipeline_success'] for load in data['loads'] for arm in load['arms'].values())
    faster=sum(load['latency']['reduction_pct']>0 for load in data['loads'])
    positive_interval=sum(load['latency']['reduction_95ci_pct'][0]>0 for load in data['loads'])
    highest=max(data['loads'],key=lambda load:load['concurrency'])
    a,b=highest['arms']['baseline'],highest['arms']['improved']
    lines[2:2]=['## 핵심 결과', '',
                f"- 전체 {data['n']:,}요청 중 기존 완료 판정(정정 전) **{completed:,}건**.",
                f"- 평균 응답시간은 {len(data['loads'])}개 부하 중 {faster}개에서 짧았다. 질문 단위 95% 구간의 하한이 0보다 큰 조건은 {positive_interval}개다.",
                f"- 최대 동시 요청 {highest['concurrency']}: 평균 **{number(a['latency_including_failures']['mean'])}→{number(b['latency_including_failures']['mean'])}초**, p95 **{number(a['latency_including_failures']['p95'])}→{number(b['latency_including_failures']['p95'])}초**.",
                '- 이 비교는 낮은 reasoning과 참조 ID 축약·복원을 함께 적용한 결과다. 전문가 정확도 유지와 지속적인 200명 처리 용량을 입증한 것은 아니다.', '']
    if highest['latency']['reduction_95ci_pct'][0]<=0<=highest['latency']['reduction_95ci_pct'][1]:
        lo,hi=highest['latency']['reduction_95ci_pct']
        lines.insert(7, f"- 최대 부하의 평균 단축률 95% 구간은 **{number(lo)}~{number(hi)}%**로 0을 포함한다. 이 구간의 지연 개선 근거는 불충분하다.")
    if highest['concurrency']==200:
        lines.insert(8, f"- 200개 구간의 기존 응답 대비 근거 ID 재현율은 {percent(highest['candidate_recall_vs_baseline'])}다. 완료율 상승만으로 품질이 유지됐다고 판단하지 않는다.")
    args.report.write_text('\n'.join(lines))


if __name__=='__main__':main()
