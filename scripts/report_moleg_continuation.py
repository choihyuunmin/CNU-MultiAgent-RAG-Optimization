"""Render completed trace-validated experiments without treating SSE as accuracy."""
import argparse
import json
from pathlib import Path


def fmt(x):return '—' if x is None else f'{x:.2f}'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.directory
    studies=[('pilot-auth','인증 보정',{'baseline':'인증 오류','improved':'인증 정상'}),
             ('pilot-dispatch','실행 창 16',{'baseline':'제한 없음','fifo':'선착순 16','improved':'후속 우선 16'}),
             ('pilot-window32','실행 창 32',{'baseline':'제한 없음','improved':'후속 우선 32'}),
             ('validation200','200개 검증',{'baseline':'제한 없음','fifo':'선착순 32','improved':'후속 우선 32'})]
    data={name:json.loads((root/name/'stage-outcomes.json').read_text()) for name,_,_ in studies}
    if any(not v['complete'] or v['audit_errors'] for v in data.values()):raise ValueError('unfinished or invalid study')
    total=sum(v['n'] for v in data.values())
    lines=['# 기능 보완과 고부하 재실험', '',
           '## 바꾼 기능', '',
           '1. 재정렬 인증을 복구하고 시작 전 실제 점수 반환을 검사한다. 통신 오류·비정상 점수를 별도로 기록한다.',
           '2. 모델 호출의 입장 대기와 120초 실행 제한을 분리한다. 사용자 전체 240초 기한에는 둘 다 포함한다.',
           '3. 완료한 모델 호출 수를 기준으로 후속 호출에 우선권을 준다. 오래 기다린 호출은 선착순으로 우선 처리한다.',
           '4. 오류 안내문을 성공으로 세지 않도록 응답과 준비·선택·모델 호출 기록을 연결한다.', '',
           '[기능·연결 예시](CONTINUATION_HARNESS_20260916.md) · [측정 방법](CONTINUATION_PROTOCOL_20260916.md)', '',
           '## 실험 규모와 비교 조건', '',
           f'실제 검색기 **{total:,}요청**을 단계적으로 실행했다. 각 조건은 2회이며 반복에서 순서를 뒤집었다. '
           '모든 조건은 낮은 worker reasoning과 참조 ID 축약을 사용한다. 인증 비교만 인증 상태가 다르고, 실행 제어 비교는 모두 인증이 정상이다.', '',
           '16개 제한은 C64에서 효과가 없어 채택하지 않았고 32개 제한을 시험했다. C4의 4문항 실행은 연결 확인용이며 기능 효과 근거로 쓰지 않는다.', '',
           '## 응답시간·오류 없는 완료', '',
           '아래 완료는 **관측된 내부 단계·외부 기능 오류가 없는 응답**이다. 전문가 답변 정확도는 아니다. '
           '평균·p95에는 모든 실패와 240초 시간 초과를 포함한다. 오류 안내문을 빨리 반환하면 평균이 낮아질 수 있으므로 완료율·처리량을 함께 본다.', '',
           '| 실험 | 동시 요청 | 조건 | 요청 수 | 평균(초) | p95(초) | 오류 없는 완료 | 오류 없는 처리량(req/s) |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for name,label,names in studies:
        for load in data[name]['loads']:
            if name=='pilot-dispatch' and load['level']==4:continue
            for arm,display in names.items():
                v=load['arms'][arm];lat=v['latency_including_failures']
                lines.append(f"| {label} | {load['level']} | {display} | {v['n']} | {fmt(lat['mean'])} | {fmt(lat['p95'])} | {v['dependency_clean_completion']}/{v['n']} ({pct(v['dependency_clean_completion']/v['n'])}) | {v['dependency_clean_throughput_rps']:.3f} |")
    lines += ['', '인증 오류 조건에서는 정상 검색 순서로 대체 응답을 반환해도 외부 기능 오류로 분류한다. '
              '재정렬을 호출하지 않은 경로는 이 오류가 없다. 인증 보정의 효과를 후속 호출 우선순위의 효과로 합산하지 않는다.', '',
              '![조건별 측정](../experiments/results/continuation-20260916/comparisons.png)', '',
              '## 평균 관측시간 변화의 불확실성', '',
              '| 실험 | 동시 요청 | 대조군 대비 조건 | 평균 단축률 | 질문 단위 95% 구간 |',
              '|---|---:|---|---:|---:|']
    for name,label,names in studies:
        for load in data[name]['loads']:
            if name=='pilot-dispatch' and load['level']==4:continue
            for arm,v in load['paired_latency_vs_baseline'].items():
                ci=v['reduction_95ci_pct']
                lines.append(f"| {label} | {load['level']} | {names[arm]} | {fmt(v['reduction_pct'])}% | {fmt(ci[0])}~{fmt(ci[1])}% |")
    lines += ['', '음수는 느려졌다는 뜻이다. 질문별 반복 평균을 단위로 bootstrap 4,000회를 계산했다. '
              '두 번의 실행으로 공유 서버의 실행 간 변동을 충분히 추정한 것은 아니다. 0을 포함하는 구간에서 확정적인 속도 개선을 주장하지 않는다.', '',
              '## 200개 조건의 반복별 결과', '',
              '| 반복 | 조건 | 응답 완료 판정(기존 방식) | 오류 없는 완료 | 평균(초) | p95(초) |',
              '|---:|---|---:|---:|---:|---:|']
    rows=[json.loads(line) for line in (root/'validation200/audited-requests.jsonl').read_text().splitlines()]
    trials=json.loads((root/'validation200/summary.json').read_text())['trials']
    names=studies[-1][2]
    for trial in trials:
        sub=[r for r in rows if r['trial']==trial['trial']]
        lines.append(f"| {trial['repeat']+1} | {names[trial['arm']]} | {trial['pipeline_success']}/{trial['n']} | {sum(r['dependency_clean_completion'] for r in sub)}/{len(sub)} | {fmt(trial['metrics']['elapsed_s']['mean'])} | {fmt(trial['metrics']['elapsed_s']['p95'])} |")
    lines += ['', '## 검색 결과의 변화', '',
              '| 실험 | 동시 요청 | 조건 | 대조 응답 대비 근거 ID 재현율 | 산정 쌍 | 대조군 자체 반복 재현율 | 원천 법령 포함률 대조→조건 |',
              '|---|---:|---|---:|---:|---:|---:|']
    for name,label,names in studies:
        evidence=json.loads((root/name/'evidence-comparison.json').read_text())
        for arm,loads in evidence['comparisons'].items():
            for item in loads:
                level=item['concurrency']
                if name=='pilot-dispatch' and level==4:continue
                load=next(x for x in data[name]['loads'] if x['level']==level)
                before,after=load['arms']['baseline'],load['arms'][arm]
                lines.append(f"| {label} | {level} | {names[arm]} | {pct(item['candidate_recall_vs_baseline'])} | {item['scorable_recall_pairs']}/{before['n']} | {pct(item['baseline_repeat_recall'])} | {pct(before['source_law_hit']['mean'])}→{pct(after['source_law_hit']['mean'])} |")
    lines += ['', '근거 ID 재현율은 반환 문서 `item_id`/`id`의 일치 정도이며 상위 법령 ID로 묶은 지표가 아니다. '
              '대조 응답이 정답이라는 뜻도 아니다. 기준 근거가 비어 있는 쌍은 산정에서 제외하고, 기준이 있는데 비교 응답이 비어 있으면 0점이다. '
              '원천 법령 포함률은 라벨이 있는 질문의 known-item 지표다. '
              '재정렬 인증을 복구하면 실제 순위·후보가 바뀌므로 이전 응답과의 일치만으로 답변 정확도를 판단하지 않는다.', '',
              '## 자동 근거 평가', '',
              '성능 측정 종료 뒤 별도 모델 `microsoft/phi-4`로 평가했다. 조건명은 모델에 제공하지 않았다. '
              'C100·C200 표본은 결과 확인 전 지정한 근거 질문 16개이며 두 반복을 모두 평가했다. '
              '인증 보정 C32 평가는 사후 검토로, 해당 예비셋의 근거 질문 10개 전부를 두 반복에서 평가했다.', '',
              '| 실험 | 동시 요청 | 반복 | 조건 | 유효/전체 | 관련성(0~2) | 근거 지지(0~2) | 근거 잘림 |',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    judgments=0
    for name,label,names in [studies[0],studies[2],studies[3]]:
        for path in sorted((root/name).glob('judge-c*-repeat*-summary.json')):
            j=json.loads(path.read_text());judgments+=j['completed']
            if j['completed']!=j['expected']:raise ValueError('unfinished judge')
            level=int(path.name.split('-')[1][1:])
            for arm,display in names.items():
                v=j['variants'][arm]
                lines.append(f"| {label} | {level} | {j['repeat']+1} | {display} | {v['valid']}/{v['n']} | {fmt(v['mean_relevance_0_to_2'])} | {fmt(v['mean_support_0_to_2'])} | {v['evidence_truncated_count']} |")
    if judgments!=200:raise ValueError('expected 160 preregistered and 40 posthoc judgments')
    lines += ['', f'자동 평가 총 {judgments}건이다. 평가 모델의 오류와 표본 한계가 있어 전문가 정확도·비열등성의 증거로 쓰지 않는다. '
              '판정별 비교 구간은 공개 요약 JSON에 있다.', '',
              '## 기존 결과 정정과 재현 범위', '',
              '- 이전 2,896건에서 숨겨진 준비·선택 단계 실패와 재정렬 401을 확인했다. 원본 측정 JSON은 보존하고 [정정 결과](STAGE_OUTCOME_CORRECTION_20260916.md)를 별도로 공개했다.',
              '- 공통 실행 제어는 순차·병렬 분기·동적 반복에 연결할 수 있지만 하나의 프로세스·이벤트 루프 범위다. 분산 배포에는 공통 호출 경계가 추가로 필요하다.',
              '- C≥32는 유한 동시 제출 묶음이다. 지속적인 200명 사용자 용량이나 모든 멀티에이전트 구조의 성능 향상을 보장하지 않는다.',
              '- 운영 소스 124개 파일을 확인했다. 독립 loopback 앱에서 실험했고 모델 프로세스 전후 일치를 검사했다. 운영 배포는 변경하지 않았다.',
              '- 공개 결과의 `stage-outcomes.json`이 새 완료 판정이다. 원자료의 `pipeline_ok`와 `summary.json` 완료 수는 기존 응답 문구 기반 값으로 보존했다.',
              '- 완료 응답은 추적 누락 없이 연결했다. 시간 초과로 응답 세션 ID를 받지 못한 요청은 실패로 유지하며, 해당 요청의 세부 실패 원인 집계는 불완전할 수 있다.', '',
              '[수치·해시·설정·감사 자료](../experiments/results/continuation-20260916/README.md)', '']
    a.output.write_text('\n'.join(lines))

if __name__=='__main__':main()
