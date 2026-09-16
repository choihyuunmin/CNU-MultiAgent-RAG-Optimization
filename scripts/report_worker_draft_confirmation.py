"""Render the completed 32-question confirmation without merging pilot controls."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    load = lambda name: json.loads((root / name).read_text())
    analysis, audit, calls, decision = (load(name) for name in
        ['analysis.json', 'stage-outcomes.json', 'call-comparison.json', 'decision.json'])
    if not analysis['complete'] or not audit['complete'] or audit['audit_errors']:
        raise ValueError('only completed and audited studies')
    lines = ['# Worker 추측 디코딩: 별도 32문항 확대 검증', '',
        f"실제 전체 검색 **{analysis['n']}건**, 동시 4/16/32, 무추측/4토큰, 각 3반복이다.",
        '앞선 단계·종단 예비 및 워밍업 23문항을 제외하고 새 32문항을 고정했다.',
        '27개국의 법령명 질문 14개, 조문 주제 8개, 대체 조문 주제 4개, 원문 바꿔쓰기 6개다.',
        '실행 순서는 기본→4토큰 / 4토큰→기본 / 기본→4토큰이다.', '',
        '## 실패 포함 응답시간과 완료', '',
        '| 동시 요청 | 조건 | 평균(초) | p95(초) | 단계·의존 기능 오류 없는 완료 | 오류 없는 처리량(req/s) |',
        '|---:|---|---:|---:|---:|---:|']
    for row in audit['loads']:
        for arm, label in [('baseline', '무추측'), ('improved', '4토큰')]:
            v = row['arms'][arm]
            t = v['latency_including_failures']
            lines.append(f"| {row['level']} | {label} | {t['mean']:.2f} | {t['p95']:.2f} | "
                         f"{v['dependency_clean_completion']}/{v['n']} | {v['dependency_clean_throughput_rps']:.3f} |")
    lines += ['', '오류 없는 완료는 답변의 전문가 정확도가 아니다. 실패도 관측 시간에 포함했다.', '',
              '## 단축률과 검색 근거', '',
              '| 동시 요청 | 평균 단축률 | 질문별 95% 구간 | 근거 ID 재현율 | 기본 자체 반복 재현율 | 원천 법령 포함률 기본→4토큰 |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in analysis['loads']:
        t = row['latency']; b, c = row['arms']['baseline'], row['arms']['improved']
        lo, hi = t['reduction_95ci_pct']
        lines.append(f"| {row['concurrency']} | {t['reduction_pct']:.2f}% | {lo:.2f}~{hi:.2f}% | "
                     f"{100*row['candidate_recall_vs_baseline']:.2f}% | {100*row['baseline_repeat_recall']:.2f}% | "
                     f"{100*b['source_law_hit']:.2f}%→{100*c['source_law_hit']:.2f}% |")
    lines += ['', '질문별 3회 평균을 단위로 bootstrap 4,000회를 수행했다. 공유 서버의 시간대 변동을',
              '충분히 추정한 것은 아니다. 근거 ID 일치는 정확도의 대체 지표가 아니다.', '',
              '## 호출 수 확인', '',
              '| 동시 요청 | 비교 쌍 | 단계별 모델 호출 시도 수 일치 | 검색 실행 횟수 일치 | 준비 출력 일치 |',
              '|---:|---:|---:|---:|---:|']
    for row in calls['loads']:
        lines.append(f"| {row['level']} | {row['paired_requests']} | {row['same_stage_call_counts']} | "
                     f"{row['same_retrieval_execution_counts']} | {row['same_preparation_outputs']} |")
    lines += ['', '시스템 프롬프트·생성 옵션·검색 후보 설정은 양쪽 동일하다. 중간 모델 출력이 달라져',
              '실제 후속 입력은 달라질 수 있다. 최종 답변의 바이트 단위 동일성을',
              '가정하지 않는다. 기존 입력 해시는 로그 ID까지 포함하므로 프롬프트 동일성 검증에 쓰지 않는다.', '']
    if (root / 'worker-telemetry.json').exists():
        telemetry = load('worker-telemetry.json')
        lines += ['## 모델 서버 구간 계측', '',
                  '실험 전용 worker의 워밍업 제외 누적 히스토그램 차이다. 각 평균의 단위는',
                  '**검색 요청이 아닌 모델 호출 하나**다. 순수 GPU 커널 시간이나 동적 비용 정책의 보정값은 아니다.', '',
                  '| 동시 요청 | 서버 완료 호출 수 기본/4토큰 | 큐 대기(초) 기본→4토큰 | 디코딩(초) 기본→4토큰 | 생성 토큰 기본→4토큰 |',
                  '|---:|---:|---:|---:|---:|']
        for row in telemetry['loads']:
            b, c = (row['arms'][arm]['histograms'] for arm in ['baseline', 'improved'])
            def pair(name):
                return f"{b[name]['mean_per_model_call']:.2f}→{c[name]['mean_per_model_call']:.2f}"
            lines.append(f"| {row['level']} | {b['request_generation_tokens']['count']:.0f}/{c['request_generation_tokens']['count']:.0f} | "
                         f"{pair('request_queue_time_seconds')} | {pair('request_decode_time_seconds')} | {pair('request_generation_tokens')} |")
        lines += ['', '앞 표는 호출 시도, 이 표는 서버 완료 지표다. C4의 접속 실패 1건은 서버 완료 수에 포함되지 않는다.', '']
    lines += [
              '## 자동 근거 평가', '',
              '측정 전에 지정한 참조 답변 보유 14문항의 C32 응답을 세 반복에서 평가했다.',
              '조건명을 제공하지 않은 단일 Phi-4 평가이며 전문가 정답 검증은 아니다.', '',
              '| 반복 | 조건 | 유효/전체 | 관련성(0~2) | 근거 지지(0~2) | 근거 잘림 |',
              '|---:|---|---:|---:|---:|---:|']
    for file in sorted(root.glob('judge-r*-summary.json')):
        data = json.loads(file.read_text())
        for arm, label in [('baseline', '무추측'), ('improved', '4토큰')]:
            v = data['variants'][arm]
            lines.append(f"| {data['repeat']+1} | {label} | {v['valid']}/{v['n']} | "
                         f"{v['mean_relevance_0_to_2']:.2f} | {v['mean_support_0_to_2']:.2f} | {v['evidence_truncated_count']} |")
    lines += ['', '## 승격 조건 점검', '']
    labels = {'mean_gain_at_least_5pct': '평균 5% 이상 단축', 'paired_ci_lower_positive': '95% 구간 하한 양수',
              'p95_regression_at_most_2pct': 'p95 악화 2% 이하', 'all_dependencies_clean': '모든 의존 기능 정상',
              'clean_goodput_no_regression': '오류 없는 처리량 유지', 'same_observed_call_counts': '모델 호출 수 일치',
              'same_observed_retrieval_counts': '검색 실행 횟수 일치', 'known_item_no_regression': '원천 법령 포함률 유지',
              'evidence_recall_at_least_baseline_repeat': '근거 변화가 기본 반복 변동 이하'}
    for row in decision['gates']:
        failed = [labels[k] for k, passed in row['checks'].items() if not passed]
        lines.append(f"- C{row['level']}: " + ('모든 점검 통과.' if not failed else '미충족 — ' + ', '.join(failed) + '.'))
    lines += ['', '**판정: 속도 개선은 확인했지만, 새 방법의 64~200개 부하 본 실험에는 승격하지 않는다.**',
              'C32 근거 재현율은 97.97%로 기본 자체 반복 99.84%보다 낮다. 이것만으로 답변 정확도',
              '저하를 단정할 수는 없지만, 이번 보수적 승격 조건은 충족하지 못했다.', '',
              '## 실패·근거 변동의 추가 확인', '',
              '- C4 개선 조건의 `q226`에서 `APIConnectionError` 1건을 확인했다. 기존 대체 경로가',
              '  응답을 반환했지만 정상 완료로 세지 않았다. 서버가 완료한 worker 호출도 이 조건만 1회 적다.',
              '- 하위 통신 원인은 기존 동결 로그에 없다. 추측 디코딩이 원인이라고 단정하지 않는다.',
              '- 후속 코드에는 예외 종류·원인 체인·OS 오류 번호를 남기도록 보완했다. 메시지·URL·키는',
              '  기록하지 않고 재시도도 추가하지 않았다. 이 진단 변경은 이번 측정에 적용하지 않았으며,',
              '  통신 실패 자체가 해결됐다는 뜻도 아니다.',
              '- 근거가 다른 쌍은 C4 2개, C16 4개, C32 6개다. 이 중 각각 2/3/5개에서 준비 출력과',
              '  검색 후보 순서도 달랐다. 이는 관측된 연관이며, 변화의 원인을 확정하는 분석은 아니다.',
              '- 다음 확인에서는 해당 질문의 모델 입력을 고정한 재생과 실제 종단 반복을 나눠',
              '  worker 출력 변화와 앞선 준비·선택 단계의 변동을 구분하고, 반환 근거를 전문가 정답과 비교해야 한다.', '',
              '## 해석 범위', '',
              '- 고정 k=4의 기존 n-gram 방식이며 새 적응형 정책의 성능 증거가 아니다.',
              '- 동일한 격리 서빙 설정끼리 비교했다. 운영 worker와 GPU 위치·컴파일·문맥 한도가 다르다.',
              '- 모델 호출 제한을 새로 추가하지 않았다. 양쪽 엔진의 기존 실험 설정 `max_num_seqs=4`는 같다.',
              '- 32문항 탐색 결과를 모든 멀티에이전트 구조·200명 지속 사용·전문가 정확도로 일반화하지 않는다.',
              '- 모델 서비스 8개의 기존 PID와 정상 응답을 확인했다. 실험 서버·앱 포트는 닫혔고 GPU 메모리가 시작 상태로 돌아왔다.',
              '- 측정 코드·실행기 해시는 끝까지 같았다. 후속 진단 보완을 포함한 코드 테스트는 339개 통과했다.',
              '', '[기능·첫 예비 결과](WORKER_DRAFT_RESULTS_20260916.md)',
              '[공개 원자료](../experiments/results/draft-budget-20260916/confirmation32-k4/analysis.json)', '']
    args.output.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
