"""Audit a completed owned-worker comparison and export only public numeric data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from analyze_moleg_structured import analyze as analyze_evidence
from audit_moleg_stage_outcomes import analyze as audit_stages
from summarize_worker_draft_calls import analyze as analyze_calls


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--app-root', type=Path, required=True)
    parser.add_argument('--app-env', type=Path, required=True)
    parser.add_argument('--serving-env', type=Path, required=True)
    parser.add_argument('--proxy-config', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.directory.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if json.loads((root / 'status.json').read_text())['phase'] != 'complete':
        raise ValueError('timing not complete')
    audit, audited = audit_stages(root)
    if audit['n'] != plan['expected_requests'] or audit['audit_errors']:
        raise ValueError('incomplete timing or trace audit')
    cases = {c['case_id'] for c in plan['cases']}
    keys = [(r['case_id'], r['arm'], r['repeat'], r['level']) for r in audited]
    expected = {(c, arm, rep, level) for c in cases for arm in ['baseline', 'improved']
                for rep in range(plan['repeats']) for level in plan['levels']}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('paired design coverage mismatch')
    write(root / 'stage-outcomes.json', audit)
    (root / 'audited-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audited))
    evidence = analyze_evidence(root)
    write(root / 'analysis.json', evidence)
    calls, records = analyze_calls(root)
    write(root / 'call-comparison.json', calls)
    (root / 'call-records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))

    # Timing is complete before any judge model is called.
    judge_plan = json.loads((root / 'judge-plan.json').read_text()) if (root / 'judge-plan.json').exists() else None
    judge_count = 0
    if judge_plan:
        from dotenv import dotenv_values
        env = dict(os.environ)
        env.update({k: v for k, v in dotenv_values(args.app_env).items() if v is not None})
        env.update(MOLEG_RAG_ROOT=str(args.app_root), MOLEG_SERVING_ENV=str(args.serving_env),
                   MOLEG_PROXY_CONFIG=str(args.proxy_config))
        all_cases = json.loads((root / 'cases.json').read_text())
        selected_cases = [c for c in all_cases if c['case_id'] in judge_plan['cases']]
        raw = [json.loads(x) for x in (root / 'private-responses.jsonl').read_text().splitlines()]
        for repeat in judge_plan['repeats']:
            dest = root / f'judge-r{repeat}'
            dest.mkdir(exist_ok=True, mode=0o700)
            selected = [r for r in raw if r['repeat'] == repeat and r['level'] == judge_plan['level']
                        and r['case_id'] in judge_plan['cases']]
            write(dest / 'cases.json', selected_cases)
            (dest / 'e2e_stream.jsonl').write_text(''.join(json.dumps({
                'case_id': r['case_id'], 'variant': r['arm'], 'repeat': repeat,
                'result': r.get('result')}, ensure_ascii=False) + '\n' for r in selected))
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_moleg_answer_judge.py')),
                '--directory', str(dest), '--judge-url', 'http://127.0.0.1:8002/v1',
                '--judge-model', judge_plan['judge'], '--repeat', str(repeat)], env=env, check=True)
            result = json.loads((dest / 'answer_judge_summary.json').read_text())
            judge_count += result['completed']
            if result['completed'] != result['expected']:
                raise ValueError('judge incomplete')
            write(root / f'judge-r{repeat}-summary.json', result)
        if judge_count != judge_plan['expected']:
            raise ValueError('judge design coverage mismatch')

    gates = []
    for load in evidence['loads']:
        level = load['concurrency']
        health = next(x for x in audit['loads'] if x['level'] == level)
        call = next(x for x in calls['loads'] if x['level'] == level)
        b, c = load['arms']['baseline'], load['arms']['improved']
        tests = {
            'mean_gain_at_least_5pct': load['latency']['reduction_pct'] >= 5,
            'paired_ci_lower_positive': load['latency']['reduction_95ci_pct'][0] > 0,
            'p95_regression_at_most_2pct': c['latency_including_failures']['p95'] <= b['latency_including_failures']['p95'] * 1.02,
            'all_dependencies_clean': all(x['dependency_clean_completion'] == x['n'] for x in health['arms'].values()),
            'clean_goodput_no_regression': health['arms']['improved']['dependency_clean_throughput_rps'] >= health['arms']['baseline']['dependency_clean_throughput_rps'],
            'same_observed_call_counts': call['same_stage_call_counts'] == call['paired_requests'],
            'same_observed_retrieval_counts': call['same_retrieval_execution_counts'] == call['paired_requests'],
            'known_item_no_regression': c['source_law_hit'] >= b['source_law_hit'],
            'evidence_recall_at_least_baseline_repeat': (load['candidate_recall_vs_baseline'] is not None
                and load['baseline_repeat_recall'] is not None
                and load['candidate_recall_vs_baseline'] >= load['baseline_repeat_recall']),
        }
        gates.append({'level': level, 'checks': tests, 'passed': all(tests.values())})
    decision = {'utc': datetime.now(timezone.utc).isoformat(), 'gates': gates,
                'eligible_for_next_pilot': len(gates) >= 2 and all(g['passed'] for g in gates),
                'full_experiment_promoted': False, 'automatic_judgments': judge_count,
                'scope': 'Exploratory engineering gate, not accuracy equivalence or universal performance.'}
    write(root / 'decision.json', decision)
    public = root / 'public'
    public.mkdir(exist_ok=True)
    for name in ['plan.json', 'summary.json', 'status.json', 'requests.jsonl', 'stage-outcomes.json',
                 'audited-requests.jsonl', 'analysis.json', 'call-comparison.json', 'call-records.jsonl',
                 'decision.json', 'judge-plan.json']:
        if (root / name).exists():
            shutil.copy2(root / name, public / name)
    for file in root.glob('judge-r*-summary.json'):
        shutil.copy2(file, public / file.name)
    write(public / 'artifact-sha256.json', {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(public.iterdir()) if f.is_file() and f.name != 'artifact-sha256.json'})
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
