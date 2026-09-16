"""Supplementary answer judging from stored responses, without new RAG requests.

Keep this post-study sensitivity analysis separate from the preplanned judges.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from export_reference_scaling import read_rows, validate


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--repeat', type=int, required=True)
    parser.add_argument('--app-env', type=Path, required=True)
    parser.add_argument('--serving-env', type=Path, required=True)
    parser.add_argument('--proxy-config', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    plan, _ = validate(root)
    if args.level not in plan['levels'] or not 0 <= args.repeat < plan['repeats']:
        raise ValueError('condition was not measured')
    ids = {c['case_id'] for c in plan['expected_by_level'][str(args.level)]}
    cases = {c['case_id']: c for c in json.loads((root/'cases.json').read_text())}
    selected = [r for r in read_rows(root/'main/private-responses.jsonl')
                if r['level'] == args.level and r['repeat'] == args.repeat]
    expected = {(case_id, arm['name']) for case_id in ids for arm in plan['arms']}
    if len(selected) != len(expected) or {(r['case_id'], r['arm']) for r in selected} != expected:
        raise ValueError('incomplete or duplicated response set')
    destination = root/f'judge-c{args.level}-repeat{args.repeat}'
    destination.mkdir(mode=0o700, exist_ok=False)
    (destination/'cases.json').write_text(json.dumps([cases[i] for i in sorted(ids)], ensure_ascii=False))
    with (destination/'e2e_stream.jsonl').open('x') as sink:
        for row in selected:
            sink.write(json.dumps({'case_id': row['case_id'], 'variant': row['arm'],
                                   'repeat': args.repeat, 'result': row.get('result')},
                                  ensure_ascii=False)+'\n')
    from dotenv import dotenv_values
    env = dict(os.environ)
    env.update({k: v for k, v in dotenv_values(args.app_env).items() if v is not None})
    env.update(MOLEG_RAG_ROOT=str(root/'pod_source'), MOLEG_SERVING_ENV=str(args.serving_env),
               MOLEG_PROXY_CONFIG=str(args.proxy_config), PYTHONPATH=str(root/'src'))
    subprocess.run([sys.executable, str(root/'scripts/evaluate_moleg_answer_judge.py'),
                    '--directory', str(destination), '--judge-url', 'http://localhost:8002/v1',
                    '--judge-model', 'microsoft/phi-4', '--repeat', str(args.repeat)],
                   env=env, check=True)
    summary = json.loads((destination/'answer_judge_summary.json').read_text())
    if summary['expected'] != summary['completed']:
        raise ValueError('supplementary judging is incomplete')
    print(json.dumps({'phase': 'complete', 'level': args.level, 'repeat': args.repeat,
                      'judgments': summary['completed'], 'scope': 'post-study sensitivity analysis'}))


if __name__ == '__main__':
    main()
