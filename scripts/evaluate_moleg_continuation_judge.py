"""Assess preregistered reference-bearing questions after performance timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--level',type=int,required=True)
    p.add_argument('--app-env',type=Path,required=True)
    p.add_argument('--serving-env',type=Path,required=True)
    p.add_argument('--proxy-config',type=Path,required=True)
    a=p.parse_args();root=a.directory.resolve();run=root/a.name
    os.umask(0o077)
    if json.loads((run/'campaign-status.json').read_text())['status']!='completed':raise ValueError('timing unfinished')
    plan=json.loads((root/'judge-plan.json').read_text())['loads'][str(a.level)]
    ids={c['case_id'] for c in plan['cases']}
    cases={c['case_id']:c for c in json.loads((root/'cases.json').read_text())}
    rows=[json.loads(line) for line in (run/'private-responses.jsonl').read_text().splitlines()]
    arms={a['name'] for a in json.loads((root/f'{a.name}-plan.json').read_text())['arms']}
    from dotenv import dotenv_values
    env=dict(os.environ)
    env.update({k:v for k,v in dotenv_values(a.app_env).items() if v is not None})
    env.update(MOLEG_RAG_ROOT=str(root/'pod_source'),MOLEG_SERVING_ENV=str(a.serving_env),
               MOLEG_PROXY_CONFIG=str(a.proxy_config),PYTHONPATH=str(root/'src'))
    for repeat in plan['repeats']:
        selected=[r for r in rows if r['level']==a.level and r['repeat']==repeat and r['case_id'] in ids]
        expected={(cid,arm) for cid in ids for arm in arms}
        if len(selected)!=len(expected) or {(r['case_id'],r['arm']) for r in selected}!=expected:raise ValueError('judge pairs missing')
        if any(r['question_sha256']!=next(c['question_sha256'] for c in plan['cases'] if c['case_id']==r['case_id']) for r in selected):raise ValueError('judge question hash changed')
        dest=run/f'judge-c{a.level}-repeat{repeat}'
        dest.mkdir(mode=0o700,exist_ok=False)
        (dest/'cases.json').write_text(json.dumps([cases[cid] for cid in sorted(ids)],ensure_ascii=False))
        (dest/'e2e_stream.jsonl').write_text(''.join(json.dumps({'case_id':r['case_id'],'variant':r['arm'],'repeat':repeat,'result':r.get('result')},ensure_ascii=False)+'\n' for r in selected))
        subprocess.run([sys.executable,str(root/'scripts/evaluate_moleg_answer_judge.py'),
                        '--directory',str(dest),'--judge-url','http://localhost:8002/v1',
                        '--judge-model','microsoft/phi-4','--repeat',str(repeat)],env=env,check=True)
        summary=json.loads((dest/'answer_judge_summary.json').read_text())
        if summary['completed']!=summary['expected']:raise ValueError('judge incomplete')

if __name__=='__main__':main()
