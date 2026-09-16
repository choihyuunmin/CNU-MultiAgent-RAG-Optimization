"""Freeze a named multi-arm study and run it on existing, unchanged model servers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_moleg_scaling import select_cases, digest
from supervise_moleg_combined import model_inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--arms',required=True)
    p.add_argument('--levels',type=int,nargs='+',required=True)
    p.add_argument('--min-requests',type=int,default=32)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=202609162)
    p.add_argument('--app-env',type=Path,required=True)
    p.add_argument('--serving-env',type=Path,required=True)
    p.add_argument('--proxy-config',type=Path,required=True)
    args=p.parse_args()
    from dotenv import dotenv_values
    os.umask(0o077)
    root=args.directory.resolve()
    if not args.name.replace('-','').isalnum():raise ValueError('invalid run name')
    if (root/args.name).exists() or (root/f'{args.name}-plan.json').exists():raise ValueError('use a new run name')
    if args.levels!=sorted(set(args.levels)) or not 1<=args.levels[0]<=args.levels[-1]<=200:raise ValueError('invalid loads')
    live=json.loads((root/'live-source-manifest.json').read_text())
    source={str(p.relative_to(root/'pod_source')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'pod_source/src').rglob('*.py')}
    if source!=live:raise ValueError('application differs from live source manifest')
    cases=json.loads((root/'cases.json').read_text())
    pools={str(l):select_cases(cases,max(args.min_requests,l),args.seed) for l in args.levels}
    write=lambda path,obj:path.write_text(json.dumps(obj,indent=2)+'\n')
    write(root/f'{args.name}-pools.json',{l:[c['case_id'] for c in pool] for l,pool in pools.items()})
    arms=json.loads((root/args.arms).read_text())
    cmd=[sys.executable,str(root/'scripts/supervise_moleg_sweep.py'),'--directory',str(root),
         '--python',sys.executable,'--app-root',str(root/'pod_source'),
         '--app-env',str(args.app_env),'--serving-env',str(args.serving_env),'--proxy-config',str(args.proxy_config),
         '--arms',args.arms,'--run-subdir',args.name,'--levels',*[str(l) for l in args.levels],
         '--repeats',str(args.repeats),'--repeat-within-level','--drain-between-trials',
         '--pool','200','--min-requests',str(args.min_requests),'--max-requests','200',
         '--level-pools',f'{args.name}-pools.json','--reference',arms[-1]['name'],'--collapse-success','.5',
         '--seed',str(args.seed),'--metrics','orchestrator=http://localhost:8000/metrics',
         '--metrics','worker=http://localhost:8006/metrics','--api-key-env','VLLM_API_KEY']
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'name':args.name,'levels':args.levels,
          'repeats':args.repeats,'arms':arms,'command':cmd,
          'application_sha256':source,
          'case_set_sha256':hashlib.sha256((root/'cases.json').read_bytes()).hexdigest(),
          'planned_requests':args.repeats*len(arms)*sum(len(pool) for pool in pools.values()),
          'scope':'Existing regression set; counterbalanced finite bursts; shared model servers; not expert accuracy.',
          'cases':{l:[{'case_id':c['case_id'],'kind':c['kind'],'question_sha256':digest(c['question'])} for c in pool] for l,pool in pools.items()},
          'configurations':{a['name']:{k:json.loads((root/a[k]).read_text()) for k in ('adapter_config','structured_harness','continuation_harness')} for a in arms},
          'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for d in ('src','scripts') for p in sorted((root/d).rglob('*.py'))}}
    write(root/f'{args.name}-plan.json',plan)
    before=model_inventory();write(root/f'{args.name}-models-before.json',before)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),VLLM_API_KEY=dotenv_values(args.serving_env)['VLLM_API_KEY'])
    try:subprocess.run(cmd,env=env,check=True)
    finally:write(root/f'{args.name}-models-after.json',model_inventory())
    subprocess.run([sys.executable,str(root/'scripts/audit_moleg_stage_outcomes.py'),
                    '--directory',str(root/args.name),'--output',str(root/args.name/'public')],env=env,check=True)

if __name__=='__main__':main()
