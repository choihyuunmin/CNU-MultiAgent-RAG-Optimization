"""Freeze and run a 1-to-200 concurrency study of reference ID shortening.

Creates no credentials. Operator environment paths are mandatory. Question pools
are stratified and fixed per load, identical across variants and both repetitions.
"""
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


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--app-env',type=Path,required=True)
    parser.add_argument('--serving-env',type=Path,required=True)
    parser.add_argument('--proxy-config',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[1,2,4,8,16,20,32,50,100,150,200])
    parser.add_argument('--min-requests',type=int,default=32)
    args=parser.parse_args()
    from dotenv import dotenv_values
    root=args.directory.resolve()
    if (root/'plan.json').exists() or (root/'main').exists():
        raise ValueError('use a fresh study directory')
    if args.levels!=sorted(set(args.levels)) or args.levels[0]<1 or args.levels[-1]>200:
        raise ValueError('invalid concurrency levels')
    cases=json.loads((root/'cases.json').read_text())
    if len(cases)!=200 or len({digest(c['question']) for c in cases})!=200:
        raise ValueError('need the complete 200 unique question regression set')
    pools={str(level):select_cases(cases,max(args.min_requests,level),20260916) for level in args.levels}
    write(root/'level-pools.json',{level:[c['case_id'] for c in pool] for level,pool in pools.items()})
    fingerprints={f'agent.{name}':hashlib.sha256((root/f'pod_source/src/agent/{name}.py').read_bytes()).hexdigest()
                  for name in ('law_analysis_agent','law_search_agent')}
    write(root/'observe.harness.json',{'capture':False,'fingerprints':fingerprints})
    write(root/'portable.harness.json',{'capture':False,'short_ids':True,'generic_reference_codec':True,
                                       'min_short_id_candidates':16,'fingerprints':fingerprints})
    arms=[{'name':'baseline','port':28610,'adapter_config':'baseline.adapter.json','structured_harness':'observe.harness.json'},
          {'name':'improved','port':28611,'adapter_config':'combined.adapter.json','structured_harness':'portable.harness.json'}]
    write(root/'arms.json',arms)
    live=json.loads((root/'live-source-manifest.json').read_text())
    observed={str(p.relative_to(root/'pod_source')):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'pod_source/src').rglob('*.py')}
    if observed!=live:
        raise ValueError('frozen application differs from live source manifest')
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(), 'method':'reference_id_shortening_and_restoration',
          'levels':args.levels,'repeats':2,'arms':arms,
          'requests_per_trial':{level:len(pool) for level,pool in pools.items()},
          'planned_requests':4*sum(len(pool) for pool in pools.values()),
          'expected_by_level':{level:[{k:c[k] for k in ['case_id','kind']}|{'question_sha256':digest(c['question'])} for c in pool]
                               for level,pool in pools.items()},
          'question_set_scope':'Existing regression set; new execution, not a new unseen benchmark.',
          'measurement_scope':'Finite closed-loop workload. At n=concurrency this is one simultaneous burst, not sustained capacity.',
          'order':'Complete both counterbalanced repetitions at each concurrency before increasing load.',
          'controls':{'baseline':'Original workflow and default worker reasoning',
                      'improved':'Same workflow, worker low reasoning, generic ID shortening for at least 16 candidates',
                      'models_and_serving':'Unchanged; shared existing model servers',
                      'retrieval':'Same tools, instructions and evidence; no candidate or text removal',
                      'cache':'No answer/result cache', 'source_files':len(live),
                      'timeout_s':240,'goodput_slo_s':30,'drain_between_trials':True},
          'quality_scope':'Evidence overlap and source labels; no expert accuracy or noninferiority guarantee',
          'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for folder in ('src','scripts') for p in sorted((root/folder).rglob('*.py'))}}
    write(root/'plan.json',plan)
    write(root/'models-before.json',model_inventory())
    env=dict(os.environ,PYTHONPATH=str(root/'src'),VLLM_API_KEY=dotenv_values(args.serving_env)['VLLM_API_KEY'])
    command=[sys.executable,str(root/'scripts/supervise_moleg_sweep.py'),'--directory',str(root),
             '--python',sys.executable,'--app-root',str(root/'pod_source'),'--app-env',str(args.app_env),
             '--serving-env',str(args.serving_env),'--proxy-config',str(args.proxy_config),
             '--run-subdir','main','--levels',*[str(n) for n in args.levels], '--repeats','2',
             '--pool','200','--min-requests',str(args.min_requests),'--max-requests','200',
             '--level-pools','level-pools.json','--repeat-within-level','--drain-between-trials',
             '--reference','improved','--collapse-success','0.5','--seed','20260916',
             '--metrics','orchestrator=http://localhost:8000/metrics','--metrics','worker=http://localhost:8006/metrics',
             '--api-key-env','VLLM_API_KEY']
    print(json.dumps({'planned_requests':plan['planned_requests'],'levels':args.levels,'repeats':2}),flush=True)
    try:
        subprocess.run(command,env=env,check=True)
    finally:
        write(root/'models-after.json',model_inventory())
    subprocess.run([sys.executable,str(root/'scripts/analyze_moleg_structured.py'),'--directory',str(root/'main')],env=env,check=True)
    print(json.dumps({'phase':'performance_complete'}),flush=True)


if __name__=='__main__':main()
