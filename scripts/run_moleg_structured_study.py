"""Freeze and run a disjoint, counterbalanced three-arm GPU study.

The directory already contains a frozen pod_source/, cases.json, pilot.cases.json,
baseline.adapter.json and combined.adapter.json. Secrets stay in operator files.
"""
from __future__ import annotations

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
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--app-env', type=Path, required=True)
    p.add_argument('--serving-env', type=Path, required=True)
    p.add_argument('--proxy-config', type=Path, required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--levels', type=int, nargs='+', default=[4,20,100])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--min-candidates', type=int, default=16)
    args=p.parse_args()
    from dotenv import dotenv_values
    r=args.directory.resolve()
    if (r/'main-plan.json').exists() or (r/'main').exists():
        raise ValueError('use a fresh study directory')
    if args.count < max(args.levels) or args.repeats < 2:
        raise ValueError('need enough unique questions for peak concurrency and >=2 repeats')
    cases=json.loads((r/'cases.json').read_text())
    pilot=json.loads((r/'pilot.cases.json').read_text())
    excluded={digest(c['question']) for c in pilot}
    selected=select_cases([c for c in cases if digest(c['question']) not in excluded],args.count,20260917)
    if len({digest(c['question']) for c in selected})!=args.count:
        raise ValueError('duplicate questions')
    (r/'main.cases.json').write_text(json.dumps(selected,ensure_ascii=False))
    fingerprints={f'agent.{name}':hashlib.sha256((r/f'pod_source/src/agent/{name}.py').read_bytes()).hexdigest()
                  for name in ('law_analysis_agent','law_search_agent')}
    for name, options in {'observe':{}, 'adaptive':{'short_ids':True,'min_short_id_candidates':args.min_candidates}}.items():
        (r/f'{name}.harness.json').write_text(json.dumps(dict(options,capture=True,fingerprints=fingerprints),indent=2))
    arms=[{'name':'baseline','port':28410,'adapter_config':'baseline.adapter.json','structured_harness':'observe.harness.json'},
          {'name':'prior','port':28411,'adapter_config':'combined.adapter.json','structured_harness':'observe.harness.json'},
          {'name':'improved','port':28412,'adapter_config':'combined.adapter.json','structured_harness':'adaptive.harness.json'}]
    (r/'arms.main.json').write_text(json.dumps(arms,indent=2))
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'unique_questions':args.count,
          'levels':args.levels,'repeats':args.repeats,'arms':arms,
          'planned_requests':args.count*len(args.levels)*args.repeats*len(arms),
          'pilot_excluded':len(excluded),'min_short_id_candidates':args.min_candidates,
          'selected':[{'case_id':c['case_id'],'kind':c['kind'],'question_sha256':digest(c['question'])} for c in selected],
          'controls':{'models':'unchanged','prompts':'same instructions, only request-local candidate IDs replaced',
                      'retrieval':'same queries/tools/indices and arguments; no typed dispatch in main',
                      'cache':'no response/result cache','worker':'prior and improved both reasoning_effort=low',
                      'output_whitespace':'unchanged','input_layout':'preserved','timeout_s':240,'slo_s':30},
          'quality_rule':'Report source-hit difference CI and baseline-repeat evidence variability; no expert/noninferiority claim from CI crossing zero',
          'code_sha256':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest()
                         for folder in ('src','scripts') for f in sorted((r/folder).rglob('*.py'))}}
    (r/'main-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    (r/'main-models-before.json').write_text(json.dumps(model_inventory(),indent=2))
    env=dict(os.environ,PYTHONPATH=str(r/'src'),VLLM_API_KEY=dotenv_values(args.serving_env)['VLLM_API_KEY'])
    argv=[sys.executable,str(r/'scripts/supervise_moleg_sweep.py'),'--directory',str(r),'--python',sys.executable,
          '--app-root',str(r/'pod_source'),'--app-env',str(args.app_env),'--serving-env',str(args.serving_env),
          '--proxy-config',str(args.proxy_config),'--arms','arms.main.json','--cases-file','main.cases.json',
          '--run-subdir','main','--levels',*[str(v) for v in args.levels],'--repeats',str(args.repeats),
          '--pool',str(args.count),'--min-requests',str(args.count),'--max-requests',str(args.count),
          '--reference','improved','--collapse-success','0.95','--metrics','orchestrator=http://localhost:8000/metrics',
          '--metrics','worker=http://localhost:8006/metrics','--api-key-env','VLLM_API_KEY']
    print(json.dumps({'planned_requests':plan['planned_requests'],'levels':args.levels,'repeats':args.repeats}),flush=True)
    try:
        subprocess.run(argv,env=env,check=True)
    finally:
        (r/'main-models-after.json').write_text(json.dumps(model_inventory(),indent=2))
    subprocess.run([sys.executable,str(r/'scripts/analyze_moleg_structured.py'),'--directory',str(r/'main')],check=True,env=env)


if __name__=='__main__':main()
