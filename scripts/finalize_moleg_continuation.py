"""Finish timing first, then run separate answer judging and numeric export."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--auth-directory',type=Path,required=True)
    p.add_argument('--app-env',type=Path,required=True)
    p.add_argument('--serving-env',type=Path,required=True)
    p.add_argument('--proxy-config',type=Path,required=True)
    a=p.parse_args();root=a.directory.resolve();old=a.auth_directory.resolve()
    os.umask(0o077)
    def status(phase,**values):
        obj={'phase':phase,'utc':datetime.now(timezone.utc).isoformat(),**values}
        temp=root/'continuation-finalization-status.tmp'
        temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(root/'continuation-finalization-status.json')
        print(json.dumps(obj),flush=True)
    try:
        status('waiting_for_timing')
        while True:
            state=json.loads((root/'validation200/campaign-status.json').read_text())
            if state['status']=='stopped':raise RuntimeError('timing stopped')
            if state['status']=='completed' and (root/'validation200/public/stage-outcomes.json').exists():break
            time.sleep(10)
        from dotenv import dotenv_values
        from supervise_moleg_sweep import wait_models_quiet
        key=dotenv_values(a.serving_env)['VLLM_API_KEY']
        status('draining_before_quality')
        asyncio.run(wait_models_quiet({'orchestrator':'http://localhost:8000/metrics','worker':'http://localhost:8006/metrics'},
                                     {'Authorization':'Bearer '+key}))
        env=dict(os.environ)
        env.update({k:v for k,v in dotenv_values(a.app_env).items() if v is not None})
        env.update(MOLEG_RAG_ROOT=str(root/'pod_source'),MOLEG_SERVING_ENV=str(a.serving_env),
                   MOLEG_PROXY_CONFIG=str(a.proxy_config),PYTHONPATH=str(root/'src'))
        paths=['--app-env',str(a.app_env),'--serving-env',str(a.serving_env),'--proxy-config',str(a.proxy_config)]
        for name,level in [('pilot-window32',100),('validation200',200)]:
            status('judging',study=name,level=level)
            subprocess.run([sys.executable,str(root/'scripts/evaluate_moleg_continuation_judge.py'),
                            '--directory',str(root),'--name',name,'--level',str(level),*paths],env=env,check=True)
        for repeat in [0,1]:
            status('judging_posthoc_auth',repeat=repeat)
            authenv=dict(env,MOLEG_RAG_ROOT=str(old/'pod_source'),PYTHONPATH=str(old/'src'))
            subprocess.run([sys.executable,str(root/'scripts/evaluate_moleg_answer_judge.py'),
                            '--directory',str(old/f'pilot-auth/judge-c32-repeat{repeat}'),
                            '--repeat',str(repeat),'--judge-url','http://localhost:8002/v1',
                            '--judge-model','microsoft/phi-4'],env=authenv,check=True)
        status('exporting')
        from export_moleg_continuation import export
        for name in ['pilot-window32','validation200']:
            export(root,name,root/name/'public')
        for parent,name,level in [(root,'pilot-window32',100),(root,'validation200',200),(old,'pilot-auth',32)]:
            for repeat in [0,1]:
                source=parent/name/f'judge-c{level}-repeat{repeat}/answer_judge_summary.json'
                value=json.loads(source.read_text())
                if value['completed']!=value['expected']:raise ValueError('incomplete judge')
                (parent/name/'public'/f'judge-c{level}-repeat{repeat}-summary.json').write_text(json.dumps(value,indent=2)+'\n')
        status('complete',performance_requests=2264,preregistered_judgments=160,posthoc_auth_judgments=40)
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        raise

if __name__=='__main__':main()
