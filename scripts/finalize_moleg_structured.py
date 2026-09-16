"""Finish validation after timing: blinded judges, then audited public export."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--app-env',type=Path,required=True)
    p.add_argument('--serving-env',type=Path,required=True)
    p.add_argument('--proxy-config',type=Path,required=True)
    args=p.parse_args();r=args.directory
    from dotenv import dotenv_values
    status_path=r/'finalization-status.json'
    def status(phase,**details):
        data={'phase':phase,'utc':datetime.now(timezone.utc).isoformat(),**details}
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(status_path)
        print(json.dumps(data),flush=True)
    try:
        status('waiting_for_performance_study')
        deadline=time.monotonic()+7200
        while True:
            campaign=json.loads((r/'main/campaign-status.json').read_text())
            if campaign['status'] in {'stopped','cancelled'}:
                raise RuntimeError('performance study did not complete')
            if campaign['status']=='completed' and (r/'main/analysis.json').exists() and (r/'main-models-after.json').exists():
                break
            if time.monotonic()>deadline:
                raise TimeoutError('performance study completion deadline')
            time.sleep(10)
        from export_moleg_structured import validate
        validate(r)  # do not judge or publish an incomplete performance study
        env=dict(os.environ,PYTHONPATH=str(r/'src'),MOLEG_RAG_ROOT=str(r/'pod_source'),
                 MOLEG_SERVING_ENV=str(args.serving_env),MOLEG_PROXY_CONFIG=str(args.proxy_config))
        env.update({k:v for k,v in dotenv_values(args.app_env).items() if v is not None})
        status('preparing_blinded_evaluation')
        subprocess.run([sys.executable,str(r/'scripts/prepare_moleg_structured_judge.py'),
                        '--directory',str(r)],check=True,env=env)
        for level in (4,20,100):
            status('judging',concurrency=level,expected_calls=96)
            subprocess.run([sys.executable,str(r/'scripts/evaluate_moleg_answer_judge.py'),
                '--directory',str(r/f'judge-c{level}'),'--judge-url','http://localhost:8002/v1',
                '--judge-model','microsoft/phi-4','--repeat','0'],check=True,env=env)
        status('exporting')
        subprocess.run([sys.executable,str(r/'scripts/export_moleg_structured.py'),
                        '--directory',str(r),'--destination',str(r/'public')],check=True,env=env)
        status('complete',expert_evaluation=False,production_deployed=False)
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        raise


if __name__=='__main__':main()
