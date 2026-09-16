"""Wait for performance timing, then assess answer evidence and audit/export."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from export_reference_scaling import export, read_rows, validate


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--app-env',type=Path,required=True)
    parser.add_argument('--serving-env',type=Path,required=True)
    parser.add_argument('--proxy-config',type=Path,required=True)
    args=parser.parse_args();root=args.directory
    def status(phase,**fields):
        value={'phase':phase,'utc':datetime.now(timezone.utc).isoformat(),**fields}
        temporary=root/'finalization-status.tmp'
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(root/'finalization-status.json')
        print(json.dumps(value),flush=True)
    status('waiting_for_performance')
    try:
        while True:
            path=root/'main/campaign-status.json'
            if path.exists():
                state=json.loads(path.read_text())
                if state['status']=='stopped':
                    raise RuntimeError('performance campaign stopped')
                if state['status']=='completed' and (root/'models-after.json').exists() and (root/'main/analysis.json').exists():
                    break
            time.sleep(15)
        plan, _=validate(root)
        responses=read_rows(root/'main/private-responses.jsonl')
        cases={c['case_id']:c for c in json.loads((root/'cases.json').read_text())}
        from dotenv import dotenv_values
        env=dict(os.environ)
        env.update({k:v for k,v in dotenv_values(args.app_env).items() if v is not None})
        env.update(MOLEG_RAG_ROOT=str(root/'pod_source'),MOLEG_SERVING_ENV=str(args.serving_env),
                   MOLEG_PROXY_CONFIG=str(args.proxy_config),PYTHONPATH=str(root/'src'))
        for level in [1,20,100,200]:
            ids={c['case_id'] for c in plan['expected_by_level'][str(level)]}
            selected=[r for r in responses if r['level']==level and r['repeat']==0]
            expected={(case_id,arm['name']) for case_id in ids for arm in plan['arms']}
            if len(selected)!=len(expected) or {(r['case_id'],r['arm']) for r in selected}!=expected:
                raise ValueError('incomplete judge input')
            directory=root/f'judge-c{level}'
            directory.mkdir(mode=0o700,exist_ok=False)
            (directory/'cases.json').write_text(json.dumps([cases[i] for i in sorted(ids)],ensure_ascii=False))
            with (directory/'e2e_stream.jsonl').open('x') as sink:
                for row in selected:
                    sink.write(json.dumps({'case_id':row['case_id'],'variant':row['arm'],'repeat':0,
                                           'result':row.get('result')},ensure_ascii=False)+'\n')
            status('judging',level=level,eligible=sum(bool(cases[i].get('reference_answer')) for i in ids)*2)
            subprocess.run([sys.executable,str(root/'scripts/evaluate_moleg_answer_judge.py'),
                            '--directory',str(directory),'--judge-url','http://localhost:8002/v1',
                            '--judge-model','microsoft/phi-4','--repeat','0'],env=env,check=True)
        status('exporting')
        audit=export(root,root/'public')
        status('complete',requests=audit['observed_requests'],model_inventory_unchanged=audit['model_inventory_unchanged'])
    except Exception as exc:
        status('failed',error_type=type(exc).__name__)
        raise


if __name__=='__main__':main()
