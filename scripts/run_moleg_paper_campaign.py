"""Sequential campaign driver; explicit wait for complete retrieval records."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--retrieval-pid',type=int,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent;result=root/'results'
    print('waiting_for_retrieval',flush=True)
    while True:
        path=result/'retrieval.jsonl'
        lines=path.read_text().splitlines() if path.exists() else []
        if len(lines)==1200:
            rows=[json.loads(x) for x in lines]
            if len({(r['case_id'],r['repeat']) for r in rows})!=1200:
                raise RuntimeError('duplicate retrieval keys')
            if not all(r['ok'] for r in rows):
                raise RuntimeError('retrieval errors must be reviewed before E2E')
            break
        try:os.kill(args.retrieval_pid,0)
        except ProcessLookupError:raise RuntimeError(f'retrieval ended early at {len(lines)} rows')
        time.sleep(5)
    print('start_e2e_stream',flush=True)
    common=[sys.executable,str(root/'evaluate_moleg_paper_e2e.py'),'--cases',str(result/'cases.json'),
        '--variant','baseline=http://127.0.0.1:28110','--variant','balanced=http://127.0.0.1:28112']
    with (root/'e2e_stream.progress').open('a') as log:
        subprocess.run(common+['--variant','speed=http://127.0.0.1:28111','--repeats','2',
            '--output',str(result/'e2e_stream.jsonl')],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('start_e2e_json',flush=True)
    with (root/'e2e_json.progress').open('a') as log:
        subprocess.run(common+['--route','/api/generate','--repeats','1',
            '--output',str(result/'e2e_json.jsonl')],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('campaign_complete',flush=True)


if __name__=='__main__':main()
