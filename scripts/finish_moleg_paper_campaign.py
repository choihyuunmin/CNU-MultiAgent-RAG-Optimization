"""Run additional audits sequentially after all primary GPU timing finishes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-pid',type=int,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parent;results=root/'results'
    while True:
        try:os.kill(args.campaign_pid,0)
        except ProcessLookupError:break
        time.sleep(10)
    for filename,expected in [('e2e_stream.jsonl',2400),('e2e_json.jsonl',800)]:
        path=results/filename
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        if len({(r['case_id'],r['variant'],r['repeat']) for r in rows})!=expected:
            raise RuntimeError(f'{filename}: main campaign incomplete, expected {expected}, got {len(rows)}')
    env={**os.environ,'MOLEG_RAG_ROOT':str(root/'app')}
    commands=[
        ['evaluate_moleg_stream_replay.py','--trace',str(results/'baseline.trace.jsonl'),'--output',str(results/'stream_replay.json')],
        ['evaluate_moleg_paper_retrieval.py','--cases',str(results/'cases.json'),'--output',str(results/'rerank_clipping.jsonl'),
         '--repeats','1','--extraction-cache',str(results/'retrieval.extraction.jsonl'),
         '--variants','raw,auth_content,strict_auth_content'],
        ['evaluate_moleg_generation_replay.py','--directory',str(results)],
        ['evaluate_moleg_answer_judge.py','--directory',str(results),'--judge-url','http://127.0.0.1:8002/v1','--judge-model','microsoft/phi-4'],
        ['evaluate_moleg_country_ablation.py','--directory',str(results)],
        ['evaluate_moleg_query_anchor.py','--directory',str(results)],
        ['summarize_moleg_paper.py','--directory',str(results),'--output',str(results/'summary.json')],
        ['export_moleg_expert_review.py','--directory',str(results)],
        ['collect_moleg_paper_environment.py','--output',str(results/'environment_after.json'),
         '--live-root',os.environ['MOLEG_LIVE_ROOT'],'--serving-python',os.environ['MOLEG_SERVING_PYTHON']],
    ]
    for command in commands:
        print('start',command[0],flush=True)
        with (root/(command[0]+'.progress')).open('a') as output:
            subprocess.run([sys.executable,str(root/command[0])]+command[1:],env=env,cwd=root,
                           stdout=output,stderr=subprocess.STDOUT,check=True)
    (results/'post_complete.json').write_text(json.dumps({'complete':True,'unix_time':time.time()})+'\n')
    print('post_complete',flush=True)


if __name__=='__main__':main()
