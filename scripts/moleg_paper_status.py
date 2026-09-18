"""Read-only, content-free progress view for a resumable paper campaign."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from summarize_moleg_paper import known_error_response


def read_complete_rows(path):
    if not path.exists():return []
    data=path.read_text();lines=data.splitlines()
    if data and not data.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def status(root):
    results={}
    expected={'retrieval.extraction.jsonl':800,'retrieval.jsonl':1200,
              'e2e_stream.jsonl':2400,'e2e_json.jsonl':800,
              'rerank_clipping.jsonl':400,'generation_replay.jsonl':360,
              'answer_judgments.jsonl':393,'country_ablation.jsonl':400,'query_anchor.jsonl':1600}
    for name,total in expected.items():
        rows=read_complete_rows(root/name)
        if not rows:continue
        results[name]={'rows':len(rows),'expected':total,
            'errors':dict(Counter(r.get('error_type','unspecified') for r in rows if r.get('ok') is False)),
            'by_variant_repeat':dict(Counter(f"{r.get('variant','all')}:{r.get('repeat',0)}" for r in rows))}
        if name.startswith('e2e_'):
            results[name]['known_app_error_responses']=dict(Counter(
                f"{r['variant']}:{known_error_response(r)}" for r in rows if known_error_response(r)))
    return {'utc':datetime.now(timezone.utc).isoformat(),'progress':results,
            'post_complete':(root/'post_complete.json').exists()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--interval',type=float,default=50)
    args=ap.parse_args()
    while True:
        value=status(args.directory);print(json.dumps(value,ensure_ascii=False),flush=True)
        if not args.watch or value['post_complete']:break
        time.sleep(args.interval)


if __name__=='__main__':main()
