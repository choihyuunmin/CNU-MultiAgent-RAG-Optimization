"""Audit the frozen-body transfer experiment; do not infer full-RAG speed."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from moleg_paper_metrics import timing, paired_cluster_ci


def summarize(rows, manifest, repeats=3):
    inputs={(r['case_id'],r['stage']) for r in manifest['inputs']}
    expected={(q,s,rep,v) for q,s in inputs for rep in range(repeats)
              for v in ['identity','gzip']}
    keys=[(r['case_id'],r['stage'],r['repeat'],r['variant']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('missing, extra or duplicate transport observations')
    if not all(r['exact'] for r in rows):
        raise ValueError('transport changed request bytes')
    byinput=defaultdict(list)
    for r in rows:byinput[(r['case_id'],r['stage'])].append(r)
    for values in byinput.values():
        if len({(r['sha256'],r['raw_bytes']) for r in values})!=1:
            raise ValueError('input changed across conditions or repetitions')
    result={'scope':'request-body compression, HTTP transfer and decompression; no GPU inference',
            'rows':len(rows),'unique_questions':len({q for q,s in inputs}),
            'repeats':repeats,'all_exact':True,'variants':{},'comparisons':{},
            'byte_scope':'HTTP body only, excludes headers and protocol overhead'}
    for v in ['identity','gzip']:
        values=[r for r in rows if r['variant']==v]
        sent=sum(r['sent_body_bytes'] for r in values)
        raw=sum(r['raw_bytes'] for r in values)
        result['variants'][v]={'n':len(values),'latency':timing([r['elapsed_s'] for r in values]),
            'mean_raw_body_bytes':raw/len(values),'mean_sent_body_bytes':sent/len(values),
            'body_reduction_pct':100*(1-sent/raw),
            'mean_compression_s':statistics.fmean(r['compress_s'] for r in values),
            'mean_receiver_decode_hash_s':statistics.fmean(r['receiver_decode_hash_s'] for r in values)}
    for stage in ['all','classify','prepare']:
        pairs=[]
        for (_,s),values in byinput.items():
            if stage!='all' and s!=stage:continue
            pairs.append(tuple(statistics.fmean(r['elapsed_s'] for r in values if r['variant']==v)
                               for v in ['identity','gzip']))
        result['comparisons'][stage]=paired_cluster_ci(pairs)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=summarize([json.loads(x) for x in args.rows.read_text().splitlines()],
                     json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
