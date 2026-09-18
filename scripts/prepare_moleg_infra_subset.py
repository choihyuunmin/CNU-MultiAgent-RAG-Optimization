"""Freeze an exploratory low-load subset, preserving the main stage assignment."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--requests',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-stage',type=int,default=24)
    args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text())
    selected=[]
    for stage in ['classify','prepare']:
        values=sorted((r for r in manifest['inputs'] if r['stage']==stage),key=lambda r:r['case_id'])
        random.Random(6202610).shuffle(values)
        selected.extend(values[:args.per_stage])
    wanted={(r['case_id'],r['stage']) for r in selected}
    requests=[r for r in json.loads(args.requests.read_text()) if (r['case_id'],r['stage']) in wanted]
    if len(requests)!=args.per_stage*2:
        raise ValueError('not enough requests or missing selected input')
    expected={(r['case_id'],r['stage']):r['payload_sha256'] for r in selected}
    assert all(r['payload_sha256']==expected[(r['case_id'],r['stage'])] for r in requests)
    args.output.write_text(json.dumps(requests,ensure_ascii=False)+'\n')
    args.output.with_suffix('.selection.json').write_text(json.dumps({
        'scope':'exploratory low-load control, not a second 400-question main experiment',
        'selection':'seed 6202610, 24 per stage, preserving main stage assignment',
        'parent_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'inputs':selected},indent=2)+'\n')
    print(json.dumps({'selected_requests':len(requests),'unique_questions':len({r['case_id'] for r in requests})}))


if __name__=='__main__':main()
