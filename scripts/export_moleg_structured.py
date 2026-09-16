"""Validate all planned pairs and export only public numeric/hash evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_moleg_structured import analyze
from export_moleg_reasoning import public_safe


def read_rows(path):
    lines=path.read_text().splitlines(keepends=True)
    if any(not line.endswith('\n') for line in lines):
        raise ValueError('incomplete JSONL tail')
    return [json.loads(line) for line in lines]


def validate(root):
    plan=json.loads((root/'main-plan.json').read_text())
    status=json.loads((root/'main/campaign-status.json').read_text())
    summary=json.loads((root/'main/summary.json').read_text())
    rows=read_rows(root/'main/requests.jsonl')
    if status['status']!='completed' or not summary['complete'] or len(rows)!=plan['planned_requests']:
        raise ValueError('incomplete planned campaign')
    expected={(c['case_id'],c['question_sha256']) for c in plan['selected']}
    groups=defaultdict(list)
    for row in rows:
        groups[(row['arm'],row['level'],row['repeat'])].append(row)
    conditions={(a['name'],level,rep) for a in plan['arms'] for level in plan['levels'] for rep in range(plan['repeats'])}
    if set(groups)!=conditions or len(summary['trials'])!=len(conditions):
        raise ValueError('missing condition')
    for key, values in groups.items():
        if len(values)!=plan['unique_questions'] or {(r['case_id'],r['question_sha256']) for r in values}!=expected:
            raise ValueError('missing, duplicate or changed question')
        trial=next(t for t in summary['trials'] if (t['arm'],t['level'],t['repeat'])==key)
        if trial['success']!=sum(r['ok'] for r in values) or trial['pipeline_success']!=sum(r['pipeline_ok'] for r in values):
            raise ValueError('summary disagrees with request records')
    for name,digest in plan['code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen runtime code changed: '+name)
    live_path=root/'live-source-manifest.json'
    if live_path.exists():
        live=json.loads(live_path.read_text())
        app_root=root/'pod_source'
        observed={str(p.relative_to(app_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (app_root/'src').rglob('*.py')}
        if observed!=live:
            raise ValueError('application differs from the live source snapshot')
    return plan, rows


def stage_metrics(root):
    out={}
    for arm in ('baseline','prior','improved'):
        traces=read_rows(root/f'main/apps/{arm}.trace.jsonl')
        calls=[x for t in traces for x in t.get('structured_harness',[])]
        stages={}
        for schema in sorted({c['schema'] for c in calls}):
            values=[c for c in calls if c['schema']==schema]
            stages[schema]={'n':len(values),
                'mean_s':statistics.fmean(c['elapsed_s'] for c in values),
                'short_id_calls':sum(bool(c.get('short_ids')) for c in values),
                'mean_input_tokens':statistics.fmean(c['usage'].get('prompt_tokens',0) for c in values),
                'mean_output_tokens':statistics.fmean(c['usage'].get('completion_tokens',0) for c in values)}
        out[arm]={'traced_requests':len(traces),'stages':stages,
                  'fallbacks':sum(len(t.get('structured_fallback',[])) for t in traces)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args();r=args.directory
    plan,rows=validate(r)
    artifacts={'plan.json':plan,
        'analysis-vs-original.json':analyze(r/'main'),
        'analysis-vs-prior.json':analyze(r/'main',baseline='prior'),
        'analysis-prior-vs-original.json':analyze(r/'main',candidate='prior'),
        'stage-metrics.json':stage_metrics(r),
        'configurations.json':{name:json.loads((r/name).read_text()) for name in
                               ('baseline.adapter.json','combined.adapter.json',
                                'observe.harness.json','adaptive.harness.json','arms.main.json')},
        'model-inventory-before.json':json.loads((r/'main-models-before.json').read_text()),
        'model-inventory-after.json':json.loads((r/'main-models-after.json').read_text()),
        'replay-layout.json':read_rows(r/'replay-layout/requests.jsonl'),
        'replay-minified.json':read_rows(r/'replay/requests.jsonl')}
    for load in plan['levels']:
        path=r/f'judge-c{load}/answer_judge_summary.json'
        if path.exists():artifacts[f'judge-c{load}-summary.json']=json.loads(path.read_text())
    if (r/'live-source-manifest.json').exists():
        artifacts['live-source-manifest.json']=json.loads((r/'live-source-manifest.json').read_text())
    for value in artifacts.values():public_safe(value)
    args.destination.mkdir(mode=0o700,exist_ok=False)
    for name,value in artifacts.items():
        (args.destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    audit={'planned_requests':plan['planned_requests'],'observed_requests':len(rows),
           'all_question_pairs_valid':True,'frozen_runtime_hashes_valid':True,
           'application_source_matches_live_snapshot':(r/'live-source-manifest.json').exists(),
           'model_inventory_unchanged':artifacts['model-inventory-before.json']==artifacts['model-inventory-after.json'],
           'exported_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.destination.iterdir())}}
    (args.destination/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='exported_sha256'}))


if __name__=='__main__':main()
