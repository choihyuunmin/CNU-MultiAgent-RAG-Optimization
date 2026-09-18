"""Audit the completed 1-to-200 study and publish numeric/hash-only artifacts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_moleg_structured import analyze
from export_moleg_reasoning import public_safe


def read_rows(path):
    rows=[]
    with path.open() as source:
        for line in source:
            if not line.endswith('\n'):
                raise ValueError('incomplete JSONL record')
            rows.append(json.loads(line))
    return rows


def validate(root):
    plan=json.loads((root/'plan.json').read_text())
    status=json.loads((root/'main/campaign-status.json').read_text())
    summary=json.loads((root/'main/summary.json').read_text())
    rows=read_rows(root/'main/requests.jsonl')
    if status['status']!='completed' or not summary['complete'] or len(rows)!=plan['planned_requests']:
        raise ValueError('planned concurrency study is incomplete')
    expected={(a['name'],level,repeat) for a in plan['arms'] for level in plan['levels'] for repeat in range(plan['repeats'])}
    groups=defaultdict(list)
    for row in rows:
        groups[(row['arm'],row['level'],row['repeat'])].append(row)
    if set(groups)!=expected or len(summary['trials'])!=len(expected):
        raise ValueError('missing or duplicated trial')
    for key,values in groups.items():
        entries=plan['expected_by_level'][str(key[1])]
        identities={(c['case_id'],c['question_sha256']) for c in entries}
        if len(values)!=len(entries) or {(r['case_id'],r['question_sha256']) for r in values}!=identities:
            raise ValueError('question membership changed between conditions')
        trial=next(t for t in summary['trials'] if (t['arm'],t['level'],t['repeat'])==key)
        if trial['pipeline_success']!=sum(r['pipeline_ok'] for r in values):
            raise ValueError('pipeline counts disagree')
        if trial['max_outstanding']!=key[1]:
            raise ValueError('requested peak concurrency was not reached')
    for name,digest in plan['code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source changed: '+name)
    live=json.loads((root/'live-source-manifest.json').read_text())
    observed={str(p.relative_to(root/'pod_source')):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'pod_source/src').rglob('*.py')}
    if observed!=live:
        raise ValueError('application source differs from the verified snapshot')
    return plan, rows


def stages(root, arms):
    result={}
    for arm in arms:
        traces=read_rows(root/f'main/apps/{arm}.trace.jsonl')
        calls=[call for trace in traces for call in trace.get('structured_harness',[])]
        schemas={}
        for schema in sorted({c['schema'] for c in calls}):
            values=[c for c in calls if c['schema']==schema]
            schemas[schema]={'n':len(values),'mean_s':statistics.fmean(c['elapsed_s'] for c in values),
                'short_id_calls':sum(bool(c.get('short_ids')) for c in values),
                'mean_input_tokens':statistics.fmean(c['usage'].get('prompt_tokens',0) for c in values),
                'mean_output_tokens':statistics.fmean(c['usage'].get('completion_tokens',0) for c in values)}
        result[arm]={'traced_requests':len(traces),'stages':schemas,
                     'fallbacks':sum(len(t.get('structured_fallback',[])) for t in traces)}
    return result


def export(root, destination):
    plan, rows=validate(root)
    artifacts={'plan.json':plan,'analysis.json':analyze(root/'main'),
               'stage-metrics.json':stages(root,[a['name'] for a in plan['arms']]),
               'model-inventory-before.json':json.loads((root/'models-before.json').read_text()),
               'model-inventory-after.json':json.loads((root/'models-after.json').read_text()),
               'live-source-manifest.json':json.loads((root/'live-source-manifest.json').read_text()),
               'configurations.json':{name:json.loads((root/name).read_text()) for name in
                 ['baseline.adapter.json','combined.adapter.json','observe.harness.json','portable.harness.json','arms.json']}}
    for level in [1,20,100,200]:
        path=root/f'judge-c{level}/answer_judge_summary.json'
        if path.exists():artifacts[f'judge-c{level}-summary.json']=json.loads(path.read_text())
    allowed=set('case_id kind index question_sha256 ok pipeline_ok error_type elapsed_s wire_s ttft_s first_event_s admission_wait_s answer_sha256 evidence_ids_sha256 answer_chars source_law_hit source_chunk_hit trial arm level repeat'.split())
    public_rows=[{k:v for k,v in row.items() if k in allowed} for row in rows]
    for value in [*artifacts.values(),public_rows]:public_safe(value)
    destination.mkdir(mode=0o700,exist_ok=False)
    for name,value in artifacts.items():
        (destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (destination/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public_rows))
    audit={'planned_requests':plan['planned_requests'],'observed_requests':len(rows),
           'all_question_pairs_valid':True,'peak_concurrency_verified':True,'frozen_runtime_hashes_valid':True,
           'application_source_matches_verified_live_snapshot':True,
           'model_inventory_unchanged':artifacts['model-inventory-before.json']==artifacts['model-inventory-after.json'],
           'exported_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())}}
    (destination/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({k:v for k,v in export(args.directory,args.destination).items() if k!='exported_sha256'}))
