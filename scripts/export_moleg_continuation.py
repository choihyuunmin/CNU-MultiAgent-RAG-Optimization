"""Export only numeric outcomes, hashes and configuration from a completed study."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_moleg_stage_outcomes import analyze


def export(root, name, destination):
    plan=json.loads((root/f'{name}-plan.json').read_text())
    run=root/name
    result,rows=analyze(run)
    if not result['complete'] or result['audit_errors']:raise ValueError('incomplete stage audit')
    expected=Counter((int(level),repeat,a['name'],c['case_id'],c['question_sha256'])
                     for level,cases in plan['cases'].items() for repeat in range(plan['repeats'])
                     for a in plan['arms'] for c in cases)
    observed=Counter((r['level'],r['repeat'],r['arm'],r['case_id'],r['question_sha256']) for r in rows)
    if observed!=expected or len(rows)!=plan['planned_requests']:raise ValueError('case coverage differs from plan')
    drift=[name for name,sha in plan['code_sha256'].items()
           if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]
    if drift:raise ValueError('frozen code changed: '+repr(drift))
    configurations={a['name']:{k:json.loads((root/a[k]).read_text())
                   for k in ('adapter_config','structured_harness','continuation_harness')} for a in plan['arms']}
    if configurations!=plan['configurations']:raise ValueError('configuration changed after freeze')
    if 'application_sha256' in plan:
        source={str(p.relative_to(root/'pod_source')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root/'pod_source/src').rglob('*.py')}
        if source!=plan['application_sha256']:raise ValueError('application source changed')
        if hashlib.sha256((root/'cases.json').read_bytes()).hexdigest()!=plan['case_set_sha256']:raise ValueError('case set changed')
    before=json.loads((root/f'{name}-models-before.json').read_text())
    after=json.loads((root/f'{name}-models-after.json').read_text())
    if before!=after:raise ValueError('model inventory changed')
    summary=json.loads((run/'summary.json').read_text())
    expected_order=[]
    for i,level in enumerate(plan['levels']):
        for repeat in range(plan['repeats']):
            arms=[a['name'] for a in plan['arms']]
            if (i+repeat)%2:arms.reverse()
            expected_order.extend((level,repeat,arm) for arm in arms)
    if [(t['level'],t['repeat'],t['arm']) for t in summary['trials']]!=expected_order:raise ValueError('trial order differs')
    if any(t['max_outstanding']!=t['level'] for t in summary['trials']):raise ValueError('target concurrency not reached')
    destination.mkdir(parents=True,exist_ok=True)
    write=lambda name,obj:(destination/name).write_text(json.dumps(obj,indent=2)+'\n')
    write('plan.json',plan);write('stage-outcomes.json',result);write('summary.json',summary)
    write('model-inventory-before.json',before);write('model-inventory-after.json',after)
    write('app-state-after.json',json.loads((run/'app-state-after.json').read_text()))
    (destination/'audited-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('audit.json',{'passed':True,'planned_requests':plan['planned_requests'],'observed_requests':len(rows),
                       'frozen_code_unchanged':True,'model_inventory_unchanged':True,
                       'frozen_configurations_unchanged':True,'application_source_unchanged':True if 'application_sha256' in plan else None,
                       'paired_case_coverage':True,'target_concurrency_reached':True,
                       'scope':'coverage, trace health and timing; not expert accuracy'})
    write('artifact-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file() and p.name!='artifact-sha256.json'})
    return {'name':name,'requests':len(rows),'passed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.directory,a.name,a.output)))

if __name__=='__main__':main()
