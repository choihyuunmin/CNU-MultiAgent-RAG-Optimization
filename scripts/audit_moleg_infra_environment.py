"""Compare allow-listed production snapshots; publish no internal endpoint list."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def audit(before,after,gpu_before,gpu_after):
    checks={}
    for resource in ['deployments','pods','services']:
        a=sorted(before[resource],key=lambda x:x['name'])
        b=sorted(after[resource],key=lambda x:x['name'])
        checks[resource+'_unchanged']=a==b
        checks[resource+'_before_sha256']=digest(a)
        checks[resource+'_after_sha256']=digest(b)
    a=sorted(gpu_before['servers'],key=lambda x:x['pid'])
    b=sorted(gpu_after['servers'],key=lambda x:x['pid'])
    checks['model_api_processes_arguments_and_gpu_unchanged']=a==b
    checks['model_api_count_before']=len(a)
    checks['model_api_count_after']=len(b)
    checks['all_checks_passed']=all(v for k,v in checks.items() if k.endswith('unchanged'))
    return {'scope':'sampled before/after deployment, pod, service and model API process records',
            'limitation':'does not prove zero user-visible contention during experiments or monitor every engine-worker restart',
            'before_utc':before['utc'],'after_utc':after['utc'],
            'gpu_before_utc':gpu_before['utc'],'gpu_after_utc':gpu_after['utc'],
            'gpu_models_before':gpu_before['gpus'],'gpu_models_after':gpu_after['gpus'],
            'checks':checks}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--private-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    values=[json.loads((args.private_dir/name).read_text()) for name in
            ['cluster_before.json','cluster_after.json','gpu_before.json','gpu_after.json']]
    result=audit(*values)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    if not result['checks']['all_checks_passed']:
        raise SystemExit('production snapshot changed; inspect before making preservation claims')
