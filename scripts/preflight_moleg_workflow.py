"""Allow-listed health/tokenization inventory; no inference or credential output."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from moleg_adapter_serving import operator_routes
from moleg_scaling_metrics import parse_metrics


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    os.environ['MOLEG_SERVING_ENV'] = '/data/project/vllm/.env'
    os.environ['MOLEG_PROXY_CONFIG'] = '/data/project/vllm/litellm/develop_test_vllm_config_v3.yaml'
    import httpx
    routes = operator_routes()
    rows = []
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        for role, route in routes.items():
            row = {'role': role, 'model': route['model']}
            try:
                headers = {'Authorization': 'Bearer ' + route['key']}
                health = await client.get(route['base'] + '/health', headers=headers)
                metrics = await client.get(route['base'] + '/metrics', headers=headers)
                row['health'] = health.status_code
                parsed = parse_metrics(metrics.text)
                row['cache_config'] = parsed['cache_config']
                row['gauges'] = {k:v for k,v in parsed['values'].items()
                                if k in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc','num_preemptions_total']}
                if role == 'orchestrator':
                    response = await client.post(route['base'] + '/tokenize', headers=headers,
                        json={'model': route['model'], 'messages': [{'role':'user','content':'health check'}],
                              'add_generation_prompt': True})
                    row['tokenize_status'] = response.status_code
                    if response.status_code == 200:
                        row['tokenize'] = {k:response.json().get(k) for k in ['count','max_model_len']}
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            rows.append(row)
    out = {'utc': datetime.now(timezone.utc).isoformat(), 'models': rows,
        'gpus': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                                        '--format=csv,noheader'], text=True).splitlines(),
        'source_files': {str(p.relative_to(root / 'pod_source')):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((root / 'pod_source/src').rglob('*.py'))}}
    path = root / 'preflight.json'
    with path.open('x') as sink:
        json.dump(out, sink, indent=2)
    print(json.dumps({'models': rows, 'source_files':len(out['source_files']), 'gpus':out['gpus']}))


if __name__ == '__main__':
    asyncio.run(main())
