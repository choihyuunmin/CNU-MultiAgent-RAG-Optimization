"""Read-only serving counters during the campaign; never captures prompts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import time


METRICS={'vllm:num_requests_running','vllm:num_requests_waiting',
    'vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc','vllm:num_preemptions_total',
    'vllm:prompt_tokens_total','vllm:generation_tokens_total',
    'vllm:prefix_cache_queries_total','vllm:prefix_cache_hits_total'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--monitor-pid',type=int,required=True)
    args=ap.parse_args()
    import requests
    from dotenv import dotenv_values
    key=dotenv_values(os.environ['MOLEG_SERVING_ENV'])['VLLM_API_KEY']
    ports=[8000,8001,8002,8005,8006,8020,8030]
    def scrape(port):
        try:
            response=requests.get(f'http://127.0.0.1:{port}/metrics',
                headers={'Authorization':'Bearer '+key},timeout=5)
            response.raise_for_status();values={}
            for line in response.text.splitlines():
                match=re.match(r'^([\w:]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)(?:\s|$)',line)
                if match and match[1] in METRICS:
                    values[match[1]]=values.get(match[1],0)+float(match[2])
            return str(port),{'status':response.status_code,'metrics':values}
        except Exception as exc:return str(port),{'error_type':type(exc).__name__}
    with args.output.open('a',buffering=1) as sink,ThreadPoolExecutor(max_workers=len(ports)) as pool:
        while True:
            try:os.kill(args.monitor_pid,0)
            except ProcessLookupError:break
            start=time.monotonic()
            row={'utc':datetime.now(timezone.utc).isoformat(),'servers':dict(pool.map(scrape,ports))}
            sink.write(json.dumps(row)+'\n')
            time.sleep(max(0,15-(time.monotonic()-start)))


if __name__=='__main__':main()
