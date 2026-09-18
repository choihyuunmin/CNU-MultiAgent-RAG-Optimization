"""Small held-out end-to-end confirmation after the fixed-K stage pilot.

Two unchanged isolated app arms share one owned worker endpoint; its engine is
recreated in ABBA order. Existing production servers are never stopped.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from supervise_moleg_sweep import launch_app, wait_ready, run_trial, wait_models_quiet


async def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:await asyncio.to_thread(process.wait,timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            await asyncio.to_thread(process.wait)


async def main():
    import httpx
    from dotenv import dotenv_values
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--gpu-python',required=True)
    p.add_argument('--app-python',required=True)
    p.add_argument('--compat',type=Path,required=True)
    p.add_argument('--app-env',type=Path,required=True)
    p.add_argument('--serving-env',type=Path,required=True)
    p.add_argument('--proxy-config',type=Path,required=True)
    p.add_argument('--name',default='e2e-k4')
    args=p.parse_args()
    os.umask(0o077)
    root=args.root.resolve();out=root/args.name;out.mkdir(mode=0o700,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',28160))
    base='http://127.0.0.1:28160'
    all_cases=json.loads((args.previous_root/'cases.json').read_text())
    excluded={r['case_id'] for r in json.loads((root/'worker-inputs.json').read_text())}
    # Source-labelled cases allow a known-item check in this small confirmation.
    eligible=[c for c in all_cases if c['case_id'] not in excluded and c.get('source_law_id')]
    if len(eligible)<9:raise ValueError('need nine held-out labelled cases')
    warmup=eligible[:1];cases=eligible[1:9]
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False)+'\n')
    config={'adapter_config':'combined.adapter.json','structured_harness':'portable.harness.json',
            'continuation_harness':'repair.continuation.json','worker_experiment_base':base+'/v1'}
    arms=[dict(config,name='baseline',port=28760),dict(config,name='improved',port=28761)]
    plan={'utc':datetime.now(timezone.utc).isoformat(),'kind':'held_out_end_to_end_worker_speculation',
          'levels':[1,4],'repeats':2,'budgets':[0,4],'expected_requests':64,
          'arms':arms,'cases':[{'case_id':c['case_id'],'question_sha256':hashlib.sha256(c['question'].encode()).hexdigest()} for c in cases],
          'stage_pilot_cases_excluded':True,'engine_context_limit':32768,'engine_max_num_seqs':4,
          'engine_gpu_memory_utilization':.155,'engine_enforce_eager':True,
          'runtime_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                            for folder in ('src','scripts') for f in sorted((root/folder).rglob('*.py'))},
          'configs':{name:json.loads((root/name).read_text()) for name in config.values() if name.endswith('.json')},
          'promote_to_full_experiment':False}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model_headers={'Authorization':'Bearer '+dotenv_values(args.serving_env)['VLLM_API_KEY']}
    metrics={'orchestrator':'http://127.0.0.1:8000/metrics','worker':base+'/metrics'}
    state={'phase':'starting','completed_requests':0};apps=[];server=None;trials=[]
    def save():
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    save()
    sinks={name:(out/name).open('x',buffering=1) for name in ['requests.jsonl','private-responses.jsonl','telemetry.jsonl','warmup.jsonl','warmup-private.jsonl','warmup-telemetry.jsonl']}
    try:
        for arm in arms:
            process,info=launch_app(arm,root,args.app_python,args.previous_root/'pod_source',args.app_env,args.serving_env,args.proxy_config,out)
            apps.append(process)
            await wait_ready(f'http://127.0.0.1:{arm["port"]}',process=process)
        async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
            for repeat in range(2):
                order=[0,4] if repeat==0 else [4,0]
                for k in order:
                    arm=arms[0 if k==0 else 1]
                    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free<24000:raise RuntimeError('insufficient free memory')
                    cmd=[args.gpu_python,'-m','vllm.entrypoints.openai.api_server','--model','openai/gpt-oss-20b',
                         '--host','127.0.0.1','--port','28160','--gpu-memory-utilization','.155',
                         '--max-model-len','32768','--max-num-seqs','4','--dtype','auto','--enforce-eager',
                         '--enable-auto-tool-choice','--tool-call-parser','openai','--reasoning-parser','openai_gptoss']
                    if k:cmd+=['--speculative-config',json.dumps({'method':'ngram','num_speculative_tokens':k,'prompt_lookup_min':2,'prompt_lookup_max':8})]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(args.compat),VLLM_PLUGINS='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                    with (out/f'k{k}-r{repeat}.server.log').open('x') as log:
                        server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(phase='loading',k=k,repeat=repeat);save()
                    deadline=time.monotonic()+480
                    while True:
                        if server.poll() is not None:raise RuntimeError('owned model failed to load')
                        try:
                            if (await client.get(base+'/health')).status_code==200:break
                        except httpx.HTTPError:pass
                        if time.monotonic()>deadline:raise TimeoutError('model readiness')
                        await asyncio.sleep(2)
                    app_base=f'http://127.0.0.1:{arm["port"]}'
                    await run_trial(app_base,warmup,1,arm['name'],-1,-1,240,30,1,
                        sinks['warmup.jsonl'],sinks['warmup-private.jsonl'],sinks['warmup-telemetry.jsonl'],metrics,model_headers)
                    for level in [1,4]:
                        await wait_models_quiet(metrics,model_headers)
                        state.update(phase='measuring',level=level);save()
                        trial=await run_trial(app_base,cases,level,arm['name'],repeat,len(trials),240,30,1,
                            sinks['requests.jsonl'],sinks['private-responses.jsonl'],sinks['telemetry.jsonl'],metrics,model_headers)
                        trials.append(trial);state['completed_requests']+=trial['n'];save()
                        (out/'summary.json').write_text(json.dumps({'complete':False,'trials':trials},indent=2)+'\n')
                        if trial['pipeline_success']!=len(cases):raise RuntimeError('pipeline error; do not escalate')
                    await wait_models_quiet(metrics,model_headers)
                    await stop(server);server=None;await asyncio.sleep(5)
        state['phase']='complete'
        (out/'summary.json').write_text(json.dumps({'complete':True,'trials':trials},indent=2)+'\n')
    except BaseException as exc:
        state.update(phase='failed',error_type=type(exc).__name__)
        raise
    finally:
        await stop(server)
        for process in apps:await stop(process)
        for sink in sinks.values():sink.close()
        save()


if __name__=='__main__':asyncio.run(main())
