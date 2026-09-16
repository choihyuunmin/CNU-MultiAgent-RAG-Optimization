"""Counterbalanced real GPU stage pilot on one owned loopback server at a time.

Never stops existing services. N-gram compatibility overlay is process-local.
Payloads and responses stay private; public rows contain hashes/counts only.
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


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def normalized_tool_calls(message):
    result=[]
    for call in message.get('tool_calls') or []:
        fn=call['function']
        result.append({'name':fn['name'],'arguments':json.loads(fn['arguments'])})
    return result


async def main():
    import httpx
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--compat',type=Path,required=True)
    p.add_argument('--port',type=int,default=28160)
    p.add_argument('--budgets',type=int,nargs='+',default=[0,4])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--levels',type=int,nargs='+',default=[1,4])
    args=p.parse_args()
    if not args.budgets or len(set(args.budgets))!=len(args.budgets) or any(k not in [0,2,4,8,16] for k in args.budgets):
        raise ValueError('invalid draft budgets')
    if args.repeats<1 or args.levels!=sorted(set(args.levels)) or args.levels[0]<1 or args.levels[-1]>4:
        raise ValueError('pilot supports positive sorted loads up to four')
    os.umask(0o077)
    args.directory.mkdir(parents=True,exist_ok=False)
    public=args.directory/'public';public.mkdir()
    inputs=json.loads(args.inputs.read_text())
    warm=[r for r in inputs if r['warmup']];cases=[r for r in inputs if not r['warmup']]
    if len(cases)<max(args.levels) or not warm:raise ValueError('invalid case set')
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],text=True)
    (public/'gpu-before.txt').write_text(gpu_before)
    # A bind check prevents mistakenly probing/killing an existing service.
    with socket.socket() as s:s.bind(('127.0.0.1',args.port))
    plan={'started_utc':datetime.now(timezone.utc).isoformat(),'budgets':args.budgets,
          'repeats':args.repeats,'levels':args.levels,'cases':len(cases),'warmup_cases':len(warm),
          'expected_requests':len(cases)*len(args.budgets)*args.repeats*len(args.levels),
          'inputs_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'real current search-tool inputs, isolated gpt-oss stage; not end-to-end retrieval',
          'gpu':0,'gpu_memory_utilization':.155,'max_model_len':32768,'max_num_seqs':4,
          'enforce_eager':True,'dynamic_k':False,'promote_to_full_experiment':False}
    (public/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    status={'phase':'starting','completed_requests':0}
    def save():
        temp=args.directory/'status.tmp';temp.write_text(json.dumps(status,indent=2)+'\n')
        temp.replace(args.directory/'status.json')
    save()
    rows=(public/'rows.jsonl').open('x',buffering=1)
    private=(args.directory/'private-responses.jsonl').open('x',buffering=1)
    process=None
    async with httpx.AsyncClient(timeout=120,trust_env=False) as client:
        base=f'http://127.0.0.1:{args.port}'
        async def counters():
            response=await client.get(base+'/metrics');response.raise_for_status()
            selected={}
            for line in response.text.splitlines():
                if line.startswith(('vllm:spec_decode_','vllm:num_preemptions_total','vllm:prefix_cache_','vllm:request_success_total')):
                    name=line.split('{',1)[0].split()[0]
                    selected[name]=selected.get(name,0)+float(line.rsplit(' ',1)[1])
            return selected
        async def one(item,k,repeat,level,sem,warmup=False):
            async with sem:
                start=time.perf_counter()
                row={'case_id':item['case_id'],'k':k,'repeat':repeat,'concurrency':level,
                     'input_sha256':digest(item['payload'])}
                try:
                    response=await client.post(base+'/v1/chat/completions',json=item['payload'])
                    response.raise_for_status();value=response.json();choice=value['choices'][0]
                    tools=normalized_tool_calls(choice['message'])
                    valid=(len(tools)==1 and tools[0]['name']=='search_laws')
                    row.update(ok=bool(valid and choice['finish_reason']=='tool_calls'),
                               arguments_exact=bool(valid and tools[0]['arguments']==item['expected_arguments']),
                               output_sha256=digest(tools),finish_reason=choice['finish_reason'],usage=value.get('usage'))
                    if not warmup:private.write(json.dumps({'case_id':item['case_id'],'k':k,'repeat':repeat,'concurrency':level,'response':value},ensure_ascii=False)+'\n')
                except Exception as exc:row.update(ok=False,arguments_exact=False,error_type=type(exc).__name__)
                row['elapsed_s']=time.perf_counter()-start
                if not warmup:
                    rows.write(json.dumps(row)+'\n');status['completed_requests']+=1;save()
                return row
        try:
            for repeat in range(args.repeats):
                order=args.budgets if repeat%2==0 else list(reversed(args.budgets))
                for k in order:
                    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free<24000:raise RuntimeError('insufficient free memory for isolated pilot')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(args.compat),VLLM_PLUGINS='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
                    cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model','openai/gpt-oss-20b',
                         '--host','127.0.0.1','--port',str(args.port),'--gpu-memory-utilization','.155',
                         '--max-model-len','32768','--max-num-seqs','4','--dtype','auto','--enforce-eager',
                         '--enable-auto-tool-choice','--tool-call-parser','openai','--reasoning-parser','openai_gptoss']
                    if k:cmd+=['--speculative-config',json.dumps({'method':'ngram','num_speculative_tokens':k,'prompt_lookup_min':2,'prompt_lookup_max':8})]
                    status.update(phase='loading',repeat=repeat,k=k);save()
                    with (args.directory/f'k{k}-r{repeat}.server.log').open('x') as log:
                        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    (public/f'k{k}-r{repeat}.argv.json').write_text(json.dumps(cmd,indent=2)+'\n')
                    deadline=time.monotonic()+480
                    while True:
                        if process.poll() is not None:raise RuntimeError('isolated server failed to start; inspect private log')
                        try:
                            response=await client.get(base+'/health',timeout=2)
                            if response.status_code==200:break
                        except httpx.HTTPError:pass
                        if time.monotonic()>deadline:raise TimeoutError('isolated server startup timeout')
                        await asyncio.sleep(2)
                    for item in warm:
                        result=await one(item,k,repeat,1,asyncio.Semaphore(1),True)
                        if not result['ok']:raise RuntimeError('warmup tool call failed')
                    for level in args.levels:
                        status.update(phase='measuring',concurrency=level);save()
                        before=await counters();start=time.perf_counter();sem=asyncio.Semaphore(level)
                        trial=await asyncio.gather(*(one(item,k,repeat,level,sem) for item in cases))
                        block={'k':k,'repeat':repeat,'concurrency':level,'wall_s':time.perf_counter()-start,
                               'counters_before':before,'counters_after':await counters()}
                        with (public/'blocks.jsonl').open('a') as f:f.write(json.dumps(block)+'\n')
                        print('trial',k,repeat,level,'mean',sum(r['elapsed_s'] for r in trial)/len(trial),'ok',sum(r['ok'] for r in trial),flush=True)
                        if any(not r['ok'] for r in trial):raise RuntimeError('pilot failure; do not escalate')
                    os.killpg(process.pid,signal.SIGTERM)
                    try:await asyncio.to_thread(process.wait,timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);await asyncio.to_thread(process.wait)
                    process=None
                    await asyncio.sleep(5)
            status['phase']='complete'
        except BaseException as exc:
            status.update(phase='failed',error_type=type(exc).__name__)
            raise
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:await asyncio.to_thread(process.wait,timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);await asyncio.to_thread(process.wait)
            save();rows.close();private.close()
            (public/'gpu-after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],text=True))


if __name__=='__main__':asyncio.run(main())
