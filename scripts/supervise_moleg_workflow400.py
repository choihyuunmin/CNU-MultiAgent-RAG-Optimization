"""One frozen 400-question campaign, owned by a bounded systemd cgroup.

Only registered isolated apps are started/stopped. Existing model processes and
production deployments are observed, never modified. Private raw evidence stays
on the operator host. This is a regression study, not an independent holdout.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys

from monitor_moleg_resources import sample
from check_moleg_data_health import DataHealth

PORTS={8000,8001,8002,8005,8006,8010,8020,8030}
USERS=[16,32,8,4,2,1]


def campaign_spec(root):
    path=root/'run-spec.json'
    spec=json.loads(path.read_text()) if path.exists() else dict(limit=400,start_repeat=0,repeats=3,seed=20260909)
    if set(spec)!=set(['limit','start_repeat','repeats','seed']) or any(type(v) is not int for v in spec.values()):
        raise ValueError('invalid frozen campaign specification')
    if spec not in [dict(limit=400,start_repeat=0,repeats=3,seed=20260909),
                    dict(limit=300,start_repeat=1,repeats=3,seed=20260909)]:
        raise ValueError('campaign specification is not an authorized protocol')
    return spec


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def utc():return datetime.now(timezone.utc).isoformat()


def identity(pid):
    path=Path('/proc')/str(pid)
    argv=path.joinpath('cmdline').read_bytes()
    # Field22 after the parenthesized comm field; survives spaces in comm.
    ticks=path.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
    return hashlib.sha256(argv).hexdigest(),ticks


def model_inventory():
    models=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            argv=path.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if '--port' not in argv or not any('vllm' in s for s in argv):continue
            port=int(argv[argv.index('--port')+1])
            if port not in PORTS:continue
            digest,ticks=identity(int(path.name))
            models.append(dict(pid=int(path.name),port=port,argv_sha256=digest,start_ticks=ticks))
        except (OSError,ValueError,IndexError):continue
    if len(models)!=8 or {m['port'] for m in models}!=PORTS:
        raise RuntimeError('expected eight existing serving processes')
    return models


class Progress:
    def __init__(self,path):
        self.path=path; self.offset=0; self.completed=0; self.sse_ok=0; self.last=None

    def read(self):
        if not self.path.exists():return
        with self.path.open() as source:
            source.seek(self.offset)
            while True:
                position=source.tell(); line=source.readline()
                if not line or not line.endswith('\n'):
                    self.offset=position;break
                row=json.loads(line)
                self.completed+=1;self.sse_ok+=int(row['ok'])
                self.last={k:row[k] for k in ['trial','policy','users','repeat']}


def infrastructure_faults(host,initial_oom,free_bytes):
    reasons=[]
    if host['vm_counters']['oom_kill']>initial_oom:reasons.append('host_oom_kill_increased')
    if host['host_memory_kib']['MemAvailable']<8*1024**2:reasons.append('host_available_ram_below_8gib')
    if free_bytes<10*1024**3:reasons.append('run_filesystem_below_10gib')
    for gpu in host.get('gpus',[]):
        if float(gpu['temperature.gpu'])>=90:reasons.append('gpu_temperature_at_least_90c')
    if host.get('gpu_error_type'):reasons.append('gpu_observation_failed')
    return reasons


async def execute(root):
    import httpx
    os.umask(0o077)
    spec=campaign_spec(root)
    planned=spec['limit']*len(USERS)*3*(spec['repeats']-spec['start_repeat'])
    started=utc()
    state={'status':'preflight','started_utc':started,'planned_requests':planned,'completed_requests':0,
           'scope':str(spec['limit'])+'-question regression, not independent holdout',
           'run_spec':spec,'release_allowed':False}
    status=root/'campaign-status.json'
    write_json(status,state)
    models=model_inventory()
    data_health=DataHealth(root)
    write_json(root/'model-inventory-before.json',models)
    host=await asyncio.to_thread(sample)
    initial_oom=host['vm_counters']['oom_kill']
    scripts=root/'scripts'
    apps=root/'apps'
    app_config=[dict(name=name,slots=slots,emission=emission,port=port,
                    adapter_config=str(root/config)) for name,slots,emission,port,config in [
        ('baseline',4,'original',28340,'observe-final.json'),
        ('fixed16',16,'immediate',28341,'observe-final.json'),
        ('budget',16,'immediate',28342,'budget-final.json')]]
    write_json(root/'arms.json',app_config)
    child=None; active_tasks=[]
    progress=Progress(root/'validation/requests.jsonl')
    log=(root/'execution.log').open('x',buffering=1)

    async def command(argv,env=None):
        nonlocal child
        child=await asyncio.create_subprocess_exec(*argv,cwd=root,stdout=log,stderr=log,env=env)
        code=await child.wait()
        if code:raise RuntimeError('owned_child_exit_'+str(code))

    async def check_models(client):
        async def one(model):
            try:
                if identity(model['pid'])!=(model['argv_sha256'],model['start_ticks']):return False
                response=await client.get('http://127.0.0.1:'+str(model['port'])+'/health')
                return response.status_code==200
            except (OSError,httpx.HTTPError):return False
        return all(await asyncio.gather(*(one(m) for m in models)))

    async def guard(client):
        failures=0
        with (root/'host-resources.jsonl').open('x',buffering=1) as sink:
            while True:
                host=await asyncio.to_thread(sample)
                sink.write(json.dumps(host)+'\n')
                fatal=infrastructure_faults(host,initial_oom,shutil.disk_usage(root).free)
                if fatal:raise RuntimeError(','.join(fatal))
                search_health=await data_health.check()
                healthy=await check_models(client) and search_health['ok']
                failures=0 if healthy else failures+1
                if failures>=3:raise RuntimeError('serving_health_or_identity_failed_three_checks')
                if (root/'STOP').exists():raise RuntimeError('operator_stop_file')
                progress.read()
                state.update(heartbeat_utc=utc(),completed_requests=progress.completed,
                             sse_success=progress.sse_ok,last_completed=progress.last,
                             serving_health_ok=healthy,data_health=search_health)
                write_json(status,state)
                await asyncio.sleep(10)

    async def campaign():
        await command([sys.executable,str(scripts/'manage_moleg_scaling_apps.py'),'start','--directory',str(apps),
            '--app-root',str(root/'pod_source'),'--app-env',str(root/'operator.env'),
            '--serving-env','/data/project/vllm/.env',
            '--proxy-config','/data/project/vllm/litellm/develop_test_vllm_config_v3.yaml',
            '--arms-manifest',str(root/'arms.json')])
        async with httpx.AsyncClient(timeout=3,trust_env=False) as client:
            for attempt in range(60):
                try:
                    for app in app_config:
                        response=await client.get('http://127.0.0.1:'+str(app['port'])+'/__scaling_state')
                        response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt==59:raise RuntimeError('isolated_apps_not_ready')
                    await asyncio.sleep(1)
        state['status']='latency_running';write_json(status,state)
        argv=[sys.executable,str(scripts/'evaluate_moleg_isolated.py')]
        for app in app_config:argv+=['--variant',app['name']+'=http://127.0.0.1:'+str(app['port'])]
        argv+=['--cases',str(root/'cases.json'),'--output',str(root/'validation'),
               '--users',*map(str,USERS),'--repeats',str(spec['repeats']),
               '--start-repeat',str(spec['start_repeat']),'--limit',str(spec['limit']),'--seed',str(spec['seed']),
               '--timeout','240','--private-responses',str(root/'validation.responses.jsonl'),
               '--continue-request-failures','--trace-directory',str(apps)]
        for name,port in [('orchestrator',8000),('worker',8006),('comparison',8030)]:
            argv+=['--metrics',name+'=http://127.0.0.1:'+str(port)+'/metrics']
        await command(argv)
        state['status']='analysis_running';write_json(status,state)
        await command([sys.executable,str(scripts/'analyze_moleg_workflow.py'),'--directory',str(root/'validation'),
                       '--workflow-directory',str(apps)])
        await command([sys.executable,str(scripts/'summarize_moleg_resources.py'),'--directory',str(root/'validation'),
                       '--host',str(root/'host-resources.jsonl'),'--output',str(root/'validation/resources.json')])
        await command([sys.executable,str(scripts/'prepare_moleg_workflow_judge.py'),'--directory',str(root),
                       '--cases',str(root/'cases.json'),'--repeat',str(spec['start_repeat'])])
        state['status']='answer_judge_running';write_json(status,state)
        env={**os.environ,'MOLEG_RAG_ROOT':str(root/'pod_source'),'MOLEG_SERVING_ENV':'/data/project/vllm/.env',
             'MOLEG_PROXY_CONFIG':'/data/project/vllm/litellm/develop_test_vllm_config_v3.yaml'}
        for users in USERS:
            await command([sys.executable,str(scripts/'evaluate_moleg_answer_judge.py'),'--directory',str(root/f'judge-u{users}'),
                           '--judge-url','http://127.0.0.1:8002/v1','--judge-model','microsoft/phi-4',
                           '--repeat',str(spec['start_repeat'])],env=env)
        state['status']='completed_evaluation_not_released'

    try:
        async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
            data_preflight=await data_health.check(full=True)
            write_json(root/'supervised-data-preflight.json',data_preflight)
            if infrastructure_faults(host,initial_oom,shutil.disk_usage(root).free) or not await check_models(client) or not data_preflight['ok']:
                raise RuntimeError('infrastructure_preflight_failed')
            active_tasks=[asyncio.create_task(campaign()),asyncio.create_task(guard(client))]
            done,_=await asyncio.wait(active_tasks,return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
            if not active_tasks[0].done():raise RuntimeError('guard_ended_unexpectedly')
    except BaseException as exc:
        state.update(status='stopped_incomplete',stop_type=type(exc).__name__,
                     stop_reason=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__)
    finally:
        for task in active_tasks:task.cancel()
        await asyncio.gather(*active_tasks,return_exceptions=True)
        if child and child.returncode is None:
            child.terminate()
            try:await asyncio.wait_for(child.wait(),10)
            except TimeoutError:child.kill();await child.wait()
        if (apps/'apps.json').exists():
            proc=await asyncio.create_subprocess_exec(sys.executable,str(scripts/'manage_moleg_scaling_apps.py'),
                'stop','--directory',str(apps),stdout=log,stderr=log)
            await proc.wait()
        (root/'operator.env').unlink(missing_ok=True)
        progress.read()
        state.update(ended_utc=utc(),completed_requests=progress.completed,sse_success=progress.sse_ok,
                     temporary_environment_removed=not (root/'operator.env').exists())
        write_json(status,state)
        log.close()
    return 0 if state['status']=='completed_evaluation_not_released' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    with asyncio.Runner() as runner:
        loop=runner.get_loop()
        task=loop.create_task(execute(args.directory.resolve()))
        loop.add_signal_handler(signal.SIGTERM,task.cancel)
        return runner.run(_await_task(task))


async def _await_task(task):return await task


if __name__=='__main__':sys.exit(main())
