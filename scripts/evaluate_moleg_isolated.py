"""Sequential isolated-app ablations using the unchanged SSE load primitive.

The legacy 'policy' column identifies the app variant, not an outer gate here.
All variants admit up to U external users. No model or deployment is changed.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

from evaluate_moleg_scaling import digest, select_cases, trial


def utc():
    return datetime.now(timezone.utc).isoformat()


def variant_order(variants, users, users_now, repeat):
    shift = (list(users).index(users_now) + repeat) % len(variants)
    names = list(variants)
    return names[shift:] + names[:shift]


def trial_schedule(variants, users, repeats, start_repeat=0):
    """Reverse load order without accidentally cancelling within-load arm rotation."""
    for repeat in range(start_repeat, repeats):
        loads = list(users)
        if repeat % 2:
            loads.reverse()
        for users_now in loads:
            for name in variant_order(variants, users, users_now, repeat):
                yield repeat, users_now, name


class VariantSink:
    def __init__(self, sink, variant):
        self.sink, self.variant = sink, variant

    def write(self, line):
        row = json.loads(line)
        row['policy'] = self.variant
        self.sink.write(json.dumps(row, ensure_ascii=False) + '\n')


async def run(args):
    import httpx
    variants = dict(x.split('=', 1) for x in args.variant)
    if len(variants) != len(args.variant) or 'baseline' not in variants:
        raise ValueError('unique variants including baseline are required')
    cases = json.loads(args.cases.read_text())
    if len({c['question'] for c in cases}) != len(cases):
        raise ValueError('duplicate questions')
    selected = select_cases(cases, args.limit, args.seed)
    if max(args.users) > len(selected):
        raise ValueError('need at least as many cases as users')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'complete': False, 'trials': []}
    (args.output/'summary.json').write_text(json.dumps(state)+'\n')
    manifest = {'utc': utc(), 'scope': 'isolated loopback app variants; no outer admission gate',
        'users': args.users, 'rates': None, 'policies': list(variants), 'repeats': args.repeats,
        'start_repeat': getattr(args, 'start_repeat', 0),
        'seed': args.seed, 'timeout_s': args.timeout, 'slo_s': args.slo,
        'sample_interval_s': args.sample_interval,
        'selected_cases': [{k: c[k] for k in ['case_id', 'kind']} |
                           {'question_sha256': digest(c['question'])} for c in selected],
        'input_file_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'request_primitive_sha256': hashlib.sha256(Path(__file__).with_name('evaluate_moleg_scaling.py').read_bytes()).hexdigest(),
        'policy_column_means': 'isolated application variant; no adaptive outer gate',
        'failure_rule': ('retain individual request failures and continue; external supervisor stops on infrastructure faults'
                         if getattr(args,'continue_request_failures',False) else 'stop after a trial with any incomplete SSE response'),
        'planned_order': [{'repeat': r, 'users': u, 'policy': n}
                          for r,u,n in trial_schedule(variants, args.users, args.repeats, getattr(args,'start_repeat',0))],
        'limitations': ['finite workload including startup/drain', 'shared warm model caches',
                        'silver source labels, not expert answer correctness']}
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        manifest['app_states_before'] = {}
        for name, base in variants.items():
            response = await client.get(base.rstrip('/') + '/__scaling_state')
            response.raise_for_status()
            data = response.json()
            if data['active'] or data['queued']:
                raise RuntimeError('isolated app is already busy')
            manifest['app_states_before'][name] = data
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        with (args.output / 'requests.jsonl').open('x', buffering=1) as requests, \
             (args.output / 'telemetry.jsonl').open('x', buffering=1) as telemetry, \
             (args.output / 'app-state.jsonl').open('x', buffering=1) as app_sink:
            for repeat in range(getattr(args,'start_repeat',0), args.repeats):
                ordered = list(selected)
                random.Random(args.seed + repeat).shuffle(ordered)
                loads = list(args.users)
                if repeat % 2:
                    loads.reverse()
                for load_index, users in enumerate(loads):
                    names = variant_order(variants, args.users, users, repeat)
                    for name in names:
                        args.base = variants[name]
                        args.variant_name = name
                        trial_id = len(state['trials'])
                        stop = asyncio.Event()
                        async def observe_app():
                            while not stop.is_set():
                                row = {'utc': utc(), 'trial': trial_id, 'policy': name,
                                       'repeat': repeat, 'users': users}
                                try:
                                    response = await client.get(args.base.rstrip('/') + '/__scaling_state')
                                    response.raise_for_status()
                                    row['state'] = response.json()
                                except Exception as exc:
                                    row['error_type'] = type(exc).__name__
                                app_sink.write(json.dumps(row) + '\n')
                                try:
                                    await asyncio.wait_for(stop.wait(), 1)
                                except TimeoutError:
                                    pass
                        started = utc()
                        observer = asyncio.create_task(observe_app())
                        try:
                            entry = await trial(args, ordered, users, None, 'baseline', repeat,
                                trial_id, VariantSink(requests, name), VariantSink(telemetry, name))
                        finally:
                            stop.set()
                            await observer
                        entry.update(policy=name, started_utc=started, ended_utc=utc())
                        if getattr(args,'trace_directory',None):
                            await wait_idle(client,args.base)
                            audit = audit_pipeline_trace(args.trace_directory/(name+'.trace.jsonl'),ordered,trial_id,name,users,repeat)
                            with (args.output/'pipeline-audit.jsonl').open('a') as health_sink:
                                for row in audit:
                                    health_sink.write(json.dumps(row)+'\n')
                            entry['traced_internal_error_requests'] = sum(r['internal_error_count']>0 for r in audit)
                            entry['traced_execute_returned'] = sum(r['execute_returned'] for r in audit)
                        state['trials'].append(entry)
                        (args.output / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')
                        print(json.dumps({'variant': name, 'users': users, 'repeat': repeat,
                            'success': entry['success'], 'n': entry['n'],
                            'mean_s': entry['metrics']['elapsed_s']['mean']}), flush=True)
                        if not getattr(args,'continue_request_failures',False) and (entry['success'] < entry['n'] or entry.get('pipeline_success', entry['n']) < entry['n']):
                            raise RuntimeError('failed requests; retained trial and stopped campaign')
    state['complete'] = True
    (args.output / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')


async def wait_idle(client, base, timeout=60):
    deadline = asyncio.get_running_loop().time()+timeout
    while asyncio.get_running_loop().time()<deadline:
        response = await client.get(base.rstrip('/')+'/__scaling_state')
        response.raise_for_status()
        state = response.json()
        gates = (state.get('adapter') or {}).get('models',{}).values()
        if not state['active'] and not state['queued'] and all(not g['active'] and not g['queued'] for g in gates):
            await asyncio.sleep(.2)  # allow final trace flush after HTTP task closure
            return
        await asyncio.sleep(1)
    raise RuntimeError('isolated app did not drain after request cancellation')


def audit_pipeline_trace(path,cases,trial_id,policy,users,repeat):
    groups = {}
    for line in path.open():
        record = json.loads(line)
        prefix,index = record['session_id'].rsplit('-',1)
        group = groups.setdefault(prefix,{})
        if int(index) in group:
            raise ValueError('duplicate traced request')
        group[int(index)] = record
    records = list(groups.values())[-1] if groups else {}
    if set(records)!=set(range(len(cases))):
        raise ValueError('incomplete pipeline trace; do not hide missing health records')
    return [dict(trial=trial_id,policy=policy,users=users,repeat=repeat,index=i,
                 case_id=case['case_id'],execute_returned='response_chars' in records[i],
                 internal_error_count=len(records[i].get('errors',[])),
                 internal_error_types=sorted({str(e.get('error','unknown')) for e in records[i].get('errors',[])}))
            for i,case in enumerate(cases)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True, help='name=http://loopback:port')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics', action='append', default=[])
    parser.add_argument('--users', type=int, nargs='+', default=[16])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--start-repeat', type=int, default=0,
                        help='first original repeat index; use only in a NEW separately reported recovery segment')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--sample-interval', type=float, default=2)
    parser.add_argument('--slo', type=float, default=30)
    parser.add_argument('--private-responses', type=Path,
                        help='optional NEW protected response file for blinded quality assessment')
    parser.add_argument('--continue-request-failures',action='store_true',
                        help='only with an external infrastructure supervisor; no failure exclusion')
    parser.add_argument('--trace-directory',type=Path,help='private app traces for fallback audit after every trial')
    args = parser.parse_args()
    if min(args.users) < 1 or args.repeats < 1 or min(args.timeout, args.sample_interval, args.slo) <= 0:
        parser.error('positive users, repeats and times required')
    if not 0 <= args.start_repeat < args.repeats:
        parser.error('start-repeat must be within the repeat range')
    args.api_key_env, args.initial_limit, args.open_max_active = None, 8, max(args.users)
    args.arrivals = 'poisson'
    args.private_response_sink = None
    if args.private_responses:
        args.private_responses.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.private_responses, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        args.private_response_sink = os.fdopen(fd, 'w', buffering=1)
    try:
        asyncio.run(run(args))
    finally:
        if args.private_response_sink:
            args.private_response_sink.close()


if __name__ == '__main__':
    main()
