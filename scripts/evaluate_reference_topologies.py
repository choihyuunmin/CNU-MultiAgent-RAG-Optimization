"""Synthetic GPU portability study, distinct from the real MOLEG RAG evaluation.

No expert/real-world task accuracy claim: generated records have exact predicate
oracles. The oracle scores AFTER execution; runtime validation only checks shape,
reference membership and uniqueness. Actual model results drive successor inputs.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cnu_rag_optimization.reference_harness import (
    Calibration, CalibrationPair, ReferenceContract, ReferenceHarness, calibrate,
)

DOMAINS = {'catalog': ('products', 'sku'), 'incidents': ('tickets', 'ticket_id'),
           'bibliography': ('articles', 'accession')}
INSTRUCTION = ('You are a record filtering agent. Apply the supplied rule to the input records. '
               'Return ALL and ONLY matching record identifiers in original input order in '
               'selected_ids. Copy identifiers exactly. Never invent records. No explanations.')
SCHEMA = {'type': 'object', 'properties': {'selected_ids': {'type': 'array', 'items': {'type': 'string'}}},
          'required': ['selected_ids'], 'additionalProperties': False}
RULES = {
    'topic': 'Select every record whose topic equals blue.',
    'active': 'Select every record whose active equals true.',
    'budget': 'Select every record whose cost is at most 50.',
}


def expected(records, rule):
    return [r for r in records if {'topic': r['topic'] == 'blue',
            'active': r['active'], 'budget': r['cost'] <= 50}[rule]]


def make_case(domain, index, split):
    collection, identifier = DOMAINS[domain]
    rng = random.Random(f'{split}/{domain}/{index}')
    records = [{identifier: str(uuid.UUID(int=rng.getrandbits(128))),
                'topic': rng.choice(['blue', 'green']), 'active': rng.choice([True, False]),
                'cost': rng.randrange(1, 101), 'description': f'{domain} record {i} with unchanged text.'}
               for i in range(24)]
    # Positive and empty output cases are both possible; no answer in prompt.
    return {'domain': domain, 'index': index, 'split': split, 'records': records}


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values)-1)*q)] if values else None


async def run(args):
    from moleg_paper_runtime import bootstrap
    from openai import AsyncOpenAI
    _, serving, config, resolve = bootstrap()
    p = next(x['litellm_params'] for x in config['model_list'] if x['model_name'] == args.model_alias)
    model = p['model'][7:]
    root = args.output
    root.mkdir(mode=0o700, exist_ok=False)
    identity = hashlib.sha256(json.dumps([model, INSTRUCTION, SCHEMA, RULES, DOMAINS], sort_keys=True).encode()).hexdigest()
    contracts = {domain: ReferenceContract(identity + '/' + domain, (collection,), identifier, ('selected_ids',))
                 for domain, (collection, identifier) in DOMAINS.items()}
    original = {d: ReferenceHarness(Calibration(c.identity, False, 0, 0, 1, 'control')) for d, c in contracts.items()}
    ungated = {d: ReferenceHarness(Calibration(c.identity, True, 0, 0, 1, 'calibration_candidate')) for d, c in contracts.items()}
    cal_cases = [make_case(d, i, 'calibration') for d in DOMAINS for i in range(12)]
    heldout = [make_case(d, i, 'heldout') for d in DOMAINS for i in range(6)]
    plan = {'schema_version': 1, 'synthetic': True, 'model': model, 'identity': identity,
            'calibration_cases': 36, 'heldout_cases': 18, 'topologies': ['chain', 'fork_join', 'loop'],
            'repeats': 2, 'workflow_concurrency': args.concurrency,
            'timing_includes_fallbacks': True, 'runtime_semantic_oracle': False,
            'gate': {'min_cases': 12, 'min_reduction': .02, 'max_excess_loss': .01},
            'source_sha256': {str(f.name): hashlib.sha256(f.read_bytes()).hexdigest() for f in [
                Path(__file__), Path(__file__).resolve().parents[1]/'src/cnu_rag_optimization/reference_harness.py']}}
    (root/'plan.json').write_text(json.dumps(plan, indent=2))
    (root/'cases.json').write_text(json.dumps({'calibration': cal_cases, 'heldout': heldout}, indent=2))
    sink = (root/'requests.jsonl').open('x', buffering=1)
    rows = []
    async with AsyncOpenAI(base_url=resolve(p['api_base']), api_key=resolve(p.get('api_key')) or serving['VLLM_API_KEY'],
                           max_retries=0, timeout=120) as client:
        async def workflow(case, topology, harness):
            domain = case['domain']
            collection, identifier = DOMAINS[domain]
            calls, fallbacks, transformed, prompt_tokens, output_tokens = 0, 0, 0, 0, 0
            async def node(records, rule):
                nonlocal calls, fallbacks, transformed, prompt_tokens, output_tokens
                raw = json.dumps({collection: records}, ensure_ascii=False, indent=2)
                async def invoke(content):
                    nonlocal calls, prompt_tokens, output_tokens
                    calls += 1
                    response = await client.chat.completions.create(model=model, temperature=0, max_tokens=2048,
                        messages=[{'role':'system','content': INSTRUCTION},
                                  {'role':'user','content': RULES[rule]+'\n'+content}],
                        response_format={'type':'json_schema','json_schema': {'name':'reference_selection', 'strict':True, 'schema':SCHEMA}})
                    if response.usage:
                        prompt_tokens += response.usage.prompt_tokens
                        output_tokens += response.usage.completion_tokens
                    if response.choices[0].finish_reason != 'stop':
                        raise ValueError('incomplete_output')
                    return response.choices[0].message.content or ''
                def validate(output, inputs):
                    ids = output.get('selected_ids') if isinstance(output, dict) else None
                    allowed = {x[identifier] for x in inputs[collection]}
                    return (set(output) == {'selected_ids'} and isinstance(ids, list)
                            and all(isinstance(x, str) and x in allowed for x in ids)
                            and len(ids) == len(set(ids)))
                result = await harness.run(raw, contracts[domain], invoke=invoke, validate=validate,
                                           protected=(INSTRUCTION, RULES[rule]))
                fallbacks += result.fallback
                transformed += result.transformed
                ids = set(json.loads(result.output)['selected_ids'])
                return [r for r in records if r[identifier] in ids]
            records = case['records']
            if topology == 'single':
                output = await node(records, 'topic')
                oracle = expected(records, 'topic')
            elif topology == 'chain':
                output, oracle = records, records
                for rule in ['topic', 'active', 'budget']:
                    output = await node(output, rule)
                    oracle = expected(oracle, rule)
            elif topology == 'fork_join':
                left, right = await asyncio.gather(node(records, 'topic'), node(records, 'active'))
                common = {r[identifier] for r in left} & {r[identifier] for r in right}
                output = await node([r for r in records if r[identifier] in common], 'budget')
                oracle = expected(expected(expected(records, 'topic'), 'active'), 'budget')
            else:
                # A model result determines whether the next refinement runs.
                output, oracle = records, records
                for rule in ['topic', 'active', 'budget']:
                    output = await node(output, rule)
                    if len(output) <= 4:
                        break
                for rule in ['topic', 'active', 'budget']:
                    oracle = expected(oracle, rule)
                    if len(oracle) <= 4:
                        break
            actual, truth = {r[identifier] for r in output}, {r[identifier] for r in oracle}
            return {'exact': actual == truth, 'recall': len(actual & truth)/len(truth) if truth else float(not actual),
                    'precision': len(actual & truth)/len(actual) if actual else float(not truth),
                    'calls': calls, 'fallbacks': fallbacks, 'transformed': transformed,
                    'prompt_tokens': prompt_tokens, 'output_tokens': output_tokens}

        async def block(cases, topology, mode, repeat, harnesses):
            sem = asyncio.Semaphore(args.concurrency)
            start = time.perf_counter()
            order = list(cases)
            random.Random(20260916+repeat).shuffle(order)
            async def one(case):
                # Fixed client concurrency; latency starts when the simulated user submits.
                async with sem:
                    started = time.perf_counter()
                    row = {k:case[k] for k in ['domain','index','split']}
                    row.update(topology=topology, mode=mode, repeat=repeat, ok=False)
                    try:
                        row.update(await workflow(case, topology, harnesses[case['domain']]), ok=True)
                    except Exception as exc:
                        row.update(error_type=type(exc).__name__, exact=False, recall=0, precision=0)
                    row['elapsed_s'] = time.perf_counter()-started
                    rows.append(row)
                    sink.write(json.dumps(row)+'\n')
            await asyncio.gather(*(one(c) for c in order))
            subset = [r for r in rows if r['split']==cases[0]['split'] and r['topology']==topology and r['mode']==mode and r['repeat']==repeat]
            summary = {'phase':cases[0]['split'], 'topology':topology,'mode':mode,'repeat':repeat,
                       'n':len(subset), 'ok':sum(r['ok'] for r in subset),'exact':sum(r['exact'] for r in subset),
                       'mean_s':statistics.fmean(r['elapsed_s'] for r in subset),'wall_s':time.perf_counter()-start}
            print(json.dumps(summary), flush=True)
            return summary

        trials = []
        for mode, repeat, harnesses in [('original',0,original),('candidate',0,ungated),('candidate',1,ungated),('original',1,original)]:
            trials.append(await block(cal_cases, 'single', mode, repeat, harnesses))
        gates = {}
        for domain in DOMAINS:
            pairs = []
            for index in range(12):
                selected = [r for r in rows if r['domain']==domain and r['index']==index]
                a, b = [[r for r in selected if r['mode']==mode] for mode in ['original','candidate']]
                pairs.append(CalibrationPair(str(index), statistics.fmean(r['elapsed_s'] for r in a),
                    statistics.fmean(r['elapsed_s'] for r in b), statistics.fmean(not r['exact'] for r in a),
                    statistics.fmean(not r['exact'] for r in b)))
            gates[domain] = calibrate(contracts[domain].identity, pairs)
        (root/'calibration.json').write_text(json.dumps({d:asdict(g) for d,g in gates.items()}, indent=2))
        print(json.dumps({'calibration':{d:asdict(g) for d,g in gates.items()}}), flush=True)
        calibrated = {d:ReferenceHarness(g) for d,g in gates.items()}
        for repeat in range(2):
            for ti, topology in enumerate(['chain','fork_join','loop']):
                order = [('original',original),('calibrated',calibrated)]
                if (repeat+ti)%2:
                    order.reverse()
                for mode,harnesses in order:
                    trials.append(await block(heldout, topology, mode, repeat, harnesses))
    sink.close()
    summaries = []
    for topology in ['chain','fork_join','loop']:
        arms = {}
        for mode in ['original','calibrated']:
            subset = [r for r in rows if r['split']=='heldout' and r['topology']==topology and r['mode']==mode]
            walls = [r['wall_s'] for r in trials if r['phase']=='heldout' and r['topology']==topology and r['mode']==mode]
            arms[mode] = {'n':len(subset), 'ok':sum(r['ok'] for r in subset), 'exact':sum(r['exact'] for r in subset),
                          'mean_s':statistics.fmean(r['elapsed_s'] for r in subset),
                          'p95_s':percentile([r['elapsed_s'] for r in subset],.95),
                          'rps':sum(r['ok'] for r in subset)/sum(walls),
                          'model_calls':sum(r.get('calls',0) for r in subset),
                          'fallbacks':sum(r.get('fallbacks',0) for r in subset),
                          'transformed':sum(r.get('transformed',0) for r in subset),
                          'prompt_tokens':sum(r.get('prompt_tokens',0) for r in subset),
                          'output_tokens':sum(r.get('output_tokens',0) for r in subset)}
        summaries.append({'topology':topology,'arms':arms,
                          'mean_reduction_pct':100*(1-arms['calibrated']['mean_s']/arms['original']['mean_s'])})
    (root/'summary.json').write_text(json.dumps({'synthetic':True,'trials':trials,'topologies':summaries}, indent=2))
    print(json.dumps({'phase':'complete', 'requests':len(rows)}), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-alias',required=True)
    parser.add_argument('--concurrency',type=int,default=4)
    asyncio.run(run(parser.parse_args()))
