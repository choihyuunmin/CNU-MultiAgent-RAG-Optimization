"""Offline oracle simulation of prompt-lookup speculative decoding policies.

Uses frozen real prompts (chat-template rendered, tokenized with the served
model's tokenizer) and the real greedy outputs logged from the production
model. Because greedy decoding is deterministic, the number of verifier steps
under any proposer policy can be computed exactly without a GPU.
"""
import argparse, json, hashlib, statistics, sys, collections, itertools
from pathlib import Path

def build_index(seq, min_n, max_n, index=None, start=0, source=0):
    """index[(n, ngram)] -> list of (source, start_pos) in insertion order."""
    if index is None: index = {}
    for j in range(start, len(seq)):
        for n in range(min_n, max_n+1):
            if j-n+1 < 0: continue
            key = (n, tuple(seq[j-n+1:j+1]))
            index.setdefault(key, []).append((source, j-n+1))
    return index

class Context:
    """Incrementally indexed prompt+generated token sequence."""
    def __init__(self, tokens, min_n, max_n):
        self.seq = list(tokens); self.min_n=min_n; self.max_n=max_n
        self.index = build_index(self.seq, min_n, max_n)
    def append(self, tok):
        self.seq.append(tok); build_index(self.seq, self.min_n, self.max_n, self.index, len(self.seq)-1)
    def lookup(self, n, order):
        L = len(self.seq)
        key = (n, tuple(self.seq[L-n:]))
        occ = self.index.get(key, [])
        # exclude the suffix itself (start == L-n); vLLM requires start+n <= L-1
        cands = [p for _,p in occ if p+n <= L-1]
        if not cands: return None
        return cands[0] if order=='earliest' else cands[-1]

def propose(ctx, k, order, datastore=None, exclude_case=None, mode='ctx_then_ds'):
    """Return draft token list following vLLM semantics (longest n first)."""
    L = len(ctx.seq)
    if k <= 0: return [], None
    for n in range(min(ctx.max_n, L), ctx.min_n-1, -1):
        pos = ctx.lookup(n, order)
        if pos is not None:
            return ctx.seq[pos+n: pos+n+k], 'ctx'
        if datastore is not None and mode=='longest_any':
            d = ds_lookup(datastore, ctx, n, k, exclude_case, order)
            if d: return d, 'ds'
    if datastore is not None and mode=='ctx_then_ds':
        for n in range(min(ctx.max_n, L), ctx.min_n-1, -1):
            d = ds_lookup(datastore, ctx, n, k, exclude_case, order)
            if d: return d, 'ds'
    return [], None

def ds_lookup(ds, ctx, n, k, exclude_case, order):
    L=len(ctx.seq); key=(n, tuple(ctx.seq[L-n:]))
    occ = ds['index'].get(key)
    if not occ: return None
    cands=[(src,p) for src,p in occ if src!=exclude_case]
    if not cands: return None
    src,p = cands[0] if order=='earliest' else cands[-1]
    seq = ds['seqs'][src]
    return seq[p+n:p+n+k]

def simulate(prompt, output, k, min_n, max_n, order, datastore=None, exclude_case=None, mode='ctx_then_ds'):
    ctx = Context(prompt, min_n, max_n)
    t = 0; steps = 0; proposed = accepted = 0; steps_with_draft = 0
    src_count = collections.Counter(); acc_by_src = collections.Counter()
    # first token comes from the prefill step
    ctx.append(output[0]); t = 1; steps = 1
    while t < len(output):
        draft, src = propose(ctx, min(k, len(output)-t+8), order, datastore, exclude_case, mode)
        a = 0
        if draft:
            steps_with_draft += 1; proposed += len(draft); src_count[src]+=1
            for d in draft:
                if t + a < len(output) and d == output[t+a]: a += 1
                else: break
            accepted += a; acc_by_src[src]+=a
        gain = a + 1
        for j in range(gain):
            if t + j < len(output): ctx.append(output[t+j])
        t += gain; steps += 1
    return {'tokens': len(output), 'steps': steps, 'proposed': proposed, 'accepted': accepted,
            'steps_with_draft': steps_with_draft, 'ctx_steps': src_count['ctx'], 'ds_steps': src_count['ds'],
            'ctx_acc': acc_by_src['ctx'], 'ds_acc': acc_by_src['ds']}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--variant', default='proxy'); ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    reqs = {(r['case_id'], r['stage']): r for r in json.loads(args.requests.read_text())}
    rows = [json.loads(l) for l in args.rows.read_text().splitlines()]
    rows = [r for r in rows if r['variant']==args.variant and r['repeat']==args.repeat and r['ok']]
    if args.limit: rows = rows[:args.limit]
    cases = []
    for r in rows:
        req = reqs[(r['case_id'], r['stage'])]['payload']
        text = tok.apply_chat_template(req['messages'], tokenize=False, add_generation_prompt=True,
                                       **req.get('chat_template_kwargs', {}))
        prompt = tok.encode(text, add_special_tokens=False)
        out = tok.encode(r['answer'], add_special_tokens=False) + [106]  # <end_of_turn>
        cases.append({'case_id': r['case_id'], 'stage': r['stage'], 'prompt': prompt, 'output': out,
                      'usage_completion': r['usage']['completion_tokens']})
    stages = sorted(set(c['stage'] for c in cases))
    print('cases', len(cases), {s: sum(c['stage']==s for c in cases) for s in stages}, file=sys.stderr)
    # tokenizer sanity: our re-tokenized output length vs served completion_tokens
    diff = [len(c['output'])-c['usage_completion'] for c in cases]
    print('retokenized-vs-usage length diff: mean', statistics.mean(diff), 'max', max(diff), 'min', min(diff), file=sys.stderr)
    # datastores per stage: all outputs of the same stage (leave-one-out applied at lookup)
    results = {'meta': {'cases': len(cases), 'stages': stages, 'length_diff_mean': statistics.mean(diff),
                        'prompt_tokens_mean': {s: statistics.mean(len(c['prompt']) for c in cases if c['stage']==s) for s in stages},
                        'output_tokens_mean': {s: statistics.mean(len(c['output']) for c in cases if c['stage']==s) for s in stages}},
               'policies': []}
    grid = []
    for (mn, mx) in [(2,4),(1,3),(3,5),(2,8),(4,8),(3,12)]:
        for k in [3,5,8,12,16,24]:
            for order in ['earliest','latest']:
                grid.append(dict(min_n=mn,max_n=mx,k=k,order=order,ds=False))
    for (mn,mx) in [(2,4),(3,5),(2,8)]:
        for k in [5,8,12,16]:
            for order in ['earliest','latest']:
                for mode in ['ctx_then_ds','longest_any']:
                    grid.append(dict(min_n=mn,max_n=mx,k=k,order=order,ds=True,mode=mode))
    ds_cache = {}
    for g in grid:
        per_stage = {}
        for s in stages:
            sub = [c for c in cases if c['stage']==s]
            ds = None
            if g['ds']:
                key=(s,g['min_n'],g['max_n'])
                if key not in ds_cache:
                    seqs={c['case_id']: c['output'] for c in sub}; index={}
                    for cid,seq in seqs.items(): build_index(seq, g['min_n'], g['max_n'], index, 0, cid)
                    ds_cache[key]={'seqs':seqs,'index':index}
                ds=ds_cache[key]
            agg = collections.Counter(); ratios=[]
            for c in sub:
                r = simulate(c['prompt'], c['output'], g['k'], g['min_n'], g['max_n'], g['order'], ds, c['case_id'], g.get('mode','ctx_then_ds'))
                for kk,v in r.items(): agg[kk]+=v
                ratios.append(r['tokens']/r['steps'])
            per_stage[s] = {'tokens': agg['tokens'], 'steps': agg['steps'], 'ideal_speedup_pooled': agg['tokens']/agg['steps'],
                            'ideal_speedup_mean_per_request': statistics.mean(ratios),
                            'acceptance_rate': agg['accepted']/agg['proposed'] if agg['proposed'] else None,
                            'accepted_per_step': agg['accepted']/agg['steps'], 'proposed_per_step': agg['proposed']/agg['steps'],
                            'draft_step_fraction': agg['steps_with_draft']/agg['steps'],
                            'ctx_steps': agg['ctx_steps'], 'ds_steps': agg['ds_steps'], 'ctx_acc': agg['ctx_acc'], 'ds_acc': agg['ds_acc'],
                            'n': len(sub)}
        results['policies'].append({'policy': g, 'stages': per_stage})
        print(json.dumps({'policy': g, **{s: round(v['ideal_speedup_pooled'],3) for s,v in per_stage.items()},
                          'acc': {s: (round(v['acceptance_rate'],3) if v['acceptance_rate'] else None) for s,v in per_stage.items()}}), flush=True)
    args.output.write_text(json.dumps(results, indent=1)+'\n')

if __name__ == '__main__': main()
