"""CPU proposer compatibility regression; NOT a model-serving speed benchmark."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time


def reference(tokens, low, high, max_len, k):
    k=min(k,max_len-len(tokens))
    if k<=0 or len(tokens)<low:return []
    for n in range(min(high,len(tokens)),low-1,-1):
        suffix=tokens[-n:]
        for start in range(len(tokens)-n):
            if tokens[start:start+n]==suffix:return tokens[start+n:start+n+k]
    return []


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result={'scope':'CPU proposer import/JIT/algorithm only; no LLM weights, no GPU speed or output-fidelity claim',
        'versions':{p:importlib.metadata.version(p) for p in ['vllm','numpy','numba','llvmlite']}}
    try:
        import numpy as np
        from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens as propose
    except Exception as e:
        result.update(import_ok=False,error_type=type(e).__name__,error=str(e)[:300])
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return
    # Character code sequences exercise a token-ID algorithm without pretending
    # to be the Gemma tokenizer or the LLM acceptance verifier.
    cases=json.loads(args.cases.read_text());rows=[]
    start=time.perf_counter()
    propose(np.array([1,2,3,1,2],dtype=np.int32),2,4,100,5)
    result['jit_warmup_s']=time.perf_counter()-start
    start=time.perf_counter()
    for c in cases:
        tokens=[ord(ch) for ch in c['question']]
        for k in [1,3,5,8]:
            for low,high in [(1,2),(2,4),(2,8)]:
                for scenario in ['natural','repeated_suffix','full_context']:
                    t=tokens+tokens[:6] if scenario=='repeated_suffix' else tokens
                    max_len=len(t) if scenario=='full_context' else len(t)+32
                    actual=propose(np.asarray(t,dtype=np.int32),low,high,max_len,k).tolist()
                    expected=reference(t,low,high,max_len,k)
                    rows.append({'case_id':c['case_id'],'k':k,'low':low,'high':high,'scenario':scenario,
                                 'exact_reference':actual==expected,'proposed_tokens':len(actual)})
    result.update(import_ok=True,question_count=len(cases),checks=len(rows),
        exact=sum(r['exact_reference'] for r in rows),cpu_wall_s=time.perf_counter()-start,rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
