"""Replay recorded visible deltas without inference, measuring exact text.

Uses the deployed emitter's real incoming chunk boundaries. Comparing complete
answers as single chunks would exaggerate pacing cost and is deliberately avoided.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from moleg_paper_runtime import bootstrap


async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    bootstrap()
    from core.query_loop._helpers import emit_streaming_delta
    records=[];seen=set()
    measured={r['session_id']:r['case_id'] for r in
        (json.loads(line) for line in (args.trace.parent/'e2e_stream.jsonl').read_text().splitlines())
        if r['variant']=='baseline' and r['repeat']==0}
    for line in args.trace.read_text().splitlines():
        r=json.loads(line)
        if r.get('session_id') not in measured:continue
        q=measured[r['session_id']]
        if q in seen or not r.get('stream_input'):continue
        seen.add(q);records.append((q,[x['delta'] for x in r['stream_input']],r['session_id']))
    sem=asyncio.Semaphore(4)
    rows=[]
    async def one(q,deltas,session):
        async with sem:
            expected=''.join(deltas)
            row={'case_id':q,'source_session_id':session,'input_chunks':len(deltas),'chars':len(expected)}
            for variant in ['paced','immediate']:
                queue=asyncio.Queue()
                start=time.perf_counter()
                for delta in deltas:
                    await emit_streaming_delta(queue,delta,delay_s=.02 if variant=='paced' else 0,
                                               chunk_chars=5 if variant=='paced' else 1000)
                elapsed=time.perf_counter()-start
                output=[]
                while not queue.empty():output.append(queue.get_nowait()['delta'])
                row[variant]={'elapsed_s':elapsed,'output_chunks':len(output),
                    'exact_text':''.join(output)==expected,
                    'sha256':hashlib.sha256(''.join(output).encode()).hexdigest()}
            rows.append(row)
    await asyncio.gather(*(one(q,d,s) for q,d,s in records))
    args.output.write_text(json.dumps({'scope':'measured application emitter only, no GPU inference or client render',
        'n':len(rows),'rows':rows},ensure_ascii=False,indent=2)+'\n')
    print('replay',len(rows),'exact',sum(r['paced']['exact_text'] and r['immediate']['exact_text'] for r in rows))


if __name__=='__main__':asyncio.run(main())
