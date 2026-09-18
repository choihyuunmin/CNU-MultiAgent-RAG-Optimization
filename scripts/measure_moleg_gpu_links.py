"""Bounded GPU copy microbenchmark; no model or serving configuration changes.

Run only after model latency experiments. At most 128 MiB of tensors per GPU.
This measures communication, not tensor-parallel LLM performance.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=30)
    args=ap.parse_args()
    import torch
    if torch.cuda.device_count()<2:
        raise RuntimeError('two GPUs required')
    free={i:torch.cuda.mem_get_info(i)[0] for i in [0,1]}
    if any(value<1_500_000_000 for value in free.values()):
        raise RuntimeError('insufficient free memory for a non-disruptive bounded copy test')
    max_bytes=64*1024*1024
    src={i:torch.full((max_bytes,),17+i,dtype=torch.uint8,device=f'cuda:{i}') for i in [0,1]}
    dst={i:torch.empty_like(src[i]) for i in [0,1]}
    for i in [0,1]:torch.cuda.synchronize(i)
    results=[]
    for size in [1024,65536,1048576,33554432,67108864]:
        for source,target in [(0,0),(0,1),(1,0),(1,1)]:
            a=src[source][:size];b=dst[target][:size]
            values=[]
            for repeat in range(args.repeats+3):
                started=time.perf_counter()
                b.copy_(a,non_blocking=True)
                torch.cuda.synchronize(target)
                if source!=target:torch.cuda.synchronize(source)
                elapsed=time.perf_counter()-started
                if repeat>=3:values.append(elapsed)
            # Full buffer content check, outside the timed transfers.
            valid=bool(torch.all(b.cpu()==17+source).item())
            if not valid:raise RuntimeError('GPU copy verification failed')
            values.sort();mean=statistics.fmean(values)
            results.append({'source':source,'target':target,'bytes':size,'n':len(values),
                            'mean_s':mean,'p95_s':values[int(.95*(len(values)-1))],
                            'effective_GBps':size/mean/1e9,'exact':valid})
    out={'utc':datetime.now(timezone.utc).isoformat(),'scope':'isolated tensor copy; not LLM tensor parallelism',
         'torch_version':torch.__version__,'initial_free_bytes':free,
         'tensor_bytes_per_gpu':2*max_bytes,
         'peer_access_0_to_1':torch.cuda.can_device_access_peer(0,1),
         'peer_access_1_to_0':torch.cuda.can_device_access_peer(1,0),
         'topology':subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
         'results':results}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'tests':len(results),'all_exact':all(r['exact'] for r in results),
                       'large_transfers':[r for r in results if r['bytes']==max_bytes]}),flush=True)


if __name__=='__main__':main()
