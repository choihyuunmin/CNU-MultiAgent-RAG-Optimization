"""Paired analysis of the 2026-09-13/14 single-pass concurrency sweep (original vs signed capability)."""
import json, glob, os, math, statistics, sys
from datetime import datetime
import numpy as np
RNG=np.random.RandomState(20260917); B=10000
def valid(r):
    resp=r.get("response"); return r.get("status")=="ok" and isinstance(resp,dict) and isinstance(resp.get("laws"),list) and isinstance(resp.get("comment"),str)
def uniq(ids):
    out=[];seen=set()
    for v in ids:
        v=str(v or "").strip()
        if v and v not in seen: out.append(v);seen.add(v)
    return out
def sign_p(pos,neg):
    n,k=pos+neg,min(pos,neg); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else None
def ci(x):
    x=np.asarray(x,float); idx=RNG.randint(0,len(x),size=(B,len(x))); v=x[idx].mean(axis=1); return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
cells={}
order={}
for root,run in [("scale-data/scale/cnu-scale-main","main"),("scale-data/scale/cnu-scale-revised","revised"),("scale-data/high/cnu-high-main","high")]:
    st=json.load(open(f"{root}/status.json"))
    for i,c in enumerate(st["completed"]):
        order[(c["concurrency"],c["mode"])]=(run,i)
    for cell in sorted(glob.glob(f"{root}/c*")):
        C=int(os.path.basename(cell)[1:])
        for m in ["original","capability"]:
            p=f"{cell}/responses/capability-trial/{m}/client_requests.jsonl"
            if not os.path.exists(p): continue
            recs=[json.loads(l) for l in open(p) if l.strip()]
            if len(recs)!=200: continue
            cells[C,m]={r["question_id"]:r for r in recs}
levels=sorted({C for C,_ in cells if (C,"original") in cells and (C,"capability") in cells})
out=[]
print(f"{'C':>4} {'first':11s} {'pairs':>5} {'orig':>8} {'cap':>8} {'saving':>7} {'%':>6} {'95% CI':>18} {'median':>7} {'pos/neg':>8} {'sign p':>7} {'SD':>6} | {'rps o/c':>13} {'p95 o/c':>13} {'<=60s o/c':>11} {'recall':>7} {'set':>6}")
for C in levels:
    A,Bc=cells[C,"original"],cells[C,"capability"]
    qs=[q for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]
    d=np.array([(A[q]["duration_ms"]-Bc[q]["duration_ms"])/1000 for q in qs])
    mo=np.mean([A[q]["duration_ms"]/1000 for q in qs]); mc=np.mean([Bc[q]["duration_ms"]/1000 for q in qs])
    pos,neg=int((d>0).sum()),int((d<0).sum()); lo,hi=ci(d)
    def tp(cell):
        recs=list(cell.values()); ok=[r for r in recs if valid(r)]
        t0=min(datetime.fromisoformat(r["started_at"]).timestamp() for r in recs); t1=max(datetime.fromisoformat(r["completed_at"]).timestamp() for r in recs)
        lat=sorted(r["duration_ms"]/1000 for r in recs)
        return len(ok)/(t1-t0), lat[int(0.95*(len(lat)-1))], sum(l<=60 for l in lat)/len(lat), t1-t0, len(recs)-len(ok)
    to,tc=tp(A),tp(Bc)
    rec=[]; same=[]
    for q in qs:
        g=set(uniq(A[q]["law_ids"])); p=set(uniq(Bc[q]["law_ids"]))
        if g: rec.append(len(g&p)/len(g))
        same.append(g==p)
    first="original" if order[(C,"original")][1]<order[(C,"capability")][1] else "capability"
    row={"C":C,"run":order[(C,"original")][0],"first":first,"pairs":len(qs),"mean_original":mo,"mean_capability":mc,"saving":float(d.mean()),"pct":float(d.mean()/mo*100),"ci":[lo,hi],"median":float(np.median(d)),"pos":pos,"neg":neg,"sign_p":sign_p(pos,neg),"sd":float(d.std(ddof=1)),
         "rps":[to[0],tc[0]],"p95":[to[1],tc[1]],"slo60":[to[2],tc[2]],"wall":[to[3],tc[3]],"failures":[to[4],tc[4]],"recall_nonempty":float(np.mean(rec)),"n_nonempty":len(rec),"exact_set":float(np.mean(same))}
    out.append(row)
    print(f"{C:4d} {first:11s} {len(qs):5d} {mo:8.2f} {mc:8.2f} {d.mean():7.2f} {d.mean()/mo*100:6.1f} [{lo:7.2f},{hi:7.2f}] {np.median(d):7.2f} {pos:3d}/{neg:<4d} {sign_p(pos,neg):7.3f} {d.std(ddof=1):6.1f} | {to[0]:6.3f}/{tc[0]:6.3f} {to[1]:6.1f}/{tc[1]:6.1f} {to[2]*100:4.0f}%/{tc[2]*100:4.0f}% {np.mean(rec)*100:6.1f} {np.mean(same)*100:5.1f}")
# engine counters where available
KEYS={"requests":"vllm:e2e_request_latency_seconds_count","queue_s":"vllm:request_queue_time_seconds_sum","waiting":"vllm:num_requests_waiting","running":"vllm:num_requests_running","kv":"vllm:kv_cache_usage_perc","preempt":"vllm:num_preemptions_total","e2e_s":"vllm:e2e_request_latency_seconds_sum"}
print("\nengine counters per cell (receiver-2 = tool/answer worker, receiver-1 = orchestrator):")
eng=[]
for f in sorted(glob.glob("scale-data/*/*/c*/*-metrics.jsonl")):
    C=int(os.path.basename(os.path.dirname(f))[1:]); m=os.path.basename(f).split("-")[0]
    first={};last={};mx={}
    for line in open(f):
        d=json.loads(line); r=d["resource"]; met=d.get("metrics")
        if not met: continue
        first.setdefault(r,d); last[r]=d
        for k in ("waiting","running","kv"):
            mx[r,k]=max(mx.get((r,k),0),met.get(KEYS[k],0))
    row={"C":C,"arm":m,"file":f}
    for r in ("receiver-1","receiver-2"):
        if r in first:
            a,b=first[r]["metrics"],last[r]["metrics"]
            row[r]={"requests":b.get(KEYS["requests"],0)-a.get(KEYS["requests"],0),"queue_s":b.get(KEYS["queue_s"],0)-a.get(KEYS["queue_s"],0),"e2e_s":b.get(KEYS["e2e_s"],0)-a.get(KEYS["e2e_s"],0),"preempt":b.get(KEYS["preempt"],0)-a.get(KEYS["preempt"],0),"max_waiting":mx[r,"waiting"],"max_running":mx[r,"running"],"max_kv":mx[r,"kv"],"span_s":last[r]["time"]-first[r]["time"]}
    eng.append(row)
    print(f"C={C:3d} {m:11s} " + " | ".join(f"{r[-1]}: req {row[r]['requests']:5.0f} queue {row[r]['queue_s']:8.1f}s e2e {row[r]['e2e_s']:8.0f}s maxwait {row[r]['max_waiting']:4.0f} maxrun {row[r]['max_running']:3.0f} maxKV {row[r]['max_kv']:.2f} preempt {row[r]['preempt']:.0f}" for r in ("receiver-1","receiver-2") if r in row))
json.dump({"levels":out,"engine":eng},open("analysis/sweep_existing.json","w"),indent=1)
