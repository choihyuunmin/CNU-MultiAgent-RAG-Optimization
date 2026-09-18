"""Item 1 and 3: orchestrator call tokens/time by stage position and load; input length vs wait/latency; engine prefill vs decode."""
import json, re, sys, statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
def ts(s):
    m=re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)", s); return datetime.fromisoformat(m.group(1)+m.group(3)).timestamp()+float("0."+m.group(2))
def cts(s): return datetime.fromisoformat(s).timestamp()
def valid(r):
    resp=r.get("response"); return r.get("status")=="ok" and isinstance(resp,dict)
# datasets: (label, level, client_requests path list for original, app log path, metrics path)
sets=[]
for r in (1,2,3):
    sets.append((4, f"review-data/main/round-{r}/responses/review-trial/original/client_requests.jsonl", "review-data/original-application.log", f"review-data/main/round-{r}/original-metrics.jsonl"))
for lvl in (8,16,32,64,100):
    for r in (1,2):
        p=f"sweep-data/merged/main/level-{lvl:03d}/round-{r}/responses/sweep-trial/original/client_requests.jsonl"
        if Path(p).exists(): sets.append((lvl,p,"sweep-data/merged/original-application.log",f"sweep-data/merged/main/level-{lvl:03d}/round-{r}/original-metrics.jsonl"))
logs={}
def load_log(p):
    if p in logs: return logs[p]
    ev=[]
    for line in open(p,errors="replace"):
        i=line.find(' {"event": "request_finished"')
        if i<0: continue
        try: e=json.loads(line[i+1:]); e["_t"]=ts(line[:35]); ev.append(e)
        except: pass
    logs[p]=ev; return ev
KEYS={"prefill":"vllm:request_prefill_time_seconds_sum","decode":"vllm:request_decode_time_seconds_sum","queue":"vllm:request_queue_time_seconds_sum","e2e":"vllm:e2e_request_latency_seconds_sum","n":"vllm:e2e_request_latency_seconds_count","prompt":"vllm:prompt_tokens_total","gen":"vllm:generation_tokens_total","preempt":"vllm:num_preemptions_total"}
out=defaultdict(list)
by_level_pos=defaultdict(lambda: defaultdict(list))
per_req=defaultdict(list)
engine=defaultdict(list)
for lvl,cp,lp,mp in sets:
    recs={x["question_id"]:x for x in (json.loads(l) for l in open(cp) if l.strip())}
    lo=min(cts(x["started_at"]) for x in recs.values())-2; hi=max(cts(x["completed_at"]) for x in recs.values())+2
    ev=[e for e in load_log(lp) if lo<=e["_t"]<=hi]
    for e in ev:
        q=re.sub(r"-r\d+-c\d+$","",e["case_id"]); rec=recs.get(q)
        if not rec or not valid(rec): continue
        orch=[c for c in e["calls"] if c["model"]=="orchestrator" and c.get("status")=="ok"]
        tot_prompt=sum(c.get("prompt_tokens") or 0 for c in orch); tot_api=sum(c.get("api_ms") or 0 for c in orch)
        per_req[lvl].append((tot_prompt, rec["duration_ms"]/1000, tot_api/1000, len(orch), max((c.get("prompt_tokens") or 0) for c in orch) if orch else 0))
        for k,c in enumerate(orch):
            by_level_pos[lvl][k+1].append((c.get("prompt_tokens") or 0, c.get("completion_tokens") or 0, (c.get("api_ms") or 0)/1000, (c.get("wait_ms") or 0)/1000))
    # engine counters (orchestrator receiver-1) for this batch
    first={};last={}
    for line in open(mp):
        d=json.loads(line)
        if not d.get("metrics"): continue
        first.setdefault(d["resource"],d); last[d["resource"]]=d
    if "receiver-1" in first:
        a,b=first["receiver-1"]["metrics"],last["receiver-1"]["metrics"]
        engine[lvl].append({k:b.get(v,0)-a.get(v,0) for k,v in KEYS.items()})
print("=== 1. orchestrator calls by stage position (original arm): level, pos, n, mean prompt, mean completion, mean api s, p95 api s, mean wait s")
for lvl in sorted(by_level_pos):
    for pos in sorted(by_level_pos[lvl]):
        v=by_level_pos[lvl][pos]
        if len(v)<20: continue
        api=[x[2] for x in v]
        print(f"C={lvl:3d} pos{pos} n={len(v):4d} prompt {st.mean(x[0] for x in v):6.0f} compl {st.mean(x[1] for x in v):5.0f} api {st.mean(api):6.2f}s p95 {sorted(api)[int(0.95*(len(api)-1))]:6.2f}s wait {st.mean(x[3] for x in v):5.2f}s")
print("\n=== 1b. engine (orchestrator) per batch: prefill vs decode vs queue seconds per request, tokens per request")
for lvl in sorted(engine):
    for e in engine[lvl]:
        n=e["n"] or 1
        print(f"C={lvl:3d} requests {e['n']:.0f} prefill/req {e['prefill']/n:6.2f}s decode/req {e['decode']/n:6.2f}s queue/req {e['queue']/n:7.2f}s e2e/req {e['e2e']/n:7.2f}s prompt tok/req {e['prompt']/n:6.0f} gen tok/req {e['gen']/n:5.0f} preempt {e['preempt']:.0f}")
print("\n=== 3. input length vs latency (original arm): per level, Spearman-like rank correlation and quartile comparison")
def rank(a):
    a=np.asarray(a,float); r=np.empty(len(a)); r[np.argsort(a)]=np.arange(len(a)); return r
for lvl in sorted(per_req):
    v=per_req[lvl]; P=np.array([x[0] for x in v],float); T=np.array([x[1] for x in v]); A=np.array([x[2] for x in v]); M=np.array([x[4] for x in v],float)
    rho=np.corrcoef(rank(P),rank(T))[0,1]; rho_a=np.corrcoef(rank(P),rank(A))[0,1]
    q1,q3=np.percentile(P,25),np.percentile(P,75)
    lo=T[P<=q1]; hi=T[P>=q3]
    # per-call: prompt tokens vs api time for the largest call (selection-like)
    calls=[x for pos in by_level_pos[lvl] for x in by_level_pos[lvl][pos]]
    cp=np.array([c[0] for c in calls],float); ca=np.array([c[2] for c in calls]); rho_c=np.corrcoef(rank(cp),rank(ca))[0,1]
    print(f"C={lvl:3d} n={len(v):4d} total orch prompt tok/req mean {P.mean():6.0f} (Q1 {q1:.0f}, Q3 {q3:.0f}) | rank corr(prompt,resp time)={rho:5.2f} corr(prompt,orch api sum)={rho_a:5.2f} per-call corr(prompt,api)={rho_c:5.2f} | resp time low-input Q {lo.mean():6.1f}s vs high-input Q {hi.mean():6.1f}s (calls/req {st.mean(x[3] for x in v):.2f})")
json.dump({"by_level_pos":{str(l):{str(p):v for p,v in d.items()} for l,d in by_level_pos.items()},"engine":{str(l):v for l,v in engine.items()},"per_req":{str(l):v for l,v in per_req.items()}},open("analysis/orch_input_load.json","w"))
