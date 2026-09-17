"""Extra sweep analyses: per-level pooled CI (question means over rounds), self-repeat agreement per arm and level,
and per-level engine/boundary means."""
import json, sys, math, statistics
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); RNG=np.random.RandomState(20260918); B=10000
def valid(r):
    resp=r.get("response"); return r.get("status")=="ok" and isinstance(resp,dict) and isinstance(resp.get("laws"),list) and isinstance(resp.get("comment"),str)
def uniq(ids):
    out=[];seen=set()
    for v in ids:
        v=str(v or "").strip()
        if v and v not in seen: out.append(v);seen.add(v)
    return out
cells={}
for p in root.rglob("client_requests.jsonl"):
    parts=p.parts; L=int([x for x in parts if x.startswith("level-")][0][6:]); r=int([x for x in parts if x.startswith("round-")][0][6:]); arm=p.parent.name
    cells[L,r,arm]={x["question_id"]:x for x in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}
levels=sorted({k[0] for k in cells}); out={}
def same(a,b): return set(uniq(a["law_ids"]))==set(uniq(b["law_ids"]))
def rec(a,b):
    g=set(uniq(a["law_ids"])); p=set(uniq(b["law_ids"])); return None if not g else len(g&p)/len(g)
print(f"{'L':>4} {'pooled saving':>14} {'95% CI':>18} {'%':>6} {'q faster/slower':>16} {'sign p':>7} | self-repeat exact-set orig/cap | orig-vs-cap exact-set r1/r2 | self-repeat recall orig/cap | cross recall r1/r2")
for L in levels:
    rounds=sorted({r for (l,r,a) in cells if l==L})
    qs=[q for q in cells[L,rounds[0],"original"] if all(valid(cells[L,r,a][q]) for r in rounds for a in ("original","capability"))]
    D=np.array([[ (cells[L,r,"original"][q]["duration_ms"]-cells[L,r,"capability"][q]["duration_ms"])/1000 for r in rounds] for q in qs])  # q x rounds
    dq=D.mean(axis=1); idx=RNG.randint(0,len(qs),size=(B,len(qs))); boots=dq[idx].mean(axis=1)
    ref=np.mean([cells[L,r,"original"][q]["duration_ms"]/1000 for r in rounds for q in qs])
    pos=int((dq>0).sum()); neg=int((dq<0).sum()); n=pos+neg; k=min(pos,neg); sp=min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
    so=np.mean([same(cells[L,rounds[0],"original"][q],cells[L,rounds[1],"original"][q]) for q in qs]) if len(rounds)>1 else float('nan')
    sc=np.mean([same(cells[L,rounds[0],"capability"][q],cells[L,rounds[1],"capability"][q]) for q in qs]) if len(rounds)>1 else float('nan')
    cross=[np.mean([same(cells[L,r,"original"][q],cells[L,r,"capability"][q]) for q in qs]) for r in rounds]
    ro=[x for x in (rec(cells[L,rounds[0],"original"][q],cells[L,rounds[1],"original"][q]) for q in qs) if x is not None] if len(rounds)>1 else []
    rc=[x for x in (rec(cells[L,rounds[0],"capability"][q],cells[L,rounds[1],"capability"][q]) for q in qs) if x is not None] if len(rounds)>1 else []
    crr=[np.mean([x for x in (rec(cells[L,r,"original"][q],cells[L,r,"capability"][q]) for q in qs) if x is not None]) for r in rounds]
    out[L]={"n":len(qs),"pooled_saving":float(dq.mean()),"ci":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"pct":float(dq.mean()/ref*100),"pos":pos,"neg":neg,"sign_p":sp,
            "self_repeat_exact_original":float(so),"self_repeat_exact_capability":float(sc),"cross_exact":[float(x) for x in cross],
            "self_repeat_recall_original":float(np.mean(ro)) if ro else None,"self_repeat_recall_capability":float(np.mean(rc)) if rc else None,"cross_recall":[float(x) for x in crr]}
    print(f"{L:4d} {dq.mean():14.2f} [{np.percentile(boots,2.5):7.2f},{np.percentile(boots,97.5):7.2f}] {dq.mean()/ref*100:6.1f} {pos:7d}/{neg:<8d} {sp:7.4f} | {so*100:5.1f}/{sc*100:5.1f} | {cross[0]*100:5.1f}/{cross[1]*100:5.1f} | {np.mean(ro)*100 if ro else float('nan'):5.1f}/{np.mean(rc)*100 if rc else float('nan'):5.1f} | {crr[0]*100:5.1f}/{crr[1]*100:5.1f}")
json.dump(out,open(sys.argv[2],"w"),indent=1)
