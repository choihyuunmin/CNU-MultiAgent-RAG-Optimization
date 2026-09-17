import json, itertools, numpy as np
from pathlib import Path
ROOT=Path("review-data"); METHODS=["original","direct","capability"]; LABEL={"original":"Original","direct":"Unsigned","capability":"Signed"}
RNG=np.random.RandomState(20260916); B=10000
def valid(r):
    resp=r.get("response"); return r.get("status")=="ok" and isinstance(resp,dict) and isinstance(resp.get("laws"),list) and isinstance(resp.get("comment"),str)
def uniq(ids):
    out=[];seen=set()
    for v in ids:
        v=str(v or "").strip()
        if v and v not in seen: out.append(v);seen.add(v)
    return out
cells={}
for r in (1,2,3):
    for m in METHODS:
        rows=[json.loads(l) for l in (ROOT/f"main/round-{r}/responses/review-trial/{m}/client_requests.jsonl").read_text().splitlines() if l.strip()]
        cells[r,m]={x["question_id"]:x for x in rows}
qtype={q:x["category"] for q,x in cells[1,"original"].items()}
common=[q for q in cells[1,"original"] if all(valid(cells[r,m][q]) for r in (1,2,3) for m in METHODS)]
def same(a,b): return set(uniq(a.get("law_ids",[])))==set(uniq(b.get("law_ids",[])))
def rec(a,b):
    g=set(uniq(a.get("law_ids",[]))); p=set(uniq(b.get("law_ids",[])))
    return None if not g else len(g&p)/len(g)
idx=RNG.randint(0,len(common),size=(B,len(common)))
print("n common", len(common))
out={}
# self-repeat per method across rounds
for m in METHODS:
    S=np.array([[same(cells[a,m][q],cells[b,m][q]) for (a,b) in [(1,2),(1,3),(2,3)]] for q in common],float)
    R=np.array([[rec(cells[a,m][q],cells[b,m][q]) for (a,b) in [(1,2),(1,3),(2,3)]] for q in common],float)
    mask=~np.isnan(R).any(axis=1)
    sb=S[idx].mean(axis=(1,2)); 
    print(f"{LABEL[m]:9s} self-repeat exact-set {S.mean()*100:.1f}% [{np.percentile(sb,2.5)*100:.1f},{np.percentile(sb,97.5)*100:.1f}] per pair {[round(x*100,1) for x in S.mean(axis=0)]} | nonempty recall {np.nanmean(R[mask])*100:.2f}% (n={mask.sum()})")
    out[LABEL[m]]={"exact_set":float(S.mean()),"per_pair":[float(x) for x in S.mean(axis=0)],"ci":(float(np.percentile(sb,2.5)),float(np.percentile(sb,97.5))),"recall":float(np.nanmean(R[mask]))}
# cross-method within round, all pairs
for a,b in [("original","direct"),("original","capability"),("direct","capability")]:
    S=np.array([[same(cells[r,a][q],cells[r,b][q]) for r in (1,2,3)] for q in common],float)
    print(f"{LABEL[a]} vs {LABEL[b]} within round exact-set {S.mean()*100:.1f}% per round {[round(x*100,1) for x in S.mean(axis=0)]}")
# per-type exact-set for candidate-vs-original and original self-repeat
types=sorted(set(qtype.values()))
for t in types:
    qs=[q for q in common if qtype[q]==t]
    so=np.mean([[same(cells[a,"original"][q],cells[b,"original"][q]) for (a,b) in [(1,2),(1,3),(2,3)]] for q in qs])
    ss=np.mean([[same(cells[r,"original"][q],cells[r,"capability"][q]) for r in (1,2,3)] for q in qs])
    su=np.mean([[same(cells[r,"original"][q],cells[r,"direct"][q]) for r in (1,2,3)] for q in qs])
    sc=np.mean([[same(cells[a,"capability"][q],cells[b,"capability"][q]) for (a,b) in [(1,2),(1,3),(2,3)]] for q in qs])
    print(f"type {t:26s} n={len(qs):3d} orig-repeat {so*100:5.1f}  signed-vs-orig {ss*100:5.1f}  unsigned-vs-orig {su*100:5.1f}  signed-repeat {sc*100:5.1f}")
# superset/subset pattern for signed vs original
add=miss=both=0
for r in (1,2,3):
    for q in common:
        g=set(uniq(cells[r,"original"][q]["law_ids"])); p=set(uniq(cells[r,"capability"][q]["law_ids"]))
        if g!=p:
            if p>g: add+=1
            elif p<g: miss+=1
            else: both+=1
print("signed vs original unequal sets: candidate superset", add, "subset", miss, "mixed", both)
# response length (comment chars) and law count by method
for m in METHODS:
    lc=[len(uniq(cells[r,m][q]["law_ids"])) for r in (1,2,3) for q in common]; cc=[cells[r,m][q]["comment_chars"] for r in (1,2,3) for q in common]
    print(f"{LABEL[m]:9s} mean law count {np.mean(lc):.3f} mean comment chars {np.mean(cc):.0f}")
json.dump(out,open("analysis/self_repeat.json","w"),indent=1)
