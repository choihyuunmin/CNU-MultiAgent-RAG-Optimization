"""Query-clustered statistics; no expert-accuracy claims from silver labels."""
from __future__ import annotations
from collections import defaultdict
import math
import random
import statistics


def percentile(values, p):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*p
    low=math.floor(index);high=math.ceil(index)
    return values[low]+(values[high]-values[low])*(index-low)


def timing(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'mean_s':statistics.fmean(values) if values else None,
            'max_s':max(values) if values else None,
            'over_60_s_count':sum(v>60 for v in values),'over_120_s_count':sum(v>120 for v in values),
            **{f'p{p}_s':percentile(values,p/100) for p in [50,95,99]}}


def paired_cluster_ci(pairs, *, seed=20260905, resamples=4000):
    """Each (baseline,candidate) is one question's mean over repetitions.

    Returns a ratio of means, not a mean of per-question speedup ratios.
    This clusters the repeated observations at their independent query unit.
    """
    if not pairs:return {'n':0}
    rng=random.Random(seed);n=len(pairs)
    baseline=statistics.fmean(x[0] for x in pairs)
    candidate=statistics.fmean(x[1] for x in pairs)
    reductions=[];deltas=[]
    for _ in range(resamples):
        sampled=rng.choices(pairs,k=n)
        b=sum(x[0] for x in sampled);c=sum(x[1] for x in sampled)
        if b: reductions.append(1-c/b)
        deltas.append((c-b)/n)
    return {'n':n,'baseline_mean':baseline,'candidate_mean':candidate,
            'speedup':baseline/candidate if candidate else None,
            'reduction_pct':100*(1-candidate/baseline) if baseline else None,
            'reduction_95ci_pct':[100*percentile(reductions,p) for p in [.025,.975]] if reductions else None,
            'delta':candidate-baseline,'delta_95ci':[percentile(deltas,p) for p in [.025,.975]],
            'bootstrap_resamples':resamples,'unit':'question (repetitions averaged)'}


def paired_difference_ci(pairs, **kwargs):
    result=paired_cluster_ci(pairs,**kwargs)
    return {k:v for k,v in result.items() if k not in ['speedup','reduction_pct','reduction_95ci_pct']}


def mcnemar_exact(pairs):
    wins=sum(bool(not b and c) for b,c in pairs)
    losses=sum(bool(b and not c) for b,c in pairs)
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'wins':wins,'losses':losses,'p_two_sided':p}


def holm_adjust(p_values):
    """Family-wise Holm correction, preserving labels and monotonicity."""
    result={};previous=0.;count=len(p_values)
    for i,(key,p) in enumerate(sorted(p_values.items(),key=lambda x:x[1])):
        previous=max(previous,min(1.,(count-i)*p))
        result[key]=previous
    return result


def rank_of(ids, expected, *, parent=False):
    for i,item in enumerate(ids):
        if (str(item).split('_')[0] if parent else str(item))==expected:
            return i+1
    return None


def known_item_metrics(ids, case):
    if not case.get('source_id'):return None
    r=rank_of(ids,case['source_id'])
    law=rank_of(ids,case['source_law_id'],parent=True)
    result = {'chunk_hit1':float(r==1),'chunk_hit5':float(r is not None and r<=5),
        'chunk_hit20':float(r is not None and r<=20),
        'chunk_mrr20':1/r if r is not None and r<=20 else 0.,
        'chunk_ndcg20':1/math.log2(r+1) if r is not None and r<=20 else 0.,
        'law_hit1':float(law==1),'law_hit5':float(law is not None and law<=5),
        'law_hit20':float(law is not None and law<=20),
        'law_mrr20':1/law if law is not None and law<=20 else 0.}
    # Named-law queries do not identify a unique random paragraph. Score exact
    # chunks only when a validated answer span grounded question generation.
    if not case.get('reference_answer'):
        for key in result:
            if key.startswith('chunk_'): result[key] = None
    return result


def id_recall(reference, candidate):
    # No evidence in reference is unscorable, never a fabricated perfect score.
    ref=set(reference)
    return len(ref.intersection(candidate))/len(ref) if ref else None


def means(rows):
    if not rows:return {}
    return {k:statistics.fmean(values) if (values := [r[k] for r in rows if r[k] is not None]) else None
            for k in rows[0]}
