import numpy as np
from common import METHODS,SEEDS,SNRS,CLASS_NAMES,CONDITIONS

def summarize(rows):
    columns=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES],'mean_vibration_scale','mean_current_scale')
    perseed=[]; summary=[]; snrsummary=[]
    for method in METHODS:
        for endpoint in ('clean','noisy'):
            values=[]
            for seed in SEEDS:
                chosen=[r for r in rows if r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                if len(chosen)!=(4 if endpoint=='clean' else 80):
                    raise ValueError('Unbalanced endpoint aggregation')
                r=dict(method=method,seed=seed,endpoint=endpoint,**{k:float(np.mean([v[k] for v in chosen])) for k in columns})
                perseed.append(r); values.append(r)
            summary.append(dict(method=method,endpoint=endpoint,**{key:value for k in columns for key,value in
                ((k+'_mean',float(np.mean([v[k] for v in values]))),(k+'_SD',float(np.std([v[k] for v in values],ddof=1))))}))
        for snr in (None,*SNRS):
            vals=[{k:float(np.mean([r[k] for r in rows if r['method']==method and r['seed']==s and r['snr_db']==snr])) for k in columns} for s in SEEDS]
            snrsummary.append(dict(method=method,snr_db=snr,**{key:value for k in columns for key,value in
                ((k+'_mean',float(np.mean([v[k] for v in vals]))),(k+'_SD',float(np.std([v[k] for v in vals],ddof=1))))}))
    paired=[]; pairseeds=[]
    for endpoint in ('clean','noisy'):
        for comparator in [m for m in METHODS if m!='physical']:
            diffs={k:[] for k in ('BA','Macro_F1')}
            for seed in SEEDS:
                a=next(r for r in perseed if r['method']=='physical' and r['endpoint']==endpoint and r['seed']==seed)
                b=next(r for r in perseed if r['method']==comparator and r['endpoint']==endpoint and r['seed']==seed)
                delta={k:a[k]-b[k] for k in diffs}
                pairseeds.append(dict(comparison='physical_minus_'+comparator,endpoint=endpoint,seed=seed,**delta))
                for k in diffs:
                    diffs[k].append(delta[k])
            paired.append(dict(comparison='physical_minus_'+comparator,endpoint=endpoint,**{key:value for k,d in diffs.items() for key,value in
                ((k+'_difference_mean',float(np.mean(d))),(k+'_difference_SD',float(np.std(d,ddof=1))),(k+'_positive_seeds',int((np.array(d)>0).sum())))}))
    return perseed,summary,snrsummary,paired,pairseeds


def condition_summaries(rows):
    columns=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES])
    seeds=[]; summary=[]
    for condition in CONDITIONS:
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                values=[]
                for seed in SEEDS:
                    chosen=[r for r in rows if r['condition']==condition and r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                    assert len(chosen)==(1 if endpoint=='clean' else 20)
                    v=dict(condition=condition,method=method,seed=seed,endpoint=endpoint,**{k:float(np.mean([r[k] for r in chosen])) for k in columns})
                    seeds.append(v); values.append(v)
                summary.append(dict(condition=condition,method=method,endpoint=endpoint,**{key:value for k in columns for key,value in
                    ((k+'_mean',float(np.mean([v[k] for v in values]))),(k+'_SD',float(np.std([v[k] for v in values],ddof=1))))}))
    return seeds,summary


