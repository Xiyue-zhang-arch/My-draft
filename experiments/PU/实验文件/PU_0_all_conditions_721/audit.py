"""Independent CPU recomputation and read-only protection audit."""
import torch
import csv
import json
import argparse
from pathlib import Path
import numpy as np
from common import PROJECT,DATA,CLASS_NAMES,METHODS,SEEDS,CONDITIONS,models,cases,sha,write_json
from prepare import data_verify, descriptor_path, source_path, verify_frozen_inputs
from descriptor_cache import archive_path

def independent(logits,y):
    a=np.asarray(logits,dtype='float64'); a-=a.max(1,keepdims=True)
    ex=np.exp(a); p=ex/ex.sum(1,keepdims=True); pred=a.argmax(1)
    cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
    recall=np.array([(pred[y==k]==k).mean() for k in range(4)])
    f1=[2*np.count_nonzero((pred==k)&(y==k))/max(1,np.count_nonzero(pred==k)+np.count_nonzero(y==k)) for k in range(4)]
    return dict(ACC=float((pred==y).mean()),BA=float(recall.mean()),Macro_F1=float(np.mean(f1)),
        CE=float(np.mean(np.log(ex.sum(1))-a[np.arange(len(y)),y])),recall=recall,confusion=cm),p

def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def audit(out):
    state=json.loads((out/'config.json').read_text(encoding='utf-8'))
    assert state['stage'] in ('auditing','complete') and state['prediction_cases']==2520
    assert len(state['completed_models'])==30 and state['weights_verified'] and state['decisions_locked']
    for name,value in state['source_hashes'].items():
        assert sha(source_path(name))==value==sha(out/'source'/name)
    cache = Path(state['cache'])
    prep = data_verify(cache)
    verify_frozen_inputs()
    windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    split=np.load(DATA/'split_indices.npz'); train,val,test=[split[k] for k in ('train','val','test')]
    assert (len(train),len(val),len(test))==(11200,3200,1600)
    sets=[{windows[int(i)]['record_index'] for i in ids} for ids in (train,val,test)]
    assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
    norm=json.loads((cache/'normalization.json').read_text(encoding='utf-8'))
    x=np.array(np.load(DATA/'signals.npy',mmap_mode='r')[train],dtype='float64')
    assert np.allclose(x.mean((0,2)),norm['waveform_mean'],rtol=0,atol=1e-12)
    assert np.allclose(x.std((0,2)),norm['waveform_std'],rtol=0,atol=1e-12)
    for profile in ('physical','wrong','statistics'):
        d=np.load(descriptor_path(profile, cache))[train]
        if profile == 'physical':
            d = d.astype(np.float64)
        q1,q3=np.quantile(d,[.25,.75],axis=0)
        assert np.array_equal(np.median(d,axis=0),norm[profile]['center'])
        assert np.array_equal(np.maximum(q3-q1,1e-6),norm[profile]['scale'])
    proof=json.loads((out/'report/verification.json').read_text(encoding='utf-8'))
    for name,value in proof['report_files_sha256'].items():
        assert sha(out/'report'/name)==value
    rows=csvread(out/'report/metrics_recomputed.csv'); lookup={(r['model_id'],r['case_id']):r for r in rows}
    assert len(rows)==len(lookup)==2520
    matrices=[]; numerical=[]; error=perror=verror=0.
    traces={}; inits={}
    for spec,row in zip(models(),state['completed_models']):
        folder=out/'models'/spec['model_id']
        assert row['epochs']==50 and row['updates']==8750 and not row['functional']
        for name,value in row['artifacts_sha256'].items():
            assert sha(folder/name)==value
        trace=json.loads((folder/'training_trace.json').read_text(encoding='utf-8')); seed=spec['seed']
        assert seed not in traces or trace==traces[seed]
        assert seed not in inits or row['shared_initialization_sha256']==inits[seed]
        traces[seed]=trace; inits[seed]=row['shared_initialization_sha256']
        with np.load(folder/'validation.npz') as z:
            assert np.array_equal(z['window_index'],val)
            m,p=independent(z['logits'],z['y']); verror=max(verror,abs(m['CE']-row['val_CE']))
        for case in cases():
            condition_ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in condition_ids])
            path=out/'predictions'/spec['model_id']/(case['case_id']+'.npz')
            assert sha(path)==proof['prediction_sha256'][str(path.relative_to(out))]
            with np.load(path) as z:
                assert np.array_equal(z['window_index'],condition_ids) and np.array_equal(z['y'],y)
                m,p=independent(z['logits'],y); perror=max(perror,float(np.abs(p-z['probability']).max()))
                reported=lookup[(spec['model_id'],case['case_id'])]
                for k in ('ACC','BA','Macro_F1','CE'):
                    error=max(error,abs(m[k]-float(reported[k])))
                for k,c in enumerate(CLASS_NAMES):
                    error=max(error,abs(m['recall'][k]-float(reported['recall_'+c])))
                matrices.append(m['confusion']); numerical.append(dict(**spec,condition=case['target'],endpoint='clean' if case['noise_seed'] is None else 'noisy',**{k:m[k] for k in ('BA','Macro_F1')}))
    with np.load(out/'report/confusions.npz') as z:
        assert np.array_equal(z['confusion'],np.stack(matrices))
    summary=csvread(out/'report/endpoints_summary.csv'); paired=csvread(out/'report/paired_differences.csv'); values={}
    for method in METHODS:
        for endpoint in ('clean','noisy'):
            r=next(r for r in summary if r['method']==method and r['endpoint']==endpoint)
            for k in ('BA','Macro_F1'):
                v=np.array([np.mean([r[k] for r in numerical if r['method']==method and r['seed']==s and r['endpoint']==endpoint]) for s in SEEDS])
                values[(method,endpoint,k)]=v
                error=max(error,abs(v.mean()-float(r[k+'_mean'])),abs(v.std(ddof=1)-float(r[k+'_SD'])))
    for r in paired:
        comparator=r['comparison'].removeprefix('physical_minus_')
        for k in ('BA','Macro_F1'):
            v=values[('physical',r['endpoint'],k)]-values[(comparator,r['endpoint'],k)]
            error=max(error,abs(v.mean()-float(r[k+'_difference_mean'])),abs(v.std(ddof=1)-float(r[k+'_difference_SD'])))
            assert int((v>0).sum())==int(r[k+'_positive_seeds'])
    condition_report=csvread(out/'report/condition_endpoints_summary.csv')
    condition_seeds=csvread(out/'report/condition_per_seed_endpoints.csv')
    assert len(condition_report)==48 and len(condition_seeds)==240
    for condition in CONDITIONS:
        for role,expected in (('train',700),('val',200),('test',100)):
            chosen=[windows[int(i)] for i in split[role] if windows[int(i)]['condition']==condition]
            assert np.array_equal(np.bincount([w['label'] for w in chosen],minlength=4),np.full(4,expected))
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                reported=next(r for r in condition_report if r['condition']==condition and r['method']==method and r['endpoint']==endpoint)
                for k in ('BA','Macro_F1'):
                    values=[]
                    for seed in SEEDS:
                        chosen=[r[k] for r in numerical if r['condition']==condition and r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                        assert len(chosen)==(1 if endpoint=='clean' else 20)
                        value=float(np.mean(chosen)); values.append(value)
                        rs=next(r for r in condition_seeds if r['condition']==condition and r['method']==method and int(r['seed'])==seed and r['endpoint']==endpoint)
                        error=max(error,abs(value-float(rs[k])))
                    error=max(error,abs(np.mean(values)-float(reported[k+'_mean'])),abs(np.std(values,ddof=1)-float(reported[k+'_SD'])))
    archives=json.loads((DATA/'archive_hashes.json').read_text(encoding='utf-8'))
    for record in archives.values():
        assert sha(archive_path(PROJECT.parents[3], record['path']))==record['sha256']
    assert max(error,perror,verror)<1e-12
    result=dict(stage='independent_audit_passed',metric_max_error=error,probability_max_error=perror,
        validation_CE_max_error=verror,prediction_cases=2520,confusion_cells=40320,descriptor_version=prep['descriptor_version'],
        original_archives_verified=len(archives),record_overlap=0,train_only_calibration_verified=True,
        no_additional_network_inference=True)
    write_json(out/'independent_audit.json',result); print('AUDIT',json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.resolve())
