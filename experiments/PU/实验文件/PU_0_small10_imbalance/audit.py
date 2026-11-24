"""Independent CPU audit of all protocols, saved predictions and data boundaries."""
import torch
import csv
import json
from pathlib import Path
import numpy as np
from common import PROJECT,DATA,PROTOCOLS,TRAIN_COUNTS,UPDATES_PER_EPOCH,CLASS_NAMES,METHODS,SEEDS,CONDITIONS,models,cases,sha,write_json
from prepare import CACHE, data_verify, descriptor_path, source_path, verify_frozen_inputs
from descriptor_cache import archive_path

def independent(logits,y):
    a=np.array(logits,dtype='float64',copy=True); a-=a.max(1,keepdims=True)
    ex=np.exp(a); p=ex/ex.sum(1,keepdims=True); pred=a.argmax(1)
    cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
    recall=np.array([(pred[y==k]==k).mean() for k in range(4)])
    f1=[2*np.count_nonzero((pred==k)&(y==k))/max(1,np.count_nonzero(pred==k)+np.count_nonzero(y==k)) for k in range(4)]
    return dict(ACC=float((pred==y).mean()),BA=float(recall.mean()),Macro_F1=float(np.mean(f1)),
        CE=float(np.mean(np.log(ex.sum(1))-a[np.arange(len(y)),y])),recall=recall,confusion=cm),p

def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def verify_data(cache=CACHE):
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    labels=np.array([w['label'] for w in windows]); raw=np.load(DATA/'signals.npy',mmap_mode='r')
    record_sets=[{windows[int(i)]['record_index'] for i in split[r]} for r in ('train','val','test')]
    assert all(not record_sets[a]&record_sets[b] for a,b in ((0,1),(0,2),(1,2)))
    assert (len(split['train']),len(split['val']),len(split['test']))==(1120,320,160)
    expected={'balanced10':[280]*4,'imbalanced':[280,28,28,28],'matched_balanced':[91]*4}
    for protocol in PROTOCOLS:
        ids=split['train_'+protocol]; assert len(ids)==TRAIN_COUNTS[protocol]
        assert set(ids).issubset(set(split['train'])) and np.array_equal(np.bincount(labels[ids],minlength=4),expected[protocol])
        norm=json.loads((cache/('normalization_'+protocol+'.json')).read_text(encoding='utf-8'))
        values=np.array(raw[ids],dtype='float64')
        assert np.allclose(values.mean((0,2)),norm['waveform_mean'],rtol=0,atol=1e-12)
        assert np.allclose(values.std((0,2)),norm['waveform_std'],rtol=0,atol=1e-12)
        for p in ('physical','wrong','statistics'):
            d=np.load(descriptor_path(p, cache))[ids]
            if p == 'physical':
                d = d.astype(np.float64)
            q1,q3=np.quantile(d,[.25,.75],axis=0)
            assert np.array_equal(np.median(d,axis=0),norm[p]['center'])
            assert np.array_equal(np.maximum(q3-q1,1e-6),norm[p]['scale'])
    for condition in CONDITIONS:
        assert np.array_equal(np.bincount(labels[split[condition]],minlength=4),[10]*4)
        valid=[int(i) for i in split['val'] if windows[int(i)]['condition']==condition]
        assert np.array_equal(np.bincount(labels[valid],minlength=4),[20]*4)
    return split,windows

def audit(out):
    state=json.loads((out/'config.json').read_text(encoding='utf-8'))
    assert state['stage'] in ('auditing','complete') and state['prediction_cases']==7560 and len(state['completed_models'])==90
    assert state['weights_verified'] and state['decisions_locked']
    for name,value in state['source_hashes'].items():
        assert sha(source_path(name))==value==sha(out/'source'/name)
    cache = Path(state['cache'])
    prep = data_verify(cache)
    verify_frozen_inputs()
    split,windows=verify_data(cache); proof=json.loads((out/'report/verification.json').read_text(encoding='utf-8'))
    for name,value in proof['report_files_sha256'].items():
        assert sha(out/'report'/name)==value
    reportrows=readcsv(out/'report/metrics_recomputed.csv'); lookup={(r['model_id'],r['case_id']):r for r in reportrows}
    assert len(reportrows)==len(lookup)==7560
    numerical=[]; matrices=[]; error=perror=verror=0.; traces={}; inits={}; degeneracy=[]
    for spec,row in zip(models(),state['completed_models']):
        protocol=spec['protocol']; folder=out/'models'/spec['model_id']
        assert row['epochs']==50 and row['updates']==UPDATES_PER_EPOCH[protocol]*50 and not row['functional']
        for name,value in row['artifacts_sha256'].items():
            assert sha(folder/name)==value
        trace=json.loads((folder/'training_trace.json').read_text(encoding='utf-8')); key=(protocol,spec['seed'])
        assert key not in traces or traces[key]==trace
        assert spec['seed'] not in inits or inits[spec['seed']]==row['shared_initialization_sha256']
        traces[key]=trace; inits[spec['seed']]=row['shared_initialization_sha256']
        history=readcsv(folder/'history.csv'); assert len(history)==50
        zero=[int(r['epoch']) for r in history if float(r['gradient_v'])==0 or float(r['gradient_c'])==0]
        with np.load(folder/'validation.npz') as z:
            assert np.array_equal(z['window_index'],split['val']) and np.array_equal(z['y'],[windows[int(i)]['label'] for i in split['val']])
            m,p=independent(z['logits'],z['y']); verror=max(verror,abs(m['CE']-row['val_CE']))
            constant=len(np.unique(z['logits'].argmax(1)))==1
        if zero or constant:
            degeneracy.append(dict(**spec,zero_gradient_epochs=zero,validation_constant_prediction=constant))
        for case in cases():
            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids])
            path=out/'predictions'/spec['model_id']/(case['case_id']+'.npz')
            assert sha(path)==proof['prediction_sha256'][str(path.relative_to(out))]
            with np.load(path) as z:
                assert np.array_equal(z['window_index'],ids) and np.array_equal(z['y'],y)
                m,p=independent(z['logits'],y); perror=max(perror,float(np.abs(p-z['probability']).max()))
                r=lookup[(spec['model_id'],case['case_id'])]
                for k in ('ACC','BA','Macro_F1','CE'):
                    error=max(error,abs(m[k]-float(r[k])))
                for j,c in enumerate(CLASS_NAMES):
                    error=max(error,abs(m['recall'][j]-float(r['recall_'+c])))
                matrices.append(m['confusion']); numerical.append(dict(**spec,condition=case['target'],endpoint='clean' if case['noise_seed'] is None else 'noisy',
                    **{k:m[k] for k in ('BA','Macro_F1')}))
    with np.load(out/'report/confusions.npz') as z:
        assert np.array_equal(z['confusion'],np.stack(matrices))
    values={}
    summary=readcsv(out/'report/endpoints_summary.csv'); seeds=readcsv(out/'report/per_seed_endpoints.csv')
    assert len(summary)==36 and len(seeds)==180
    for protocol in PROTOCOLS:
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                r=next(r for r in summary if r['protocol']==protocol and r['method']==method and r['endpoint']==endpoint)
                for k in ('BA','Macro_F1'):
                    vv=[]
                    for seed in SEEDS:
                        v=[x[k] for x in numerical if x['protocol']==protocol and x['method']==method and x['endpoint']==endpoint and x['seed']==seed]
                        assert len(v)==(4 if endpoint=='clean' else 80)
                        value=float(np.mean(v)); vv.append(value)
                        rs=next(r for r in seeds if r['protocol']==protocol and r['method']==method and r['endpoint']==endpoint and int(r['seed'])==seed)
                        error=max(error,abs(value-float(rs[k])))
                    values[(protocol,method,endpoint,k)]=np.array(vv)
                    error=max(error,abs(np.mean(vv)-float(r[k+'_mean'])),abs(np.std(vv,ddof=1)-float(r[k+'_SD'])))
    for r in readcsv(out/'report/paired_differences.csv'):
        comparator=r['comparison'].removeprefix('physical_minus_')
        for k in ('BA','Macro_F1'):
            d=values[(r['protocol'],'physical',r['endpoint'],k)]-values[(r['protocol'],comparator,r['endpoint'],k)]
            error=max(error,abs(d.mean()-float(r[k+'_difference_mean'])),abs(d.std(ddof=1)-float(r[k+'_difference_SD'])))
            assert int((d>0).sum())==int(r[k+'_positive_seeds'])
    for r in readcsv(out/'report/protocol_effects.csv'):
        a,b=r['comparison'].split('_minus_')
        for k in ('BA','Macro_F1'):
            d=values[(a,r['method'],r['endpoint'],k)]-values[(b,r['method'],r['endpoint'],k)]
            error=max(error,abs(d.mean()-float(r[k+'_difference_mean'])),abs(d.std(ddof=1)-float(r[k+'_difference_SD'])))
            assert int((d>0).sum())==int(r[k+'_positive_seeds'])
    condition_summary=readcsv(out/'report/condition_endpoints_summary.csv')
    assert len(condition_summary)==144
    for r in condition_summary:
        for k in ('BA','Macro_F1'):
            vv=[]
            for seed in SEEDS:
                picked=[v[k] for v in numerical if v['protocol']==r['protocol'] and v['method']==r['method'] and v['condition']==r['condition'] and v['endpoint']==r['endpoint'] and v['seed']==seed]
                assert len(picked)==(1 if r['endpoint']=='clean' else 20)
                vv.append(float(np.mean(picked)))
            error=max(error,abs(np.mean(vv)-float(r[k+'_mean'])),abs(np.std(vv,ddof=1)-float(r[k+'_SD'])))
    for seed in SEEDS:
        a,b=traces[('imbalanced',seed)],traces[('matched_balanced',seed)]
        assert all(x['ordinal_permutation_sha256']==y['ordinal_permutation_sha256'] and x['CUDA_rng_start_sha256']==y['CUDA_rng_start_sha256'] for x,y in zip(a,b))
    archives=json.loads((DATA/'archive_hashes.json').read_text(encoding='utf-8'))
    for r in archives.values():
        assert sha(archive_path(PROJECT.parents[3], r['path']))==r['sha256']
    assert max(error,perror,verror)<1e-12
    write_json(out/'degeneracy_summary.json',degeneracy)
    result=dict(stage='independent_audit_passed',metric_max_error=error,probability_max_error=perror,validation_CE_max_error=verror,
        models=90,prediction_cases=7560,confusion_cells=120960,descriptor_version=prep['descriptor_version'],original_archives_verified=len(archives),
        train_only_calibration_verified=True,common_validation_test_verified=True,matched_training_random_streams_verified=True,
        degenerate_model_count=len(degeneracy),no_additional_network_inference=True)
    write_json(out/'independent_audit.json',result); print('AUDIT',json.dumps(result),flush=True)
    return result
