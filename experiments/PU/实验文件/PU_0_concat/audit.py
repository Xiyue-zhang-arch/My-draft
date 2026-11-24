"""Independent CPU audit of new and frozen-reference saved predictions."""
import torch
import argparse,csv
from pathlib import Path
import numpy as np
from common import *
from prepare import data_verify, source_path, verify_frozen_inputs
from descriptor_cache import archive_path
from report import entries

def independent(a,y):
    a=np.asarray(a,dtype='float64');a-=a.max(1,keepdims=True);e=np.exp(a);p=e/e.sum(1,keepdims=True);pred=a.argmax(1)
    cm=np.zeros((4,4),dtype='int64');np.add.at(cm,(y,pred),1)
    recall=np.array([np.mean(pred[y==i]==i) for i in range(4)])
    f1=np.array([2*np.count_nonzero((y==i)&(pred==i))/max(1,np.count_nonzero(y==i)+np.count_nonzero(pred==i)) for i in range(4)])
    return dict(ACC=float(np.mean(y==pred)),BA=float(recall.mean()),Macro_F1=float(f1.mean()),CE=float(np.mean(np.log(e.sum(1))-a[np.arange(len(y)),y])),recall=recall,confusion=cm),p
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def audit(out,output):
    if output.exists():raise FileExistsError(output)
    state=read(out/'config.json')
    if state['stage'] not in ('auditing','complete') or len(state['completed_models'])!=30 or state['prediction_cases']!=1890:raise ValueError('Incomplete result')
    cache=Path(state['cache']);cp=data_verify(cache);verify_frozen_inputs()
    for n,h in state['source_hashes'].items():
        if sha(source_path(n))!=h or sha(out/'source'/n)!=h:raise ValueError('Source changed')
    for n,h in cp['files_sha256'].items():
        if sha(cache/n)!=h:raise ValueError('Cache changed')
    raw_archives=read(DATA/'archive_hashes.json')
    for r in raw_archives.values():
        if sha(archive_path(PROJECT.parents[3], r['path']))!=r['sha256']:raise ValueError('Raw archive changed')
    index=np.load(FROZEN_SPLIT/'split_indices.npz');windows=read(DATA/'windows.json');train=index['train'];val=index['val'];norm=read(cache/'normalization.json')
    if {windows[int(i)]['record_index'] for i in train}&{windows[int(i)]['record_index'] for i in val}:raise ValueError('Source record overlap')
    x=np.array(np.load(DATA/'signals.npy',mmap_mode='r')[train],dtype='float64')
    if not np.allclose(x.mean((0,2)),norm['waveform_mean'],rtol=0,atol=1e-12) or not np.allclose(x.std((0,2)),norm['waveform_std'],rtol=0,atol=1e-12):raise ValueError('Wrong source calibration')
    d=np.load(cache/'physical.npy')[train].astype(np.float64);q1,q3=np.quantile(d,[.25,.75],axis=0)
    if not np.array_equal(np.median(d,axis=0),norm['physical']['center']) or not np.array_equal(np.maximum(q3-q1,1e-6),norm['physical']['scale']):raise ValueError('Wrong descriptor calibration')
    rp=read(out/'report/verification.json')
    for n,h in rp['report_files_sha256'].items():
        if sha(out/'report'/n)!=h:raise ValueError('Report changed')
    rows=csvread(out/'report/metrics_recomputed.csv');lookup={(r['model_id'],r['case_id']):r for r in rows}
    if len(rows)!=1890 or len(lookup)!=1890:raise ValueError('Wrong metric count')
    error=prob_error=val_error=0.;traces={};inits={}
    for r in state['completed_models']:
        folder=out/'models'/r['model_id']
        if r['functional'] or r['epochs']!=50 or r['updates']!=2500:raise ValueError('Wrong training budget')
        for n,h in r['artifacts_sha256'].items():
            if sha(folder/n)!=h:raise ValueError('Checkpoint/artifact changed')
        t=read(folder/'training_trace.json');key=(r['seed'],r['head_width'])
        if key in traces and t!=traces[key]:raise ValueError('Paired RNG stream mismatch')
        traces[key]=t
        for role,h in r['shared_initialization_sha256'].items():
            key=(r['seed'],role)
            if key in inits and inits[key]!=h:raise ValueError('Shared role initialization mismatch')
            inits[key]=h
        with np.load(folder/'validation.npz') as z:
            y=np.array([windows[int(i)]['label'] for i in val]);m,p=independent(z['logits'].copy(),y)
            if not np.array_equal(z['y'],y) or not np.array_equal(z['window_index'],val):raise ValueError('Validation identity mismatch')
            val_error=max(val_error,abs(m['CE']-r['val_CE']));prob_error=max(prob_error,float(np.max(np.abs(p-z['probability']))))
    cms=[];values={};clean={};invariance=0;count=0
    for spec,case,path in entries(out):
        expected=rp['new_prediction_sha256'][str(path.relative_to(out))]
        if sha(path)!=expected:raise ValueError('Saved prediction changed')
        with np.load(path) as z:
            ids=index[case['target']];y=np.array([windows[int(i)]['label'] for i in ids]);m,p=independent(z['logits'].copy(),y)
            if not np.array_equal(z['y'],y) or not np.array_equal(z['window_index'],ids):raise ValueError('Target identity mismatch')
            r=lookup[(spec['model_id'],case['case_id'])];prob_error=max(prob_error,float(np.max(np.abs(p-z['probability']))))
            for k in ('ACC','BA','Macro_F1','CE'):error=max(error,abs(m[k]-float(r[k])))
            for i,c in enumerate(CLASS_NAMES):error=max(error,abs(m['recall'][i]-float(r['recall_'+c])))
            endpoint='clean' if case['snr_db'] is None else 'noisy';values.setdefault((spec['method'],spec['seed'],endpoint),[]).append(m['BA']);cms.append(m['confusion']);count+=1
            if spec['method'] in ('current_plain','current_physical'):
                key=(spec['model_id'],case['target'])
                if case['snr_db'] is None:clean[key]=z['logits'].copy()
                else:
                    if not np.array_equal(clean[key],z['logits']):raise ValueError('Unused modality invariance failed')
                    invariance+=1
    with np.load(out/'report/confusions.npz') as z:
        if not np.array_equal(z['confusion'],np.stack(cms)):raise ValueError('Confusion mismatch')
    summaries=[]
    for r in csvread(out/'report/endpoints_summary.csv'):
        a=np.array([np.mean(values[(r['method'],s,r['endpoint'])]) for s in SEEDS]);error=max(error,abs(a.mean()-float(r['BA_mean'])),abs(a.std(ddof=1)-float(r['BA_SD'])))
        summaries.append(dict(method=r['method'],endpoint=r['endpoint'],BA_mean=float(a.mean()),BA_SD=float(a.std(ddof=1))))
    for r in csvread(out/'report/paired_differences.csv'):
        d=np.array([np.mean(values[(r['method'],s,r['endpoint'])])-np.mean(values[(r['comparator'],s,r['endpoint'])]) for s in SEEDS])
        error=max(error,abs(d.mean()-float(r['BA_difference_mean'])),abs(d.std(ddof=1)-float(r['BA_difference_SD'])))
        if int((d>1e-12).sum())!=int(r['positive_seeds']):raise ValueError('Paired seed direction mismatch')
    if max(error,prob_error,val_error)>1e-12 or invariance!=600 or count!=1890:raise ValueError('Independent numerical audit failed')
    result=dict(stage='independent_audit_passed',new_models=30,referenced_models=0,new_predictions=1890,referenced_predictions=0,combined_confusion_cells=30240,
                metric_max_error=error,probability_max_error=prob_error,source_validation_CE_max_error=val_error,
                current_only_unused_channel_invariance=600,descriptor_version=cp['descriptor_version'],raw_archives_unchanged=len(raw_archives),
                new_network_inference_calls=0,summaries=summaries,auditor_sha256=sha(__file__))
    write(output,result);print('AUDIT PASSED',error,prob_error,flush=True);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run.resolve(),a.output.resolve())
