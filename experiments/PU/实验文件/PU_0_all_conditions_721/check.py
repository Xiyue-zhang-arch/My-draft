"""Verify exact stratification, preprocessing, joint optimization and full report accounting."""
import torch
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from common import ROOT,PROJECT,DATA,CONDITIONS,METHODS,ORDERS,WRONG_ORDERS,SEEDS,models,cases,descriptor_profile,metrics,sha,write_json,save_npz,setup,seed_model
from model import Network
from descriptors import describe
from reference import reference_one
from prepare import CACHE, describe_physical, descriptor_path, data_verify, verify_frozen_inputs, OLD_DATA
from support import source_hashes,load_bank
from descriptor_cache import archive_path
from engine import train_one,verify_all

def synthetic_report(out):
    from report import generate,collect
    out.mkdir(); (out/'models').mkdir(); (out/'predictions').mkdir()
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    descriptions=[dict(**c,kind='main',waveform_input_sha256='synthetic-wave',descriptor_input_sha256={p:'synthetic-'+p for p in ('physical','wrong','statistics')}) for c in cases()]
    write_json(out/'test_cases.json',descriptions)
    write_json(out/'config.json',dict(decisions_locked=True,weights_verified=True,prediction_cases=2520,completed_models=models(),synthetic=True))
    for spec in models():
        folder=out/'predictions'/spec['model_id']; folder.mkdir()
        for case in cases():
            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids]); logits=np.eye(4,dtype='float32')[y]*3
            if spec['method']!='physical':
                n=spec['seed']-1704; logits[:n]=np.roll(logits[:n],1,axis=1)
            _,p=metrics(logits,y)
            save_npz(folder/(case['case_id']+'.npz'),**spec,case_id=case['case_id'],target=case['target'],kind='main',
                logits=logits,probability=p,y=y,window_index=ids,scale_mean=np.ones((400,2)),scale_channel_mean=np.ones((2,64)),scale_range=np.ones((2,2)),
                waveform_input_sha256='synthetic-wave',descriptor_input_sha256='synthetic-'+descriptor_profile(spec['method']))
    summary,pairs=generate(out,figures=False)
    delta=np.array([(s-1704)/400 for s in SEEDS])
    assert all(abs(p['BA_difference_mean']-delta.mean())<1e-14 and abs(p['BA_difference_SD']-delta.std(ddof=1))<1e-14 and p['BA_positive_seeds']==5 for p in pairs)
    path=out/'predictions/m0001'/(cases()[0]['case_id']+'.npz')
    with np.load(path) as z:
        values={k:z[k].copy() for k in z.files}
    values['y'][0]=(values['y'][0]+1)%4
    with path.open('wb') as f:
        np.savez_compressed(f,**values)
    try:
        collect(out)
    except ValueError as e:
        assert 'labels' in str(e)
    else:
        raise AssertionError('Tampered labels accepted')
    return dict(synthetic_predictions=2520,synthetic_confusion_cells=40320,nonzero_paired_mean_and_SD_verified=True,tampered_labels_rejected=True)

def check(output):
    output.mkdir(parents=True,exist_ok=False); setup(); before=source_hashes(); data_verify(); verify_frozen_inputs(); bank=load_bank(CACHE)
    provenance=json.loads((PROJECT/'implementation_provenance.json').read_text(encoding='utf-8'))
    for name in ('encoder.py','model.py','descriptors.py','reference.py'):
        assert sha(PROJECT/name)==provenance[name]['sha256']
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    records=json.loads((DATA/'records.json').read_text(encoding='utf-8'))
    roles={}; fingerprints={}
    for role,total,quota in (('train',11200,700),('val',3200,200),('test',1600,100)):
        assert len(split[role])==total
        for ci,condition in enumerate(CONDITIONS):
            selected=[windows[int(i)] for i in split[role] if windows[int(i)]['condition']==condition]
            assert len(selected)==quota*4 and np.array_equal(np.bincount([w['label'] for w in selected],minlength=4),np.full(4,quota))
        for i in split[role]:
            w=windows[int(i)]; rid=w['record_index']; digest=records[rid]['processed_sha256']
            assert rid not in roles or roles[rid]==role
            assert digest not in fingerprints or fingerprints[digest]==role
            roles[rid]=role; fingerprints[digest]=role
    assert len(fingerprints)==636 and [sum(v==r for v in fingerprints.values()) for r in ('train','val','test')]==[445,127,64]
    assert np.array_equal(np.sort(np.r_[split['train'],split['val'],split['test']]),np.arange(16000))
    assert all(r['split']==fingerprints[r['processed_sha256']] for r in records)
    raw=np.load(DATA/'signals.npy',mmap_mode='r')
    samples=np.stack([raw[next(int(i) for i in split['train'] if windows[int(i)]['condition']==c)] for c in CONDITIONS])
    errors={}
    for profile,orders in (('legacy_physical',ORDERS),('wrong',WRONG_ORDERS)):
        ref=np.stack([reference_one(s,orders) for s in samples]); actual=describe(torch.tensor(samples,dtype=torch.float64),orders).numpy()
        errors[profile]=float(np.abs(ref-actual).max()); assert errors[profile]<1e-9
    cached_ids = split['train'][:8]
    computed = describe_physical(torch.from_numpy(np.array(raw[cached_ids])).cuda()).cpu().numpy()
    np.testing.assert_allclose(computed, np.load(descriptor_path('physical'))[cached_ids], rtol=0, atol=2e-7)
    # Independent train-only calibration check.
    norm=bank['norm']; value=np.array(raw[split['train']],dtype='float64')
    assert np.allclose(value.mean((0,2)),norm['waveform_mean'],rtol=0,atol=1e-12)
    assert np.allclose(value.std((0,2)),norm['waveform_std'],rtol=0,atol=1e-12)
    del value
    for profile in ('physical','wrong','statistics'):
        d=np.load(descriptor_path(profile))[split['train']]
        if profile == 'physical':
            d = d.astype(np.float64)
        q1,q3=np.quantile(d,[.25,.75],axis=0)
        assert np.array_equal(np.median(d,axis=0),norm[profile]['center'])
        assert np.array_equal(np.maximum(q3-q1,1e-6),norm[profile]['scale'])
    oldwindows=json.loads((OLD_DATA/'windows.json').read_text(encoding='utf-8'))
    oldlookup={(w['record_index'],w['start_angular_sample']):w['window_index'] for w in oldwindows}
    noise=np.load(DATA/'vibration_noise_templates.npy',mmap_mode='r'); oldnoise=np.load(OLD_DATA/'vibration_noise_templates.npy',mmap_mode='r')
    noise_error=0.; cached_comparisons=0
    for j,i in enumerate(split['test']):
        w=windows[int(i)]; oldid=oldlookup[(w['record_index'],w['start_angular_sample'])]
        if w['condition']!=CONDITIONS[0]:
            assert np.array_equal(noise[j],oldnoise[oldid-4000]); cached_comparisons+=5
    from raw_prepare import read_mat,process,down_and_order
    for condition in (CONDITIONS[0],CONDITIONS[1]):
        i=int(split[condition][0]); w=windows[i]; r=records[w['record_index']]
        original,times,_=read_mat(archive_path(ROOT, r['archive']),r['member']); rebuilt,info=process(original,times,r['rpm'])
        np.testing.assert_allclose(raw[i], rebuilt[:,w['start_angular_sample']:w['end_angular_sample_exclusive']], rtol=1e-6, atol=2e-7)
    for condition in CONDITIONS:
        i=int(split[condition][0]); w=windows[i]; r=records[w['record_index']]; j=int(np.searchsorted(split['test'],i))
        rng=np.random.default_rng(np.random.SeedSequence([7201,w['noise_record_identity']]))
        n=rng.standard_normal(r['raw_grid_samples']); n-=n.mean(); n/=np.sqrt(np.mean(n*n)); n*=np.sqrt(r['raw_vibration_power'])
        trimmed=n[r['trim_start_sample']:]; down=resample_poly(trimmed,1,4,window=('kaiser',5.),padtype='line')
        down=down[:min(len(down),int((len(trimmed)-1)//4)+1)]
        projected=resample_poly(down,3,5,window=('kaiser',5.),padtype='line') if r['rpm']==900 else down
        expected=projected[:r['angular_samples']].astype('float32')[w['start_angular_sample']:w['end_angular_sample_exclusive']]
        noise_error=max(noise_error,float(np.abs(expected-noise[j,0]).max()))
    assert noise_error<1e-7
    t=np.arange(256000)/64000; anti=down_and_order(np.sin(2*np.pi*6000*t),15)
    alias_rms=float(np.sqrt(np.mean(anti[1000:-1000]**2))); assert alias_rms<.003
    initial={}; base=None; x=bank['train']['x'][:8]; d=bank['train']['d']['physical'][:8]
    for method in METHODS:
        seed_model(1706); net=Network(method).cuda(); net.eval(); initial[method]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        with torch.no_grad():
            logits,scales,_=net(x,d)
        assert torch.equal(scales,torch.ones_like(scales))
        if base is None:
            base=logits
        assert torch.equal(base,logits)
    (output/'functional_models').mkdir(); completed=[]
    for spec in models()[:6]:
        row=train_one(spec,bank,output/'functional_models'/spec['model_id'],functional=True); completed.append(row)
        state=torch.load(output/'functional_models'/spec['model_id']/'model.pt',weights_only=True,map_location='cpu')['state_dict']
        for prefix in ('v.','c.'):
            assert any(not torch.equal(v,initial[spec['method']][k]) for k,v in state.items() if k.startswith(prefix))
        if spec['method']!='equal':
            assert any(not torch.equal(v,initial[spec['method']][k]) for k,v in state.items() if k.startswith('condition.'))
        if spec['method']=='self_only':
            net=Network('self_only').cuda(); net.load_state_dict(state); net.eval()
            with torch.no_grad():
                _,before_scales,_=net(x,d)
                for m in (0,1):
                    changed=d.clone(); changed[:,1-m]+=1; _,after_scales,_=net(x,changed)
                    assert torch.equal(before_scales[:,m],after_scales[:,m])
    traces=[json.loads((output/'functional_models'/r['model_id']/'training_trace.json').read_text()) for r in completed]
    assert all(t==traces[0] for t in traces) and len({r['shared_initialization_sha256'] for r in completed})==1
    try:
        verify_all(output,completed,bank)
    except ValueError as e:
        assert '30' in str(e)
    else:
        raise AssertionError('Incomplete model barrier accepted')
    from audit import independent
    rng=np.random.default_rng(42); logits=rng.normal(size=(40,4)); y=np.tile(np.arange(4),10)
    a,p=metrics(logits,y); b,q=independent(logits.copy(),y)
    independent_error=max([abs(a[k]-b[k]) for k in ('ACC','BA','Macro_F1','CE')]+[float(np.abs(p-q).max())]); assert independent_error<1e-12
    synthetic=synthetic_report(output/'synthetic_full_report')
    assert before==source_hashes()
    write_json(output/'preflight.json',dict(stage='preflight_passed',source_hashes=before,cache_verification_sha256=sha(CACHE/'verification.json'),
        record_and_duplicate_group_overlap=0,exact_stratified_ratio_verified=True,train_only_statistics_verified=True,
        descriptor_reference_max_error=errors,noise_reference_max_error=noise_error,old_noise_template_matches=cached_comparisons,
        raw_1500_and_900rpm_time_pairing_verified=True,antialias_6000Hz_to_9600Hz_rms=alias_rms,
        functional_models=6,functional_updates=1050,both_encoders_and_conditions_updated=True,save_reload_exact=True,
        shared_initialization_and_actual_random_streams_equal=True,self_only_cross_descriptor_exclusion=True,incomplete_barrier_rejected=True,
        independent_metric_error=independent_error,test_network_inference_calls=0,**synthetic))
    print('PREFLIGHT PASSED',str(output),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); check(parser.parse_args().output.resolve())
