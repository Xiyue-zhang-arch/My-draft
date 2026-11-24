"""Nested sampling, normalization, paired training and complete synthetic-report checks."""
import torch
import argparse
import json
from pathlib import Path
import numpy as np
from common import PROJECT,DATA,PROTOCOLS,SEEDS,METHODS,CONDITIONS,ORDERS,WRONG_ORDERS,models,cases,descriptor_profile,metrics,sha,write_json,save_npz,setup,seed_model
from model import Network
from descriptors import describe
from reference import reference_one
from prepare import CACHE, describe_physical, descriptor_path, data_verify, verify_frozen_inputs, PARENT,PARENT_DATA
from support import source_hashes,load_bank
from engine import train_one,verify_all
from audit import verify_data,independent

def synthetic_report(out):
    from report import generate,collect
    out.mkdir(); (out/'models').mkdir(); (out/'predictions').mkdir()
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    descriptions=[dict(**c,kind='main',waveform_input_sha256={p:'synthetic-'+p for p in PROTOCOLS},
        descriptor_input_sha256={p:{k:'synthetic-'+p+'-'+k for k in ('physical','wrong','statistics')} for p in PROTOCOLS}) for c in cases()]
    write_json(out/'test_cases.json',descriptions)
    write_json(out/'config.json',dict(decisions_locked=True,weights_verified=True,prediction_cases=7560,completed_models=models(),synthetic=True))
    for spec in models():
        protocol=spec['protocol']; folder=out/'predictions'/spec['model_id']; folder.mkdir()
        for case in cases():
            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids]); logits=np.eye(4,dtype='float32')[y]*3
            n=PROTOCOLS.index(protocol)+(0 if spec['method']=='physical' else spec['seed']-1704)
            if n:
                logits[:n]=np.roll(logits[:n],1,axis=1)
            _,p=metrics(logits,y)
            save_npz(folder/(case['case_id']+'.npz'),**spec,case_id=case['case_id'],target=case['target'],kind='main',
                logits=logits,probability=p,y=y,window_index=ids,scale_mean=np.ones((40,2)),scale_channel_mean=np.ones((2,64)),scale_range=np.ones((2,2)),
                waveform_input_sha256='synthetic-'+protocol,descriptor_input_sha256='synthetic-'+protocol+'-'+descriptor_profile(spec['method']))
    summary,pairs,effects=generate(out,figures=False)
    expected=np.array([(s-1704)/40 for s in SEEDS])
    assert all(abs(p['BA_difference_mean']-expected.mean())<1e-14 and abs(p['BA_difference_SD']-expected.std(ddof=1))<1e-14 and p['BA_positive_seeds']==5 for p in pairs)
    for r in effects:
        a,b=r['comparison'].split('_minus_'); delta=-(PROTOCOLS.index(a)-PROTOCOLS.index(b))/40
        assert abs(r['BA_difference_mean']-delta)<1e-14 and abs(r['BA_difference_SD'])<1e-14
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
    return dict(synthetic_predictions=7560,synthetic_confusion_cells=120960,within_protocol_nonzero_pairing_verified=True,
        protocol_effect_nonzero_mean_verified=True,tampered_labels_rejected=True)

def check(output):
    output.mkdir(parents=True,exist_ok=False); setup(); before=source_hashes(); data_verify(); verify_frozen_inputs(); split,windows=verify_data()
    for name in ('encoder.py','model.py','descriptors.py','reference.py','raw_prepare.py'):
        assert sha(PROJECT/name)==sha(PARENT/name)
    parent_indices=np.load(DATA/'parent_indices.npz'); retained=parent_indices['retained_parent_window_indices']
    parent_split=np.load(PARENT_DATA/'split_indices.npz'); parent_windows=json.loads((PARENT_DATA/'windows.json').read_text(encoding='utf-8'))
    raw=np.load(DATA/'signals.npy',mmap_mode='r'); parent_raw=np.load(PARENT_DATA/'signals.npy',mmap_mode='r')
    assert np.array_equal(raw,parent_raw[retained])
    for role,count in (('train',1120),('val',320),('test',160)):
        selected=retained[split[role]]
        assert len(selected)==count and set(selected).issubset(set(parent_split[role]))
        for ci,condition in enumerate(CONDITIONS):
            for bi,bearing in enumerate(('K003','K005','KA01','KA04','KI14','KI05','KB23','KB24')):
                actual=[int(i) for i in selected if parent_windows[int(i)]['condition']==condition and parent_windows[int(i)]['bearing']==bearing]
                assert len(actual)==({'train':35,'val':10,'test':5}[role])
    for protocol in PROTOCOLS:
        train=split['train_'+protocol]
        for ci,condition in enumerate(CONDITIONS):
            labels=[windows[int(i)]['label'] for i in train if windows[int(i)]['condition']==condition]
            expected=[70]*4 if protocol=='balanced10' else [70,7,7,7] if protocol=='imbalanced' else [22 if k==ci else 23 for k in range(4)]
            assert np.array_equal(np.bincount(labels,minlength=4),expected)
    pos=np.searchsorted(parent_split['test'],retained[split['test']])
    assert np.array_equal(np.load(DATA/'vibration_noise_templates.npy'),np.load(PARENT_DATA/'vibration_noise_templates.npy',mmap_mode='r')[pos])
    for p in ('physical','wrong','statistics'):
        assert np.array_equal(np.load(DATA/(p+'_all.npy')),np.load(PARENT_DATA/(p+'_all.npy'),mmap_mode='r')[retained])
    sample_ids=[next(int(i) for i in split['train'] if windows[int(i)]['condition']==c and windows[int(i)]['label']==label) for c in CONDITIONS for label in (0,1)]
    samples=np.array(raw[sample_ids]); errors={}
    for p,orders in (('legacy_physical',ORDERS),('wrong',WRONG_ORDERS)):
        ref=np.stack([reference_one(s,orders) for s in samples]); actual=describe(torch.tensor(samples,dtype=torch.float64),orders).numpy()
        errors[p]=float(np.abs(ref-actual).max()); assert errors[p]<1e-9
    cached_ids = split['train'][:8]
    computed = describe_physical(torch.from_numpy(np.array(raw[cached_ids])).cuda()).cpu().numpy()
    np.testing.assert_allclose(computed, np.load(descriptor_path('physical'))[cached_ids], rtol=0, atol=2e-7)
    banks={p:load_bank(CACHE,p) for p in PROTOCOLS}; initial={}; base=None
    x=banks['balanced10']['train']['x'][:8]; d=banks['balanced10']['train']['d']['physical'][:8]
    for method in METHODS:
        seed_model(1706); net=Network(method).cuda(); net.eval(); initial[method]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        with torch.no_grad():
            logits,scales,_=net(x,d)
        assert torch.equal(scales,torch.ones_like(scales))
        if base is None:
            base=logits
        assert torch.equal(base,logits)
    (output/'functional_models').mkdir(); completed=[]; traces={}
    for spec in [m for m in models() if m['seed']==1706]:
        protocol=spec['protocol']; folder=output/'functional_models'/spec['model_id']
        row=train_one(spec,banks[protocol],folder,functional=True); completed.append(row)
        state=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')['state_dict']
        for prefix in ('v.','c.'):
            assert any(not torch.equal(v,initial[spec['method']][k]) for k,v in state.items() if k.startswith(prefix))
        if spec['method']!='equal':
            assert any(not torch.equal(v,initial[spec['method']][k]) for k,v in state.items() if k.startswith('condition.'))
        trace=json.loads((folder/'training_trace.json').read_text()); assert protocol not in traces or trace==traces[protocol]
        traces[protocol]=trace
        if spec['method']=='self_only':
            net=Network('self_only').cuda(); net.load_state_dict(state); net.eval()
            with torch.no_grad():
                _,s,_=net(x,d)
                for m in (0,1):
                    other=d.clone(); other[:,1-m]+=1; _,ss,_=net(x,other)
                    assert torch.equal(s[:,m],ss[:,m])
    assert len({r['shared_initialization_sha256'] for r in completed})==1
    assert traces['imbalanced'][0]['ordinal_permutation_sha256']==traces['matched_balanced'][0]['ordinal_permutation_sha256']
    assert traces['imbalanced'][0]['CUDA_rng_start_sha256']==traces['matched_balanced'][0]['CUDA_rng_start_sha256']
    try:
        verify_all(output,completed,banks)
    except ValueError as e:
        assert '90' in str(e)
    else:
        raise AssertionError('Incomplete model barrier accepted')
    rng=np.random.default_rng(42); logits=rng.normal(size=(40,4)); y=np.tile(np.arange(4),10)
    a,p=metrics(logits,y); b,q=independent(logits,y)
    metric_error=max([abs(a[k]-b[k]) for k in ('ACC','BA','Macro_F1','CE')]+[float(np.abs(p-q).max())]); assert metric_error<1e-12
    synthetic=synthetic_report(output/'synthetic_full_report'); assert before==source_hashes()
    write_json(output/'preflight.json',dict(stage='preflight_passed',source_hashes=before,cache_verification_sha256=sha(CACHE/'verification.json'),
        exact_ten_percent_nested_subset_verified=True,record_role_inheritance_verified=True,training_class_counts_verified=True,
        common_validation_test_verified=True,train_only_protocol_normalization_verified=True,noise_exact_parent_subset=True,
        descriptor_reference_max_error=errors,functional_models=18,functional_updates=180,both_encoders_and_conditions_updated=True,
        shared_initialization_and_within_protocol_actual_random_streams_equal=True,matched_protocol_random_streams_equal=True,
        self_only_cross_descriptor_exclusion=True,save_reload_exact=True,incomplete_barrier_rejected=True,
        independent_metric_error=metric_error,test_network_inference_calls=0,**synthetic))
    print('PREFLIGHT PASSED',str(output),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); check(parser.parse_args().output.resolve())
