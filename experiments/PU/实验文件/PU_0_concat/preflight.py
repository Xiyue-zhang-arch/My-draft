"""Real-source checks and full-scope synthetic extension report."""
import torch
import argparse
from pathlib import Path
import numpy as np
from common import *
from support import load_bank,make_model,predict,verify_one,initial_hashes
from prepare import describe_physical, verify_frozen_inputs
from run import train_one,verify_all,barrier
from report import generate

def simulate(out,cache):
    out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    index=np.load(FROZEN_SPLIT/'split_indices.npz');windows=read(DATA/'windows.json')
    descriptions = [
        dict(**case, waveform_input_sha256='synthetic-wave', descriptor_input_sha256='synthetic-harmonic')
        for case in cases()
    ]
    write(out/'test_cases.json',descriptions)
    for spec in models():
        folder=out/'predictions'/spec['model_id'];folder.mkdir();n=2 if spec['method'].startswith('concat') else 1
        for case in cases():
            ids=index[case['target']];y=np.array([windows[int(i)]['label'] for i in ids]);a=np.tile(np.array([.1,.2,.3,.4],dtype='float32'),(4000,1));a[np.arange(4000),(y+spec['seed'])%4]+=1
            m,p=metrics(a,y);desc=next(r for r in descriptions if r['case_id']==case['case_id'])
            npz(folder/(case['case_id']+'.npz'),logits=a,probability=p,scale_mean=np.ones((4000,n)),y=y,window_index=ids,
                waveform_input_sha256=desc['waveform_input_sha256'],descriptor_input_sha256=desc['descriptor_input_sha256'],case_id=case['case_id'],target=case['target'],**spec)
    write(out/'config.json',dict(synthetic=True,stage='synthetic_reporting',completed_models=models(),cache=str(cache),decisions_locked=True,weights_verified=True,prediction_cases=1890))
    proof=generate(out,synthetic=True,figures=False)
    bad=out/'predictions/m0001'/(cases()[0]['case_id']+'.npz')
    with np.load(bad) as z:values={k:z[k] for k in z.files}
    values['y']=values['y'].copy();values['y'][0]=(values['y'][0]+1)%4
    with bad.open('wb') as f:np.savez_compressed(f,**values)
    try:generate(out,synthetic=True,figures=False)
    except ValueError as e:
        if 'identity' not in str(e):raise
    else:raise AssertionError('Tampering accepted')
    return dict(synthetic_new_prediction_rows=1890,frozen_reference_rows_recomputed=0,combined_metric_rows=1890,tampered_labels_rejected=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    setup();cache=args.cache.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    before=source_hashes();verify_frozen_inputs();bank=load_bank(cache);raw=np.array(np.load(DATA/'signals.npy',mmap_mode='r')[bank['train']['ids'][:8]])
    gpu = describe_physical(torch.from_numpy(raw).cuda()).cpu().numpy()
    expected = np.load(cache/'physical.npy')[bank['train']['ids'][:8]]
    cache_error = float(np.max(np.abs(gpu-expected)))
    if cache_error > 2e-7:
        raise ValueError('Four-harmonic descriptor cache differs')
    x=bank['train']['x'][:8];d=bank['train']['d'][:8]
    # Direct concatenation must be exactly the unscaled two post-CNN vectors.
    net=make_model(models()[4]);net.eval();features={};head_input=[]
    hooks=[net.v.net[-1].register_forward_hook(lambda m,a,b:features.__setitem__('v',b.detach())),
           net.c.net[-1].register_forward_hook(lambda m,a,b:features.__setitem__('c',b.detach())),
           net.head[0].register_forward_pre_hook(lambda m,a:head_input.append(a[0].detach()))]
    with torch.no_grad():a,_=net(x,d)
    if not torch.equal(head_input[0],torch.cat([features['v'],features['c']],1)):raise ValueError('Fusion is not direct concatenation')
    for hook in hooks:hook.remove()
    # In eval mode only, the deliberately matched initialization is equivalent to mean fusion.
    ref=torch.nn.functional.linear((features['v']+features['c'])*.5,net.head[1].weight[:,:64]*2,net.head[1].bias)
    initialization_error=float((a-ref).abs().max().detach())
    if initialization_error>2e-6:raise ValueError('Matched mean/concat initialization differs')
    del net
    (out/'models').mkdir();rows=[];traces={};initial={};exclusion_checks=0
    for spec in models()[:6]:
        row=train_one(spec,bank,out/'models'/spec['model_id'],functional=True);verify_one(out,row,bank);rows.append(row)
        group=row['head_width'];trace=read(out/'models'/spec['model_id']/'training_trace.json')
        if group in traces and trace!=traces[group]:raise ValueError('Within-width random stream mismatch')
        traces[group]=trace
        for role,h in row['shared_initialization_sha256'].items():
            if role in initial and initial[role]!=h:raise ValueError('Shared initialization differs')
            initial[role]=h
        ck=torch.load(out/'models'/spec['model_id']/'model.pt',map_location='cpu',weights_only=True)
        net=make_model(spec);net.load_state_dict(ck['state_dict'],strict=True);net.eval()
        if len(net.modalities)==1:
            other=1-net.modalities[0];changed_x=x.clone();changed_d=d.clone();changed_x[:,other]+=torch.randn_like(changed_x[:,other])*100;changed_d[:,other]+=100
            with torch.no_grad():a,_=net(x,d);b,_=net(changed_x,changed_d)
            if not torch.equal(a,b):raise ValueError('Unimodal model used the other modality')
            exclusion_checks+=1
        del net
        print('PREFLIGHT REAL',spec['method'],flush=True)
    try:verify_all(out,rows,bank)
    except ValueError as e:
        if '30' not in str(e):raise
    else:raise AssertionError('Functional models bypassed barrier')
    try:barrier(dict(completed_models=rows,decisions_locked=False,weights_verified=False))
    except ValueError:pass
    else:raise AssertionError('Unlocked targets accepted')
    simulated=simulate(out/'synthetic_report',cache)
    if source_hashes()!=before:raise ValueError('Code changed during preflight')
    proof=dict(stage='preflight_passed',source_hashes=before,cache_verification_sha256=sha(cache/'verification.json'),
               functional_models=6,real_epochs_each=1,updates_each=50,strict_reload_equal=True,
               four_harmonic_descriptor_cache_max_error=cache_error,direct_unscaled_concatenation_confirmed=True,mean_concat_eval_initialization_max_error=initialization_error,
               single_modality_waveform_and_descriptor_exclusion_checks=exclusion_checks,shared_role_initialization_equal=True,
               actual_random_streams_equal_within_head_width=True,cross_width_dropout_equality_not_claimed=True,
               incomplete_models_target_barrier_rejected=True,new_target_network_calls=0,**simulated)
    write(out/'preflight.json',proof);print('PREFLIGHT PASSED',str(out),flush=True)
if __name__=='__main__':main()
