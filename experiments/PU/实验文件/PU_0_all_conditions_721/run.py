"""Thirty all-condition models followed by complete mixed-condition evaluation."""
import os
import torch
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from common import ROOT,PROJECT,DATA,models,cases,descriptor_profile,sha,write_json,setup,waveform_transform,descriptor_transform
from model import Network
from prepare import CACHE, VERSION, descriptor_banks, source_path, verify_frozen_inputs,array_hash
from support import SOURCE_FILES,source_hashes,load_bank,verify_cache,predict
from engine import train_one,verify_all,save_output

def evaluate(out,bank,state,update):
    if not state['weights_verified'] or not state['decisions_locked'] or len(state['completed_models'])!=30:
        raise ValueError('All thirty verified models required before testing')
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    disk=np.load(DATA/'signals.npy',mmap_mode='r'); noise=np.load(DATA/'vibration_noise_templates.npy',mmap_mode='r')
    nets=[]
    for spec in models():
        ck=torch.load(out/'models'/spec['model_id']/'model.pt',weights_only=True,map_location='cpu')
        net=Network(spec['method']).cuda(); net.load_state_dict(ck['state_dict'],strict=True); nets.append((spec,net))
        (out/'predictions'/spec['model_id']).mkdir()
    descriptions=[]
    for case in cases():
        ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64')
        noise_positions=np.searchsorted(split['test'],ids)
        raw=torch.from_numpy(np.array(disk[ids])).cuda()
        if case['noise_instance'] is not None:
            raw[:,0]+=torch.from_numpy(np.array(noise[noise_positions,case['noise_instance']])).cuda()*10**(-case['snr_db']/20)
        x=waveform_transform(raw,bank['norm']); db=descriptor_banks(raw)
        ds={p:descriptor_transform(v,bank['norm'],p) for p,v in db.items()}
        wh=array_hash(x.cpu().numpy()); dh={p:array_hash(v.cpu().numpy()) for p,v in ds.items()}
        metadata=dict(**case,kind='main',waveform_input_sha256=wh,descriptor_input_sha256=dh)
        if case['noise_instance'] is not None:
            clean=np.array(disk[ids,0],dtype='float64'); perturb=raw[:,0].cpu().numpy().astype('float64')-clean
            snr=10*np.log10(np.mean((clean-clean.mean(-1,keepdims=True))**2,axis=1)/np.mean(perturb**2,axis=1))
            metadata['postprocessing_window_SNR_min_median_max']=np.quantile(snr,[0,.5,1]).tolist()
        for spec,net in nets:
            profile=descriptor_profile(spec['method']); p=predict(net,x,ds[profile],y,extras=case['noise_instance'] is None)
            save_output(out/'predictions'/spec['model_id']/(case['case_id']+'.npz'),p,y,ids,
                        dict(**spec,case_id=case['case_id'],target=case['target'],kind='main',
                             waveform_input_sha256=wh,descriptor_input_sha256=dh[profile]),extras=case['noise_instance'] is None)
            state['prediction_cases']+=1
        descriptions.append(metadata); update()
        print('TEST',case['case_id'],state['prediction_cases'],'/2520',flush=True)
    write_json(out/'test_cases.json',descriptions)
    assert state['prediction_cases']==2520

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args(); setup(); out=args.output.resolve(); before=source_hashes()
    pre=json.loads(args.preflight.read_text(encoding='utf-8'))
    if pre['stage']!='preflight_passed' or pre['source_hashes']!=before or pre['cache_verification_sha256']!=sha(CACHE/'verification.json'):
        raise ValueError('Matching complete preflight required')
    verify_cache(CACHE); verify_frozen_inputs()
    lock=PROJECT/'ACTIVE_RUN.json'
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),output=str(out)),f)
    out.mkdir(parents=True,exist_ok=False)
    state=dict(stage='training',pid=os.getpid(),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),cache=str(CACHE),descriptor_version=VERSION,
        source_hashes=before,expected_models=30,expected_prediction_cases=2520,completed_models=[],prediction_cases=0,
        decisions_locked=False,weights_verified=False,test_inference_accessed=False,preflight_sha256=sha(args.preflight),
        experiment_type='all_conditions_stratified_record_holdout_721',historical_dataset_exposure=True)
    def update():
        write_json(out/'config.json',state)
    try:
        update()
        for name in ('source','models','predictions'):
            (out/name).mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(source_path(name),out/'source'/name)
        write_json(out/'model_plan.json',models()); write_json(out/'case_plan.json',cases())
        write_json(out/'data_reference.json',dict(cache=str(CACHE),verification_sha256=sha(CACHE/'verification.json'),
            frozen_inputs=str(DATA),frozen_inputs_verification_sha256=sha(DATA/'verification.json')))
        bank=load_bank(CACHE)
        for spec in models():
            state['active_model']=spec
            def callback(value):
                state['active_progress']=value; update()
            row=train_one(spec,bank,out/'models'/spec['model_id'],callback=callback)
            state['completed_models'].append(row); update()
        state['stage']='weights_verification'; update()
        proof=verify_all(out,state['completed_models'],bank)
        if source_hashes()!=before:
            raise ValueError('Code changed during training')
        write_json(out/'weights_verification.json',proof)
        state.update(stage='testing',weights_verified=True,decisions_locked=True,locked_at=time.strftime('%Y-%m-%d %H:%M:%S'),test_inference_accessed=True); update()
        evaluate(out,bank,state,update)
        state['stage']='reporting'; update()
        from report import generate
        generate(out)
        state['stage']='auditing'; update()
        from audit import audit
        audit(out)
        if source_hashes()!=before:
            raise ValueError('Code changed before completion')
        state.update(stage='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S')); update()
        write_json(out/'RUN_COMPLETE.json',dict(stage='complete',config_sha256=sha(out/'config.json'),audit_sha256=sha(out/'independent_audit.json')))
        lock.unlink(); print('COMPLETE',str(out),flush=True)
    except BaseException:
        state.update(stage='failed',error=traceback.format_exc()); update(); raise

if __name__=='__main__':
    main()
