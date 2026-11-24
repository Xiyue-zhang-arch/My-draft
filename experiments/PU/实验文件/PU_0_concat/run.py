"""Thirty new PU_0 concatenation/unimodal models with a verified test barrier."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import argparse,csv,shutil,time,traceback
from pathlib import Path
import numpy as np
from common import *
from support import load_bank,make_model,predict,initial_hashes,verify_one
from prepare import VERSION, data_verify, describe_physical, source_path, verify_frozen_inputs

def norm(parameters):
    a=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(a).sum().sqrt() if a else torch.zeros((),device='cuda')
def train_one(spec,bank,folder,functional=False,callback=None):
    folder.mkdir(parents=True,exist_ok=False);net=make_model(spec);initial=initial_hashes(net)
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0.);seed(spec['seed']+100000)
    trace=[];zero=[];changed_initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if k.startswith('condition.')}
    epochs=1 if functional else 50;started=time.perf_counter()
    with (folder/'history.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['epoch','train_CE','val_CE','val_BA','updates','gradient_v','gradient_c','gradient_condition','seconds'])
        for epoch in range(1,epochs+1):
            tick=time.perf_counter();perm=np.random.default_rng(np.random.SeedSequence([6101,spec['seed'],epoch])).permutation(3200)
            ids=bank['train']['ids'][perm];order=torch.from_numpy(ids).cuda();net.train()
            t=dict(epoch=epoch,actual_window_order_sha256=ah(ids),cuda_rng_start_sha256=ah(torch.cuda.get_rng_state().numpy()),head_width=net.head_width,dropout_calls=50)
            total=torch.zeros((),device='cuda');g=torch.zeros(3,device='cuda');updates=0
            for b in range(0,len(ids),64):
                chosen=order[b:b+64];optimizer.zero_grad(set_to_none=True)
                logits,_=net(bank['x'][chosen],bank['d'][chosen]);loss=torch.nn.functional.cross_entropy(logits,bank['y'][chosen]);loss.backward()
                g+=torch.stack([norm(net.v.parameters()) if hasattr(net,'v') else torch.zeros((),device='cuda'),
                                norm(net.c.parameters()) if hasattr(net,'c') else torch.zeros((),device='cuda'),
                                norm(net.condition.parameters()) if hasattr(net,'condition') else torch.zeros((),device='cuda')])
                optimizer.step();total+=loss.detach()*len(chosen);updates+=1
            t['cuda_rng_end_sha256']=ah(torch.cuda.get_rng_state().numpy());trace.append(t)
            ce=float(total.cpu())/3200;grads=(g/50).cpu().tolist()
            if updates!=50 or not np.isfinite(ce) or not np.isfinite(grads).all():raise ValueError('Incomplete/nonfinite optimization')
            if any(grads[m]==0 for m in net.modalities):zero.append(epoch)
            val=predict(net,bank['val']['x'],bank['val']['d'],bank['val']['y'])
            writer.writerow([epoch,ce,val['metrics']['CE'],val['metrics']['BA'],updates,*grads,time.perf_counter()-tick]);f.flush()
            if callback:callback(dict(epoch=epoch,train_CE=ce,val_CE=val['metrics']['CE'],val_BA=val['metrics']['BA']))
    current=initial_hashes(net)
    if functional and any(current[k]==h for k,h in initial.items() if k in ('vibration','current')):raise ValueError('Functional encoder failed to update')
    condition_changed=any(not torch.equal(v,net.state_dict()[k].cpu()) for k,v in changed_initial.items()) if changed_initial else False
    if functional and net.physical and not condition_changed:raise ValueError('Functional condition module failed to update')
    if functional and net.head_width==128 and torch.equal(net.head[1].weight[:,:64],net.head[1].weight[:,64:]):raise ValueError('Concatenated classifier remained tied')
    torch.save(dict(spec=spec,state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch=epochs,functional=functional,
                    normalization_sha256=sha(bank['cache']/'normalization.json')),folder/'model.pt')
    npz(folder/'validation.npz',logits=val['logits'],probability=val['probability'],scale_mean=val['scale_mean'],y=bank['val']['y'],window_index=bank['val']['ids'])
    write(folder/'training_trace.json',trace)
    row=dict(**spec,functional=functional,epochs=epochs,updates=epochs*50,head_width=net.head_width,modalities=list(net.modalities),
             parameters=sum(p.numel() for p in net.parameters()),condition_parameters=sum(p.numel() for p in net.condition.parameters()) if net.physical else 0,
             shared_initialization_sha256=initial,condition_updated=condition_changed,zero_active_encoder_gradient_epochs=zero,
             val_CE=val['metrics']['CE'],val_BA=val['metrics']['BA'],seconds=time.perf_counter()-started,
             artifacts_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    del net,optimizer
    return row

def verify_all(out,rows,bank):
    if len(rows)!=30 or [{k:r[k] for k in ('model_id','method','seed')} for r in rows]!=models():raise ValueError('Exact 30 complete models required')
    traces={};inits={};maxerror=0.
    for row in rows:
        if row['functional'] or row['epochs']!=50 or row['updates']!=2500:raise ValueError('Incomplete/functional model')
        ce=verify_one(out,row,bank);maxerror=max(maxerror,abs(ce-row['val_CE']))
        group=(row['seed'],row['head_width']);t=read(out/'models'/row['model_id']/'training_trace.json')
        if group in traces and t!=traces[group]:raise ValueError('Within-width actual batch/Dropout streams differ')
        traces[group]=t
        for role,h in row['shared_initialization_sha256'].items():
            key=(row['seed'],role)
            if key in inits and inits[key]!=h:raise ValueError('Shared role initialization differs')
            inits[key]=h
    return dict(stage='all_weights_verified',models=30,strict_validation_reload_equal=True,max_CE_error=maxerror,
                paired_random_streams_equal_within_classifier_width=True,cross_width_dropout_equality_not_claimed=True)
def barrier(state):
    if not state['decisions_locked'] or not state['weights_verified'] or len(state['completed_models'])!=30:raise ValueError('Complete verified 30-model barrier required')
def test_all(out,bank,state,update):
    barrier(state);disk=np.load(DATA/'signals.npy',mmap_mode='r');noise=np.load(DATA/'vibration_noise_templates.npy',mmap_mode='r')
    index=np.load(FROZEN_SPLIT/'split_indices.npz');windows=read(DATA/'windows.json')
    nets=[]
    for spec in models():
        (out/'predictions'/spec['model_id']).mkdir();net=make_model(spec)
        ck=torch.load(out/'models'/spec['model_id']/'model.pt',map_location='cpu',weights_only=True);net.load_state_dict(ck['state_dict'],strict=True);nets.append((spec,net))
    descriptions=[]
    for case in cases():
        ids=index[case['target']];y=np.array([windows[int(i)]['label'] for i in ids])
        raw=torch.from_numpy(np.array(disk[ids])).cuda()
        if case['noise_instance'] is not None:raw[:,0]+=torch.from_numpy(np.array(noise[ids-4000,case['noise_instance']])).cuda()*10**(-case['snr_db']/20)
        x=wave_transform(raw,bank['norm']);d=descriptor_transform(describe_physical(raw),bank['norm'])
        wh=ah(x.cpu().numpy());dh=ah(d.cpu().numpy())
        desc=dict(**case,waveform_input_sha256=wh,descriptor_input_sha256=dh)
        for spec,net in nets:
            p=predict(net,x,d,y)
            npz(out/'predictions'/spec['model_id']/(case['case_id']+'.npz'),logits=p['logits'],probability=p['probability'],scale_mean=p['scale_mean'],
                y=y,window_index=ids,waveform_input_sha256=wh,descriptor_input_sha256=dh,case_id=case['case_id'],target=case['target'],**spec)
            state['prediction_cases']+=1
        descriptions.append(desc);update();print('TEST',case['case_id'],state['prediction_cases'],'/1890',flush=True)
    write(out/'test_cases.json',descriptions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();setup();before=source_hashes();pre=read(args.preflight);cache=args.cache.resolve();out=args.output.resolve()
    if pre['stage']!='preflight_passed' or pre['source_hashes']!=before or pre['cache_verification_sha256']!=sha(cache/'verification.json'):raise ValueError('Missing matching preflight')
    data_verify(cache)
    verify_frozen_inputs()
    out.mkdir(parents=True,exist_ok=False);lock=PROJECT/'ACTIVE_RUN.json'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),output=str(out)),f)
    state=dict(stage='training',started_at=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),cache=str(cache),source_hashes=before,
               completed_models=[],expected_models=30,prediction_cases=0,expected_new_predictions=1890,referenced_predictions=0,descriptor_version=VERSION,
               decisions_locked=False,weights_verified=False,target_inference_accessed=False)
    def update():write(out/'config.json',state)
    try:
        update()
        for name in ('models','predictions','source'):(out/name).mkdir()
        for name in before:shutil.copy2(source_path(name),out/'source'/name)
        write(out/'model_plan.json',models());write(out/'case_plan.json',cases())
        write(out/'data_reference.json',dict(data=str(DATA),cache=str(cache),data_verification_sha256=sha(DATA/'verification.json'),cache_verification_sha256=sha(cache/'verification.json'),frozen_split=str(FROZEN_SPLIT),frozen_split_verification_sha256=sha(FROZEN_SPLIT/'verification.json')))
        bank=load_bank(cache)
        for spec in models():
            state['active_model']=spec
            def callback(v):state['active_progress']=v;update()
            row=train_one(spec,bank,out/'models'/spec['model_id'],callback=callback);state['completed_models'].append(row);update()
            print('TRAINED',spec['model_id'],spec['method'],spec['seed'],flush=True)
        state['stage']='weights_verification';update();proof=verify_all(out,state['completed_models'],bank)
        if source_hashes()!=before:raise ValueError('Source changed')
        data_verify(cache);write(out/'weights_verification.json',proof)
        state.update(stage='testing',weights_verified=True,decisions_locked=True,locked_at=time.strftime('%Y-%m-%d %H:%M:%S'));update()
        state['target_inference_accessed']=True;update();test_all(out,bank,state,update)
        state['stage']='reporting';update()
        from report import generate
        generate(out)
        state['stage']='auditing';update()
        from audit import audit
        audit(out,out/'independent_audit.json')
        if source_hashes()!=before:raise ValueError('Source changed before completion')
        state.update(stage='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'));update()
        write(out/'completion_manifest.json',dict(stage='complete_and_frozen',new_models=30,reference_models=0,new_predictions=1890,reference_predictions=0,
               new_confusion_cells=30240,combined_confusion_cells=30240,files_sha256={n:sha(out/n) for n in ('config.json','independent_audit.json','report/REPORT.md','report/REVIEW.md','report/verification.json')}))
        lock.unlink();print('COMPLETE',str(out),flush=True)
    except BaseException:state.update(stage='failed',error=traceback.format_exc());update();raise
if __name__=='__main__':main()
