"""Thirty joint models; 1890 primary and 90 diagnostic cases only after the complete barrier."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from common import (ROOT,PROJECT,DATA,TARGETS,METHODS,BATCH,EPOCHS,UPDATES_PER_EPOCH,ETA,models,cases,diagnostic_cases,descriptor_profile,
                    sha,write_json,save_npz,setup,seed_model,waveform_transform,descriptor_transform,metrics)
from model import Network
from prepare import descriptor_banks,data_verify
from support import SOURCE_FILES,source_hashes,load_bank,verify_cache,predict,fit_probes


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def shared_hash(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():
        if not k.startswith('condition.'):
            h.update(k.encode()); h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def gradient_norm(parameters):
    terms=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(terms).sum().sqrt() if terms else torch.zeros((),device='cuda')


def save_output(path,p,y,ids,metadata,extras=False):
    values={k:p[k] for k in ('logits','probability','scale_mean','scale_channel_mean','scale_range')}
    if extras:
        values.update({k:p[k] for k in ('pre_features','channel_scales')})
    save_npz(path,**values,y=y,window_index=ids,**metadata)


def train_one(spec,bank,folder,functional=False,callback=None):
    folder.mkdir(parents=True,exist_ok=False)
    seed_model(spec['seed']); net=Network(spec['method']).cuda(); init=shared_hash(net)
    count=sum(p.numel() for p in net.parameters()); conditional=sum(p.numel() for n,p in net.named_parameters() if n.startswith('condition.'))
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0.)
    seed_model(spec['seed']+100000)
    profile=descriptor_profile(spec['method']); traces=[]; started=time.perf_counter()
    with (folder/'history.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['epoch','train_CE','val_CE','val_BA','updates','gradient_v','gradient_c','gradient_condition','seconds'])
        for epoch in range(1,(1 if functional else EPOCHS)+1):
            tick=time.perf_counter()
            permutation=np.random.default_rng(np.random.SeedSequence([6101,spec['seed'],epoch])).permutation(len(bank['train']['ids']))
            ids=bank['train']['ids'][permutation]
            traces.append(dict(epoch=epoch,global_window_permutation_sha256=array_hash(ids),CUDA_rng_start_sha256=array_hash(torch.cuda.get_rng_state().cpu().numpy())))
            order=torch.from_numpy(ids).cuda(); net.train(); total=torch.zeros((),device='cuda'); grads=torch.zeros(3,device='cuda'); updates=0
            for begin in range(0,len(order),BATCH):
                batch=order[begin:begin+BATCH]; optimizer.zero_grad(set_to_none=True)
                logits,_,_=net(bank['x'][batch],bank['d'][profile][batch])
                loss=torch.nn.functional.cross_entropy(logits,bank['y'][batch]); loss.backward()
                grads+=torch.stack([gradient_norm(net.v.parameters()),gradient_norm(net.c.parameters()),
                                    gradient_norm(net.condition.parameters()) if hasattr(net,'condition') else torch.zeros((),device='cuda')])
                optimizer.step(); total+=loss.detach()*len(batch); updates+=1
            ce=float(total.cpu())/len(ids); grad=(grads/updates).cpu().numpy()
            if updates!=UPDATES_PER_EPOCH or not np.isfinite(ce) or not np.isfinite(grad).all():
                raise ValueError('Invalid real optimization or gradients')
            val=predict(net,bank['val']['x'],bank['val']['d'][profile],bank['val']['y'])
            writer.writerow([epoch,ce,val['metrics']['CE'],val['metrics']['BA'],updates,*grad.tolist(),time.perf_counter()-tick]); f.flush()
            progress=dict(epoch=epoch,train_CE=ce,val_CE=val['metrics']['CE'],val_BA=val['metrics']['BA'])
            if callback:
                callback(progress)
            if epoch==1 or epoch%10==0 or functional:
                print(spec['model_id'],spec['method'],spec['seed'],'epoch',epoch,'train',round(ce,6),'val',round(val['metrics']['CE'],6),flush=True)
    before=predict(net,bank['val']['x'],bank['val']['d'][profile],bank['val']['y'],extras=True)
    norm_hash=sha(bank['cache']/'normalization.json')
    with (folder/'model.pt').open('xb') as f:
        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},spec=spec,epoch=epoch,
                        functional=functional,normalization_sha256=norm_hash),f)
    del net
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    net=Network(spec['method']).cuda(); net.load_state_dict(ck['state_dict'],strict=True)
    after=predict(net,bank['val']['x'],bank['val']['d'][profile],bank['val']['y'],extras=True)
    for k in ('logits','scale_mean','pre_features','channel_scales'):
        if not np.array_equal(before[k],after[k]):
            raise ValueError('Strict reload changed source validation outputs')
    save_output(folder/'validation.npz',after,bank['val']['y'],bank['val']['ids'],spec,extras=True)
    training=predict(net,bank['train']['x'],bank['train']['d'][profile],bank['train']['y'],extras=True)
    save_npz(folder/'train_features.npz',pre_features=training['pre_features'],y=bank['train']['y'],window_index=bank['train']['ids'])
    probe=fit_probes(training['pre_features'],bank['train']['y'],after['pre_features'])
    save_npz(folder/'probe.npz',**probe,training_window_index=bank['train']['ids'],validation_window_index=bank['val']['ids'])
    write_json(folder/'training_trace.json',traces)
    row=dict(**spec,functional=functional,epochs=epoch,updates=UPDATES_PER_EPOCH*epoch,parameters=count,condition_parameters=conditional,
             shared_initialization_sha256=init,val_CE=after['metrics']['CE'],val_BA=after['metrics']['BA'],
             final_train_eval_CE=training['metrics']['CE'],final_train_eval_BA=training['metrics']['BA'],
             seconds=time.perf_counter()-started,checkpoint_reproduction_max_error=0.,
             artifacts_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    write_json(folder/'complete.json',row)
    return row


def verify_all(out,completed,bank):
    if len(completed)!=30 or [{k:r[k] for k in ('model_id','seed','method')} for r in completed]!=models():
        raise ValueError('All exact 30 models required before target testing')
    traces={}; initial={}; records=[]
    for row in completed:
        if row['functional'] or row['epochs']!=50 or row['updates']!=UPDATES_PER_EPOCH*EPOCHS:
            raise ValueError('Functional or incomplete model')
        folder=out/'models'/row['model_id']
        for name,expected in row['artifacts_sha256'].items():
            if sha(folder/name)!=expected:
                raise ValueError('Completed artifact changed')
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        if ck['functional'] or ck['epoch']!=50 or ck['normalization_sha256']!=sha(bank['cache']/'normalization.json'):
            raise ValueError('Checkpoint identity/normalization mismatch')
        spec={k:row[k] for k in ('model_id','seed','method')}
        if ck['spec']!=spec:
            raise ValueError('Checkpoint spec mismatch')
        net=Network(row['method']).cuda(); net.load_state_dict(ck['state_dict'],strict=True)
        val=predict(net,bank['val']['x'],bank['val']['d'][descriptor_profile(row['method'])],bank['val']['y'],extras=True)
        with np.load(folder/'validation.npz') as z:
            for key in ('logits','pre_features','channel_scales'):
                if not np.array_equal(z[key],val[key]):
                    raise ValueError('Full source validation verification failed')
            if not np.array_equal(z['window_index'],bank['val']['ids']):
                raise ValueError('Validation identity changed')
        if abs(row['val_CE']-val['metrics']['CE'])>1e-12:
            raise ValueError('Source CE failed recomputation')
        trace=json.loads((folder/'training_trace.json').read_text(encoding='utf-8'))
        seed=row['seed']
        if seed in traces and trace!=traces[seed]:
            raise ValueError('Cross-method training order/Dropout streams differ')
        if seed in initial and row['shared_initialization_sha256']!=initial[seed]:
            raise ValueError('Shared initialization differs')
        traces[seed]=trace; initial[seed]=row['shared_initialization_sha256']
        records.append(dict(model_id=row['model_id'],checkpoint_sha256=row['artifacts_sha256']['model.pt'],val_CE=val['metrics']['CE']))
    return dict(verified_count=30,shared_initialization_equal=True,batch_order_and_dropout_streams_equal=True,records=records)

