"""统一V2训练器，所有候选使用相同采样、逐epoch选模和停止规则。"""
import time
import torch
import numpy as np
import pandas as pd
from models import Model,loss_for,TRAINABLE_VARIANTS
from diagnose import R


def train(x,w,protocol,variant,seed,subset,folder,device):
    if variant not in TRAINABLE_VARIANTS:
        raise ValueError(f'后续训练仅允许{TRAINABLE_VARIANTS}；{variant}仅保留历史加载支持')
    if folder.exists():raise FileExistsError(folder)
    R.U.seed_all(seed)
    ti=np.flatnonzero(w.role.eq('train'));vi=np.flatnonzero(w.role.eq('validation'))
    y=w.label.to_numpy();rpm=w.rpm.to_numpy(np.float32)
    weights=R.U.weights_for(w,vi)
    model=Model(variant).to(device)
    R.U.sync();start=time.perf_counter()
    model.fit(x[ti],rpm[ti],device)
    R.U.sync();fit_seconds=time.perf_counter()-start
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=.00001)
    views=list(R.conditions(x[vi],protocol,False))
    folder.mkdir(parents=True,exist_ok=False)
    w.iloc[ti].to_csv(folder/'selected_windows.csv',index=False)
    best,stale,step,best_epoch,best_update=float('inf'),0,0,0,0
    history=[]
    for epoch in range(1,101):
        order=R.U.sample_epoch(w,ti,np.random.default_rng(np.random.SeedSequence([subset,seed,epoch,20261011])))
        losses=[];seconds=0.
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        lr=optimizer.param_groups[0]['lr']
        for offset in range(0,len(order),32):
            model.train();R.U.sync();start=time.perf_counter()
            ix=order[offset:offset+32];observed=x[ix]
            if protocol=='awgn_mixed':
                observed=R.observe(observed,np.random.default_rng(np.random.SeedSequence(
                    [20261012,subset,seed,epoch,offset])))[0]
            optimizer.zero_grad(set_to_none=True)
            logits=model(torch.as_tensor(observed,device=device),torch.as_tensor(rpm[ix],device=device))
            total,main,aux=loss_for(model,logits,torch.as_tensor(y[ix],device=device))
            if not torch.isfinite(total):raise ValueError('nonfinite training loss')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();step+=1
            losses.append([float(total.detach()),float(main.detach()),float(aux.detach())])
            R.U.sync();seconds+=time.perf_counter()-start
        peak=float(torch.cuda.max_memory_allocated()/1024**2) if device.type=='cuda' else None
        start=time.perf_counter()
        values=[R.U.nll(R.predict(model,observed,rpm[vi],device),y[vi],weights) for _,_,observed in views]
        R.U.sync();val_seconds=time.perf_counter()-start
        score=values[0] if protocol=='clean_only' else .5*values[0]+.1*sum(values[1:])
        if not np.isfinite(score):raise ValueError('nonfinite validation')
        improved=score<best-1e-6
        if improved:
            best,stale,best_epoch,best_update=score,0,epoch,step
            torch.save(dict(variant=variant,model_config=R.U.CFG,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),folder/'model.pt')
        else:stale+=1
        averages=np.mean(losses,axis=0)
        history.append(dict(epoch=epoch,updates=step,lr=lr,train_loss=averages[0],main_ce=averages[1],
            auxiliary_ce=averages[2],validation_nll=score,stale_epochs=stale,checkpoint_updated=improved,
            train_seconds=seconds,validation_seconds=val_seconds,peak_gpu_mib=peak,
            alpha=float(model.alpha.detach()) if variant=='residual_scale' else 1.))
        pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
        scheduler.step()
        if epoch==1 or epoch%10==0 or stale>=20:
            print(f'{folder.parent.name}/{protocol}/{variant}/seed{seed} epoch={epoch} val={score:.6f} stale={stale}',flush=True)
        if stale>=20:break
    h=pd.DataFrame(history)
    info=dict(protocol=protocol,variant=variant,seed=seed,subset=subset,training_windows=len(ti),validation_windows=len(vi),
        epochs_trained=epoch,updates_trained=step,best_epoch=best_epoch,best_update=best_update,
        validation_nll=best,early_stopped=stale>=20,loss='ordinary_ce' if variant!='data_aux_ce' else 'ordinary_ce+0.1_data_ce',
        label_smoothing=0.,parameters=sum(p.numel() for p in model.parameters()),fit_seconds=fit_seconds,
        total_train_seconds=float(h.train_seconds.sum()),train_seconds_per_epoch=float(h.train_seconds.mean()),
        validation_seconds=float(h.validation_seconds.sum()),
        peak_gpu_mib=float(h.peak_gpu_mib.max()) if device.type=='cuda' else None)
    R.write_json(folder/'parameters.json',info)
    return info


def load(folder,device):
    saved=torch.load(folder/'model.pt',map_location=device,weights_only=True)
    model=Model(saved['variant']).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()
