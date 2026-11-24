"""训练及固定验证集诊断；不读取测试预测，不修改V1。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import pandas as pd
import json
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT/'experiments/CWRU/实验文件/CWRU_V1'))
import runtime as R
from splits import read_fixed
from reporting import scores

BASE = PROJECT/'experiments/CWRU/实验结果/CWRU_V1/baseline_20261020'
OUT = PROJECT/'experiments/CWRU/实验结果/CWRU_V2/workflow_20261020'


@torch.no_grad()
def features(model, x, rpm, device):
    return np.concatenate([model.evidence.extract(torch.as_tensor(x[i:i+32],device=device),
        torch.as_tensor(rpm[i:i+32],device=device)/60)[0].cpu().numpy() for i in range(0,len(x),32)])


@torch.no_grad()
def inspect(model, x, rpm, labels, device):
    n=model.network
    predictions={a:[] for a in [0.,.1,.5,1.]}
    details=[]
    for start in range(0,len(x),32):
        wave=torch.as_tensor(x[start:start+32],device=device)
        speed=torch.as_tensor(rpm[start:start+32],device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            seq=n.encoder(wave)
        d=n.data_norm(seq.mean(1)+(seq.var(1,unbiased=False)+1e-8).sqrt())
        f,_=model.evidence.extract(wave,speed/60)
        p=n.physics(model.evidence.standardize(f))
        gate=n.gate(torch.cat([d,p],dim=-1))
        residual=gate*p
        for alpha in predictions:
            logits=n.classifier(n.dropout(n.fused_norm(d+alpha*residual)))
            predictions[alpha].append(logits.softmax(-1).cpu().numpy())
        values=torch.stack([d.norm(dim=1),p.norm(dim=1),residual.norm(dim=1),
            gate.mean(dim=1),gate.std(dim=1)],dim=1).cpu().numpy()
        details.extend(values)
    predictions={a:np.concatenate(v) for a,v in predictions.items()}
    table=pd.DataFrame(details,columns=['data_norm','physical_norm','residual_norm','gate_mean','gate_sd'])
    table['residual_data_ratio']=table.residual_norm/table.data_norm.clip(lower=1e-8)
    table['label']=labels
    table['correct']=predictions[1.].argmax(1)==labels
    return predictions,table


def run():
    if (BASE/'config.json').exists() is False:
        raise FileNotFoundError(BASE)
    cfg=json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    assert cfg['stage']=='complete' and cfg['trained_models']==100
    OUT.mkdir(parents=True,exist_ok=False)
    out=OUT/'diagnosis';out.mkdir()
    (OUT/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    R.write_json(OUT/'config.json',dict(stage='diagnosis',baseline=str(BASE),trained_models=0,
        target_test_accessed=False,started_at=__import__('datetime').datetime.now().astimezone().isoformat()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R.U.DEVICE=device;R.U.configure()
    _,original,raw=R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    w=read_fixed(BASE/'splits/window_random')
    np.testing.assert_array_equal(w.window_index,original.window_index)
    selected=w.role.isin(['train','validation'])
    a=w.loc[selected].reset_index(drop=True)
    x=R.U.normalize(raw[selected.to_numpy()])
    rpm=a.rpm.to_numpy(np.float32);y=a.label.to_numpy()
    ti=np.flatnonzero(a.role.eq('train'));vi=np.flatnonzero(a.role.eq('validation'))
    validation_weights=R.U.weights_for(a,vi)
    curves=[]; validations=[]; interventions=[]; details=[]
    for protocol in ['clean_only','awgn_mixed']:
        for method in ['fusion','data_only']:
            for seed in range(1706,1711):
                folder=BASE/'models/window_random'/protocol/method/f'seed_{seed}'
                h=pd.read_csv(folder/'training.csv')
                info=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
                assert h.epoch.tolist()==list(range(1,len(h)+1))
                assert h.updates.eq(16*h.epoch).all()
                assert info['label_smoothing']==0 and info['loss']=='cross_entropy'
                best=h[h.checkpoint_updated].iloc[-1]
                assert int(best.epoch)==info['best_epoch']
                curves.append(dict(protocol=protocol,method=method,seed=seed,epochs=len(h),
                    best_epoch=info['best_epoch'],best_nll=info['validation_nll'],
                    last_nll=float(h.validation_nll.iloc[-1]),last_train_ce=float(h.train_loss.iloc[-1]),
                    early_stopped=info['early_stopped'],best_in_last_10=info['best_epoch']>=91))
                model=R.load_model(folder,device)
                nll=[]
                for condition,noise_seed,observed in R.conditions(x[vi],protocol,False):
                    probabilities=R.predict(model,observed,rpm[vi],device)
                    loss=R.U.nll(probabilities,y[vi],validation_weights)
                    values,_=scores(y[vi],probabilities)
                    nll.append(loss)
                    validations.append(dict(protocol=protocol,method=method,seed=seed,
                        condition=condition,noise_seed=noise_seed,validation_nll=loss,**values))
                score=nll[0] if protocol=='clean_only' else .5*nll[0]+.1*sum(nll[1:])
                np.testing.assert_allclose(score,info['validation_nll'],atol=2e-6,rtol=1e-5)
                if method=='fusion':
                    # 仅固定验证集的反事实诊断，不冒充重训后的候选。
                    views=[('clean',x[vi]),('white_-4',R.fixed(x[vi],-4,6201,1)[0])]
                    for condition,observed in views:
                        predicted,table=inspect(model,observed,rpm[vi],y[vi],device)
                        for alpha,p in predicted.items():
                            values,_=scores(y[vi],p)
                            interventions.append(dict(protocol=protocol,seed=seed,condition=condition,
                                residual_multiplier=alpha,nll=R.U.nll(p,y[vi],validation_weights),**values))
                        table['window_index']=a.iloc[vi].window_index.to_numpy()
                        table['record_id']=a.iloc[vi].record_id.to_numpy()
                        table['protocol']=protocol;table['seed']=seed;table['condition']=condition
                        details.append(table)
                del model
                print('BASELINE VALIDATION',protocol,method,seed,'OK',flush=True)
    pd.DataFrame(curves).to_csv(out/'training_diagnosis.csv',index=False)
    pd.DataFrame(validations).to_csv(out/'baseline_validation.csv',index=False)
    pd.DataFrame(interventions).to_csv(out/'residual_interventions.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'fusion_validation_windows.csv',index=False)
    model=R.load_model(BASE/'models/window_random/awgn_mixed/fusion/seed_1706',device)
    clean=features(model,x,rpm,device)
    center=model.evidence.center.cpu().numpy();scale=model.evidence.scale.cpu().numpy()
    clean_z=(clean-center)/scale
    # 分离度由训练数据计算，噪声敏感性分别在训练和验证上汇总。
    means=np.stack([clean_z[ti][y[ti]==c].mean(0) for c in range(4)])
    between=means.var(0)
    within=np.mean(np.stack([clean_z[ti][y[ti]==c].var(0) for c in range(4)]),axis=0)
    separation=between/(within+1e-8)
    types=['harmonic_strength','coverage','harmonic_contrast','recurrence','recurrence_contrast',
           'phase_concentration','phase_contrast','event_rate']
    shapes=['top_energy_fraction','kurtosis','crest','relative_rms','spectral_flatness']
    names=[dict(feature=j,band=j//53,
        family=R.U.CFG['families'][(j%53)//8] if j%53<48 else 'shape',
        descriptor=types[j%8] if j%53<48 else shapes[j%53-48]) for j in range(212)]
    # 在频带内部索引描述类型，不能直接用全局j%8。
    for row in names:
        k=row['feature']%53
        if k<48:row['descriptor']=types[k%8]
    feature_rows=[]
    for snr in [-4,0,4]:
        for ns in [6201,6202,6203]:
            # role内生成噪声，与既有验证的窗口顺序保持一致。
            for role,ix in [('train',ti),('validation',vi)]:
                observed=R.fixed(x[ix],snr,ns,1)[0]
                z=(features(model,observed,rpm[ix],device)-center)/scale
                difference=z-clean_z[ix]
                for label in [-1,0,1,2,3]:
                    mask=np.ones(len(ix),dtype=bool) if label==-1 else y[ix]==label
                    if not mask.any():continue
                    for row in names:
                        j=row['feature']
                        feature_rows.append(dict(**row,role=role,label=label,snr=snr,noise_seed=ns,
                            clean_separation=float(separation[j]),
                            normalized_rmse=float(np.sqrt(np.mean(difference[mask,j]**2))),
                            clean_clip_fraction=float(np.mean(abs(clean_z[ix][mask,j])>=8)),
                            noisy_clip_fraction=float(np.mean(abs(z[mask,j])>=8))))
        print('FEATURE DIAGNOSIS SNR',snr,flush=True)
    f=pd.DataFrame(feature_rows)
    f.to_csv(out/'physical_feature_diagnosis.csv',index=False)
    grouped=f.groupby(['role','label','snr','band','descriptor'],as_index=False)[
        ['clean_separation','normalized_rmse','clean_clip_fraction','noisy_clip_fraction']].mean()
    grouped.to_csv(out/'physical_group_diagnosis.csv',index=False)
    v=pd.DataFrame(validations)
    iv=pd.DataFrame(interventions)
    mean=v.groupby(['protocol','method','condition'])[['balanced_acc','validation_nll']].mean().reset_index()
    imean=iv.groupby(['protocol','condition','residual_multiplier'])[['balanced_acc','nll']].mean().reset_index()
    # 仅建议候选顺序，不根据测试信息决策；两候选均仍须重训验证。
    subset=imean[(imean.protocol=='awgn_mixed')&(imean.condition=='white_-4')]
    base=float(subset.loc[subset.residual_multiplier.eq(1),'nll'].iloc[0])
    reduced=float(subset.loc[subset.residual_multiplier.eq(.1),'nll'].iloc[0])
    R.write_json(out/'decision.json',dict(baseline_validation_reproduced=True,
        next_stage='paired_small_sample_baseline',candidate_order=['residual_scale','data_aux_ce'],
        residual_scaling_supported_by_counterfactual_nll=reduced<base,
        baseline_noisy_validation_nll=base,reduced_noisy_validation_nll=reduced,
        test_predictions_read=False,notes='干预只用于诊断，不视为候选模型结果；按预定两候选上限探索'))
    (out/'REPORT.md').write_text('# 阶段1至3 训练、物理特征和融合诊断\n\n'
        '只读V1权重、训练记录以及训练/验证窗口；未读取目标测试预测。20个基础模型的最佳验证NLL与训练记录一致，基线可运行。\n\n'
        '## 验证集结果\n\n'+mean.round(6).to_markdown(index=False)+'\n\n'
        '## 固定验证集物理残差干预\n\n'+imean.round(6).to_markdown(index=False)+'\n\n'
        '残差倍率0仍保留融合LayerNorm，不等于原数据单分支。门控均值不是物理贡献率。干预后没有重新训练，不能把其结果当作新方法性能。\n\n'
        '物理特征分析在当前训练集标准化坐标中报告加噪偏移、裁剪比例和训练类间/类内分离度。该诊断使用3个固定验证噪声种子，'
        '正式选模仍只用原6201。诊断没有自动去噪或按类修改特征。\n\n'
        '## 阶段决策\n\n进入固定嵌套小样本基线。之后最多分别检验残差缩放和数据辅助CE两个单变量候选，'
        '每个直接对比原融合。是否进入跨负载由预先规定的验证门槛决定，不追逐测试性能。\n',
        encoding='utf-8')
    R.write_json(OUT/'config.json',dict(stage='diagnosis_complete',baseline=str(BASE),trained_models=0,
        target_test_accessed=False,diagnosis_models=20,decision='proceed_paired_small_sample_baseline'))
    print('DIAGNOSIS COMPLETE',OUT,flush=True)


if __name__=='__main__':
    run()
