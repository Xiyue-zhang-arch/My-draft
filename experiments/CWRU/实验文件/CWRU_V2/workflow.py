"""2026-10-20已完成的历史阶段实现，不作为后续实验入口。

历史候选顺序保留用于解释冻结结果。2026-10-21后续仅比较baseline_data与
data_aux_ce，见AUXILIARY_DESIGN.md；training.train已限制新增训练的变体。
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
import json
import math
from datetime import datetime
from diagnose import R,BASE,OUT,HERE
from models import complexity,baseline_equivalence
from training import train,load
from splits import read_fixed
from reporting import scores

from experiment_paths import resolve_result_path

SEEDS=[1706,1707,1708]
SUBSETS=[4101,4102,4103]
RATIOS=[.1,.2,.5]
PROTOCOLS=['clean_only','awgn_mixed']


def prepare_subsets(full,root):
    root.mkdir(exist_ok=False)
    rows=[];usage=[]
    for subset in SUBSETS:
        orders={}
        for label,g in full[full.role.eq('train')].groupby('label'):
            rng=np.random.default_rng(np.random.SeedSequence([subset,int(label),20261020]))
            # 按记录轮转构建每类固定顺序，小预算严格嵌套，不按结果挑子集。
            records=rng.permutation(g.record_index.unique())
            queues={rid:list(rng.permutation(g.index[g.record_index.eq(rid)].to_numpy())) for rid in records}
            order=[]
            while any(queues.values()):
                for rid in records:
                    if queues[rid]:order.append(queues[rid].pop())
            orders[int(label)]=order
        previous=set()
        for ratio in RATIOS:
            chosen=sorted(i for label,order in orders.items() for i in order[:math.ceil(ratio*len(order))])
            assert previous.issubset(chosen);previous=set(chosen)
            w=full.copy()
            w.loc[w.role.eq('train'),'role']='unused_train'
            w.loc[chosen,'role']='train'
            assert len(chosen)=={.1:49,.2:98,.5:242}[ratio]
            folder=root/f'ratio_{int(ratio*100):03d}_subset_{subset}';folder.mkdir()
            w.to_csv(folder/'windows.csv',index=False)
            for role in ['train','validation','test']:
                ids=w.loc[w.role.eq(role),'window_index']
                (folder/f'{role}.txt').write_text(''.join(f'{i}\n' for i in ids),encoding='utf-8')
            for rid,g in w.groupby('record_id'):
                usage.append(dict(ratio=ratio,subset=subset,record_id=rid,
                    train=int(g.role.eq('train').sum()),validation=int(g.role.eq('validation').sum()),
                    test=int(g.role.eq('test').sum()),unused_train=int(g.role.eq('unused_train').sum())))
            rows.append(dict(ratio=ratio,subset=subset,training_windows=len(chosen),validation_windows=103,test_windows=103))
    pd.DataFrame(rows).to_csv(OUT/'subset_support.csv',index=False)
    pd.DataFrame(usage).to_csv(OUT/'training_subset_file_usage.csv',index=False)
    (OUT/'DATA_USAGE.md').write_text('# V2数据使用\n\n全部40记录，完整687窗，原始数据只读。'
        '基础划分完全复用V1的481/103/103，不重抽验证测试。训练10%/20%/50%分别49/98/242窗，'
        '正常/IR/OR/B配额分别5/14/15/15、10/28/30/30、25/69/74/74。'
        '每预算3个固定嵌套子集，train/validation/test.txt为固定窗口编号。仍使用103验证标签。'
        '\n\n[原始MAT、通道与全部集合用量](../../CWRU_V1/baseline_20261020/dataset_file_usage.csv)'
        '\n\n[本轮逐记录小样本用量](training_subset_file_usage.csv)'
        '\n\n若进入跨负载，完全复用V1来源划分，使用全部来源训练窗口；不会按目标结果选取窗口。'
        '训练角色之外的来源窗口不挪入验证，目标记录不参与选模。',encoding='utf-8')


def validation(model,w,x,device,meta):
    ix=np.flatnonzero(w.role.eq('validation'))
    rpm=w.iloc[ix].rpm.to_numpy(np.float32);y=w.iloc[ix].label.to_numpy()
    weights=R.U.weights_for(w,ix)
    rows=[];arrays={'labels':y,'window_indices':w.iloc[ix].window_index.to_numpy()}
    for condition,noise_seed,observed in R.conditions(x[ix],meta['protocol'],False):
        p=R.predict(model,observed,rpm,device)
        values,_=scores(y,p)
        rows.append(dict(**meta,condition=condition,validation_nll=R.U.nll(p,y,weights),**values))
        arrays[f'{condition}_n{noise_seed}']=p
    return rows,arrays


def select_candidate(frame,variant,ratios,full=False):
    """所有筛选只使用验证；固定子集内训练种子与配对差值的SD都检查。"""
    f=frame[(frame.ratio.isin(ratios))&(frame.domain=='window_random')]
    keys=['ratio','subset','protocol','seed','condition']
    cand=f[f.variant.eq(variant)].set_index(keys)
    base=f[f.variant.eq('baseline_fusion')].set_index(keys)
    common=cand.index.intersection(base.index)
    if len(common)!=len(cand) or not len(common):raise ValueError('候选缺少配对基线')
    c=cand.loc[common].reset_index();b=base.loc[common].reset_index()
    c['gain']=c.balanced_acc.to_numpy()-b.balanced_acc.to_numpy()
    c['or_gain']=c.recall_OR.to_numpy()-b.recall_OR.to_numpy()
    noisy=c[c.protocol.eq('awgn_mixed')&c.condition.isin(['white_-4','white_-2'])]
    # 先在同模型平均两个噪声档，再对固定子集计算跨种子波动。
    paired=noisy.groupby(['ratio','subset','seed'],as_index=False)[['balanced_acc','gain','or_gain']].mean()
    # 每个噪声档分别检查种子波动，不能靠两档平均掩盖不稳定。
    sd=noisy.groupby(['ratio','subset','condition'])[['balanced_acc','gain']].std(ddof=1)
    gain=float(paired.groupby(['ratio','subset']).gain.mean().mean())
    positive=float((paired.gain>1e-12).mean())
    or_gain=float(paired.groupby(['ratio','subset']).or_gain.mean().mean())
    clean={p:float(c[c.protocol.eq(p)&c.condition.eq('clean')].groupby(['ratio','subset']).gain.mean().mean())
           for p in PROTOCOLS}
    if full:
        # 全样本要求维持性能，并检查数据单分支对照。
        data=f[f.variant.eq('baseline_data')].set_index(keys).loc[common].reset_index()
        gap=c.balanced_acc.to_numpy()-data.balanced_acc.to_numpy()
        data_gap=float(np.min(pd.Series(gap).groupby([c.protocol,c.condition]).mean()))
    else:data_gap=None
    conditions=dict(noisy_gain=gain>=(-.005 if full else .005),
        positive_pairs=True if full else positive>=2/3,
        or_preservation=or_gain>=-.01,
        clean_preservation=all(value>=-.005 for value in clean.values()),
        fixed_subset_seed_sd=bool((sd.balanced_acc<=.005+1e-12).all()),
        paired_gain_sd=bool((sd.gain<=.005+1e-12).all()),
        data_reference=True if not full else data_gap>=-.005)
    return dict(variant=variant,ratios=ratios,passed=all(conditions.values()),checks=conditions,
        noisy_gain_pp=100*gain,positive_pair_fraction=positive,or_gain_pp=100*or_gain,
        clean_gain_pp={k:100*v for k,v in clean.items()},
        max_seed_sd_pp=float(sd.balanced_acc.max()*100),max_paired_sd_pp=float(sd.gain.max()*100),
        min_data_gap_pp=None if data_gap is None else 100*data_gap,
        decision_data='fixed validation only; no test metrics')


def main():
    state=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    if state['stage']!='diagnosis_complete' or (OUT/'models').exists():
        raise ValueError('只启动一次；不覆盖或自动续训已有模型')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R.U.DEVICE=device;R.U.configure()
    records,original,raw=R.load_dataset(R.PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    full=read_fixed(BASE/'splits/window_random')
    np.testing.assert_array_equal(original.window_index,full.window_index)
    x=R.U.normalize(raw)
    state.update(stage='preparing_small_sample',hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        precision='CUDA encoder BF16, remainder FP32',baseline_small_planned=108,
        max_candidate_count=2,seed_list=SEEDS,subset_seeds=SUBSETS,validation_only_selection=True,
        candidate_order=['data_aux_ce','residual_scale'])
    R.write_json(OUT/'config.json',state)
    baseline_equivalence(device)
    prepare_subsets(full,OUT/'subsets')
    records.to_csv(OUT/'records.csv',index=False)
    (OUT/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    pd.DataFrame([complexity(v,device) for v in ['baseline_fusion','baseline_data','data_aux_ce','residual_scale']]).to_csv(
        OUT/'complexity.csv',index=False)
    jobs=[];val_rows=[];decisions=[]

    def persist(stage):
        state.update(stage=stage,trained_models=sum(j['origin']=='V2' for j in jobs),
            referenced_v1_models=sum(j['origin']=='V1' for j in jobs),last_update=datetime.now().astimezone().isoformat())
        R.write_json(OUT/'config.json',state)
        if jobs:pd.DataFrame(jobs).to_csv(OUT/'models_registry.csv',index=False)
        if val_rows:pd.DataFrame(val_rows).to_csv(OUT/'validation_metrics.csv',index=False)

    def run_case(variant,ratio,subset,protocol,seed,domain='window_random'):
        split_path=(BASE/'splits'/domain if domain!='window_random' else
            BASE/'splits/window_random' if ratio==1 else OUT/'subsets'/f'ratio_{int(ratio*100):03d}_subset_{subset}')
        w=read_fixed(split_path)
        # V1 loader requires all roles represented; unused_train保留只用于数据来源追踪。
        folder=OUT/'models'/domain/variant/f'ratio_{int(ratio*100):03d}_subset_{subset}'/protocol/f'seed_{seed}'
        info=train(x,w,protocol,variant,seed,subset,folder,device)
        info.update(domain=domain,ratio=ratio,origin='V2',folder=str(folder),split_path=str(split_path))
        R.write_json(folder/'parameters.json',info)
        model=load(folder,device)
        meta={k:info[k] for k in ['domain','variant','ratio','subset','protocol','seed']}
        rows,arrays=validation(model,w,x,device,meta)
        val_rows.extend(rows)
        np.savez_compressed(folder/'validation_predictions.npz',**arrays)
        jobs.append(info);del model
        persist(state['stage'])
        print('MODEL COMPLETE',len(jobs),domain,variant,ratio,subset,protocol,seed,flush=True)

    try:
        # 固定全样本基线只引用，不重新训练或测试。
        for method,variant in [('fusion','baseline_fusion'),('data_only','baseline_data')]:
            for protocol in PROTOCOLS:
                for seed in range(1706,1711):
                    folder=BASE/'models/window_random'/protocol/method/f'seed_{seed}'
                    info=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
                    info.update(domain='window_random',variant=variant,ratio=1.,subset=0,origin='V1',
                        folder=str(folder),split_path=str(BASE/'splits/window_random'))
                    jobs.append(info)
                    diag=pd.read_csv(OUT/'diagnosis/baseline_validation.csv')
                    match=diag[diag.method.eq(method)&diag.protocol.eq(protocol)&diag.seed.eq(seed)]
                    for row in match.to_dict('records'):
                        row.pop('method');row.update(domain='window_random',variant=variant,ratio=1.,subset=0)
                        val_rows.append(row)
        persist('small_sample_baseline_training')
        for ratio in RATIOS:
            for subset in SUBSETS:
                for protocol in PROTOCOLS:
                    for variant in ['baseline_fusion','baseline_data']:
                        for seed in SEEDS:run_case(variant,ratio,subset,protocol,seed)
        persist('small_sample_baseline_complete')
        # 先完成基线再启动候选。初筛仅10/20%，不通过不扩大训练规模。
        small=pd.DataFrame(val_rows)
        small=small[small.ratio.lt(1)]
        (OUT/'SMALL_SAMPLE_BASELINE.md').write_text('# 小样本基线验证结果\n\n'
            '108模型已完成来源验证选模，测试尚未读取。以下只用于诊断及后续候选筛选。\n\n'+
            small.groupby(['ratio','protocol','variant','condition'])[['balanced_acc','validation_nll']].mean().reset_index().round(6).to_markdown(index=False),
            encoding='utf-8')
        selected=None
        for variant in state['candidate_order']:
            persist(f'{variant}_screen_training')
            for ratio in [.1,.2]:
                for subset in SUBSETS:
                    for protocol in PROTOCOLS:
                        for seed in SEEDS:run_case(variant,ratio,subset,protocol,seed)
            decision=select_candidate(pd.DataFrame(val_rows),variant,[.1,.2])
            decision['stage']='screen';decisions.append(decision)
            R.write_json(OUT/f'{variant}_screen_decision.json',decision)
            print('CANDIDATE SCREEN',decision,flush=True)
            if not decision['passed']:continue
            persist(f'{variant}_confirmation_training')
            for subset in SUBSETS:
                for protocol in PROTOCOLS:
                    for seed in SEEDS:run_case(variant,.5,subset,protocol,seed)
            for protocol in PROTOCOLS:
                for seed in range(1706,1711):run_case(variant,1.,0,protocol,seed)
            half=select_candidate(pd.DataFrame(val_rows),variant,[.5])
            full_decision=select_candidate(pd.DataFrame(val_rows),variant,[1.],full=True)
            half['stage']='half_sample';full_decision['stage']='full_sample'
            decisions.extend([half,full_decision])
            R.write_json(OUT/f'{variant}_confirmation_decision.json',dict(half=half,full=full_decision))
            if half['passed'] and full_decision['passed']:
                selected=variant;break
        state.update(selected_candidate=selected,decisions_locked=True)
        R.write_json(OUT/'STAGE_DECISIONS.json',dict(selected_candidate=selected,decisions=decisions,
            reason='候选通过全部基础验证门槛，进入跨负载' if selected else '候选未通过全部稳定性门槛，停止跨负载扩展；不放宽门槛'))
        if selected:
            persist('load_transfer_training')
            for source in range(4):
                for protocol in PROTOCOLS:
                    for seed in SEEDS:run_case(selected,1.,0,protocol,seed,f'load_source_{source}')
            # 完整跨负载参照仅引用同种子V1结果，不重复推理。
            for source in range(4):
                for method,variant in [('fusion','baseline_fusion'),('data_only','baseline_data')]:
                    for protocol in PROTOCOLS:
                        for seed in SEEDS:
                            folder=BASE/f'models/load_source_{source}'/protocol/method/f'seed_{seed}'
                            info=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
                            info.update(domain=f'load_source_{source}',variant=variant,ratio=1.,subset=0,origin='V1',
                                folder=str(folder),split_path=str(BASE/f'splits/load_source_{source}'))
                            jobs.append(info)
        persist('all_models_selected_decisions_locked')
        # 所有决定已经锁定，包括不通过候选；此时才打开目标测试。
        metrics=[];evaluations=[]
        state['target_test_accessed']=True
        persist('testing')
        for info in jobs:
            folder=resolve_result_path(info['folder']);w=read_fixed(resolve_result_path(info['split_path']))
            source=-1 if info['domain']=='window_random' else int(info['domain'][-1])
            targets=[-1] if source==-1 else [t for t in range(4) if t!=source]
            model=load(folder,device) if info['origin']=='V2' else None
            latency_ms=info.get('latency_ms')
            for target in targets:
                ix=np.flatnonzero(w.role.eq('test') if target==-1 else w.load_hp.eq(target))
                labels=w.iloc[ix].label.to_numpy();rpm=w.iloc[ix].rpm.to_numpy(np.float32)
                meta={k:info[k] for k in ['domain','variant','ratio','subset','protocol','seed','origin']}
                meta.update(source_load_hp=source,target_load_hp=target)
                tag='test_random' if target==-1 else f'test_hp{target}'
                if info['origin']=='V1':
                    with np.load(folder/tag/'predictions.npz') as saved:
                        np.testing.assert_array_equal(saved['window_indices'],w.iloc[ix].window_index)
                        np.testing.assert_array_equal(saved['labels'],labels)
                        for key in saved.files:
                            if '_n' not in key:continue
                            condition,noise_seed=key.rsplit('_n',1)
                            values,_=scores(labels,saved[key])
                            metrics.append(dict(**meta,condition=condition,noise_seed=int(noise_seed),**values))
                else:
                    destination=folder/tag;destination.mkdir(exist_ok=False)
                    arrays={'labels':labels,'window_indices':w.iloc[ix].window_index.to_numpy()}
                    for condition,noise_seed,observed in R.conditions(x[ix],info['protocol'],True):
                        p=R.predict(model,observed,rpm,device)
                        arrays[f'{condition}_n{noise_seed}']=p
                        values,_=scores(labels,p)
                        metrics.append(dict(**meta,condition=condition,noise_seed=noise_seed,**values))
                    np.savez_compressed(destination/'predictions.npz',**arrays)
                    R.write_json(destination/'evaluation.json',meta)
                    if latency_ms is None:latency_ms=R.latency(model,raw[ix],rpm,device)
                evaluations.append(dict(**meta,test_windows=len(ix),latency_ms=latency_ms))
                state['evaluated_pairs']=len(evaluations)
                R.write_json(OUT/'config.json',state)
            if info['origin']=='V2':
                info['latency_ms']=latency_ms;R.write_json(folder/'parameters.json',info)
            del model
            print('TESTED MODEL',len(evaluations),info['variant'],info['ratio'],info['seed'],flush=True)
        pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
        pd.DataFrame(evaluations).to_csv(OUT/'evaluations.csv',index=False)
        persist('reporting')
        from final_report import report
        report(OUT)
        state.update(stage='complete',completed_at=datetime.now().astimezone().isoformat(),metric_rows=len(metrics))
        R.write_json(OUT/'config.json',state)
        print('WORKFLOW COMPLETE',OUT,flush=True)
    except BaseException as error:
        state.update(stage='failed',failure_type=type(error).__name__,failure_message=str(error))
        R.write_json(OUT/'config.json',state)
        raise


if __name__=='__main__':
    main()
