"""核对全部冻结概率，按迁移方向及种子汇总，保留所有负增益。"""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from experiment_paths import validate_output_path

CONDITIONS=[('clean','Clean'),('white_-4','-4 dB'),('white_-2','-2 dB'),('white_+0','0 dB'),('white_+2','+2 dB'),('white_+4','+4 dB')]
METRICS=['acc','balanced_acc','f1_macro','recall_N','recall_IR','recall_OR','recall_B']
KEYS=['source_load_hp','target_load_hp','protocol','method','ratio','subset','seed','condition','noise_seed']
LABELS=['N','IR','OR','B']


def table(frame):
    return ['|'+'|'.join(map(str,frame.columns))+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+['|'+'|'.join(map(str,row))+'|' for row in frame.itertuples(index=False,name=None)]


def write_usage(out):
    records=pd.read_csv(out/'records.csv');files=[];subsets=[];sources=[]
    assert len(records)==40
    for source in range(4):
        w=pd.read_csv(out/f'source_hp{source}'/'windows.csv')
        canonical=w[w.role.eq('train')].reset_index(drop=True)
        paths=list((out/f'source_hp{source}').glob('*/*/ratio_100_subset_0_seed_*/selected_windows.csv'))
        assert len(paths)==20
        for path in paths:pd.testing.assert_frame_equal(pd.read_csv(path),canonical,check_dtype=False)
        sources.append(dict(source_HP=source,candidate_train=int(w.candidate_role.eq('train').sum()),
            selected_train=int(w.role.eq('train').sum()),validation=int(w.role.eq('validation').sum()),
            unused_source_train=int(w.role.eq('unused_source_train').sum()),
            train_records=int(w.loc[w.role.eq('train'),'record_id'].nunique()),validation_records=int(w.loc[w.role.eq('validation'),'record_id'].nunique())))
        for r in records.itertuples():
            g=w[w.record_id.eq(r.record_id)]
            subsets.append(dict(source_load_hp=source,record_id=r.record_id,raw_file=Path(r.path).name,
                raw_path=r.path,DE_variable=r.variable,candidate_training_windows=int(g.candidate_role.eq('train').sum()),
                actual_training_windows=int(g.role.eq('train').sum()),validation_windows=int(g.role.eq('validation').sum()),
                unused_source_training_windows=int(g.role.eq('unused_source_train').sum()),model_count_sharing_selection=20,
                representative_selected_windows=paths[0].relative_to(out).as_posix()))
        for target in range(4):
            if target==source:continue
            tw=pd.read_csv(out/f'source_hp{source}'/f'test_hp{target}'/'windows.csv')
            for r in records.itertuples():
                g=tw[tw.record_id.eq(r.record_id)]
                assert len(g)==r.samples//24000
                files.append(dict(source_load_hp=source,target_load_hp=target,record_id=r.record_id,
                    raw_file=Path(r.path).name,raw_path=r.path,DE_variable=r.variable,label=r.label,
                    load_hp=int(r.record_id.rsplit('_',1)[1]),raw_samples=r.samples,complete_windows=len(g),
                    discarded_tail_samples=int(r.samples%24000),candidate_training_windows=int(g.candidate_role.eq('train').sum()),
                    training_windows=int(g.role.eq('train').sum()),validation_windows=int(g.role.eq('validation').sum()),
                    test_windows=int(g.role.eq('test').sum()),unused_windows=int(g.role.str.startswith('unused').sum()),
                    excluded_windows=0))
    assert len(files)==480 and len(subsets)==160
    f=pd.DataFrame(files);f.to_csv(out/'dataset_file_usage.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(subsets).to_csv(out/'training_subset_file_usage.csv',index=False,encoding='utf-8-sig')
    support=f.groupby(['source_load_hp','target_load_hp','label'])[['training_windows','validation_windows','test_windows']].sum().reset_index()
    support.to_csv(out/'direction_class_support.csv',index=False)
    pd.DataFrame(sources).to_csv(out/'source_budgets.csv',index=False)
    lines=['# 单一来源迁移的数据使用情况','','全部40个MAT文件均纳入本实验，687个完整窗口。没有排除OR014_3、OR014_1或其他记录。每个原始文件均在相应目标负载中完整测试。各方向只用一个来源和一个目标，其他负载不用，不将方向内未用记录称为全局排除。','',
        '原始文件只读，48kHz指定DE通道和转速，每窗24000点、0.5秒，无重叠，不足一窗的尾部按统一规则不用。','',*table(pd.DataFrame(sources)),'',
        '## 每方向的集合数量','',*table(pd.read_csv(out/'directions.csv')),'',
        '[逐方向逐MAT用量、原始路径与DE变量](dataset_file_usage.csv) · [逐来源训练候选及实际子集用量](training_subset_file_usage.csv) · [逐方向逐类数量](direction_class_support.csv)','',
        '每个来源的windows.csv保留全部687窗的训练、验证、未用角色及候选角色。source_hpS/test_hpT/windows.csv给出具体方向的测试角色。模型selected_windows.csv给出实际73训练窗的编号、记录及采样区间，20个模型共用一次抽取。','',
        '固定划分seed20261019，来源记录内部80%四舍五入，保证至少1训练和1验证。IR014_0仅2窗，来源时1训练1验证，目标时2窗均测试。随后固定抽取seed5101按正常8、内圈17、外圈24、滚动体24取共同73窗。未入选候选不用，不并入验证。','',
        '每来源训练验证共享全部10个MAT文件，无采样点重叠。每个方向的目标10个MAT与来源完全分离。训练验证共享记录仍可能使验证偏向同记录表现，记录留出不等于独立轴承留出。','',
        '各来源验证数量19/39/40/40，不声称验证标签预算跨来源匹配。每来源73训练窗外仍有额外验证标注预算。没有小样本实验，不以73窗训练加验证包装为只有73个标注窗口。','',
        '同来源同一个模型跨三个目标复用。各目标数据会被其他来源重复测试，40文件687窗不能乘以方向数、模型数、种子数或噪声重复数作为独立样本。历史测试曝光仍存在。','',
        '## 每个原始文件的记录与窗口','',*table(records[['record_id','variable','samples','rpm']])]
    (out/'DATA_USAGE.md').write_text('\n'.join(lines),encoding='utf-8')


def report(directory):
    out=validate_output_path(directory,existing=True)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    training_description = (
        f"普通交叉熵，标签平滑0；最多{cfg['max_epochs']}个epoch，每个epoch结束验证并选模，"
        f"连续{cfg['early_stopping_patience']}个epoch未改善则早停，最小改善量{cfg['selection_min_delta']}。"
        '余弦学习率按epoch推进。不同模型实际更新数和选模次数可能不同，不声称实际计算预算一致。'
        if 'max_epochs' in cfg else '固定600更新、每步32观测、每50验证、无提前停止，标签平滑0.05。'
    )
    budget_description = (
        f"训练batch32，最多{cfg['max_epochs']}个epoch，早停耐心{cfg['early_stopping_patience']}个epoch；"
        '实际epochs_trained、updates_trained及best_epoch见每模型parameters.json。'
        if 'max_epochs' in cfg else '训练batch32，600更新，共19200有放回观测；'
    )
    complexity=pd.read_csv(out/'complexity.csv');evaluations=pd.read_csv(out/'evaluations.csv')
    saved=pd.read_csv(out/'metrics.csv')
    assert len(complexity)==cfg['planned_models']==80 and len(evaluations)==240
    pooled={};details=[];supports={}
    for row in evaluations.to_dict('records'):
        s,t=row['source_load_hp'],row['target_load_hp']
        w=pd.read_csv(out/f'source_hp{s}'/f'test_hp{t}'/'windows.csv');test=w[w.role.eq('test')]
        support=np.bincount(test.label,minlength=4);supports[(s,t)]=support
        folder=out/f'source_hp{s}'/row['protocol']/row['method']/f"ratio_100_subset_0_seed_{row['seed']}"/f'test_hp{t}'
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['window_indices'],test.window_index)
            np.testing.assert_array_equal(z['labels'],test.label)
            for condition,_ in (CONDITIONS[:1] if row['protocol']=='clean_only' else CONDITIONS):
                counted=np.zeros((4,4),dtype=np.int64)
                keys=[k for k in z.files if k.startswith(condition+'_n')]
                assert len(keys)==(1 if condition=='clean' else 5)
                for key in keys:
                    p=z[key];y=z['labels'];pred=p.argmax(1)
                    assert p.shape==(len(test),4) and np.isfinite(p).all()
                    np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
                    counts=confusion_matrix(y,pred,labels=range(4));recalls=counts.diagonal()/support
                    details.append({k:row[k] for k in KEYS[:7]}|dict(condition=condition,noise_seed=int(key.rsplit('_n',1)[1]),
                        acc=float(np.trace(counts)/len(y)),balanced_acc=float(recalls.mean()),
                        f1_macro=f1_score(y,pred,labels=range(4),average='macro',zero_division=0),**dict(zip(METRICS[3:],recalls))))
                    counted+=counts
                np.testing.assert_array_equal(counted,z[condition+'_confusion_counts'])
                group=(s,t,row['protocol'],row['method'],condition)
                pooled[group]=pooled.get(group,np.zeros((4,4),dtype=np.int64))+counted
    detailed=pd.DataFrame(details)
    checked=detailed.merge(saved,on=KEYS,suffixes=('_check','_saved'),validate='one_to_one')
    assert len(checked)==len(detailed)==len(saved)==3240
    for metric in ['acc','f1_macro']:np.testing.assert_allclose(checked[metric+'_check'],checked[metric+'_saved'],atol=1e-12,rtol=0)
    detailed.to_csv(out/'class_metrics.csv',index=False)
    per_model=detailed.groupby(KEYS[:-1],as_index=False)[METRICS].mean()
    per_model.to_csv(out/'per_seed_direction.csv',index=False)
    group=['source_load_hp','target_load_hp','protocol','method','ratio','condition']
    by_direction=per_model.groupby(group)[METRICS].agg(['mean','std','count']).reset_index()
    by_direction.columns=['_'.join(k for k in col if k) for col in by_direction.columns]
    assert by_direction.acc_count.eq(5).all()
    by_direction.to_csv(out/'per_direction_summary.csv',index=False)
    means=per_model.groupby(group,as_index=False)[METRICS].mean()
    overall=means.groupby(['protocol','method','ratio','condition'])[METRICS].agg(['mean','std','min','max','count']).reset_index()
    overall.columns=['_'.join(k for k in col if k) for col in overall.columns]
    assert overall.acc_count.eq(12).all();overall.to_csv(out/'summary.csv',index=False)
    paired=per_model.pivot(index=['source_load_hp','target_load_hp','protocol','ratio','subset','seed','condition'],columns='method',values=METRICS)
    delta=pd.DataFrame({m+'_delta':paired[(m,'fusion')]-paired[(m,'data_only')] for m in METRICS}).reset_index()
    delta.to_csv(out/'paired_seed_differences.csv',index=False)
    direction_delta=delta.groupby(['source_load_hp','target_load_hp','protocol','ratio','condition'],as_index=False)[[m+'_delta' for m in METRICS]].mean()
    direction_delta.to_csv(out/'paired_direction_differences.csv',index=False)
    np.savez_compressed(out/'confusion_counts.npz',**{f'source{s}_test{t}_{p}_{m}_{c}':v for (s,t,p,m,c),v in pooled.items()})
    cs=complexity.groupby(['protocol','method'],as_index=False)[['parameters','latency_ms','train_seconds_per_epoch','total_train_seconds','peak_gpu_mib','fit_seconds','validation_forward_seconds']].mean()
    cs.to_csv(out/'complexity_summary.csv',index=False)
    assert evaluations.latency_independently_measured_here.sum()==80
    write_usage(out)
    figures=out/'figures';figures.mkdir();images=[]
    for protocol,method in complexity[['protocol','method']].drop_duplicates().itertuples(index=False,name=None):
        for condition,title in (CONDITIONS[:1] if protocol=='clean_only' else CONDITIONS):
            fig,axes=plt.subplots(4,3,figsize=(13,15))
            for (s,t),ax in zip(sorted(supports),axes.flat):
                counts=pooled[(s,t,protocol,method,condition)];fraction=counts/counts.sum(1,keepdims=True)
                ax.imshow(100*fraction,vmin=0,vmax=100,cmap='Blues')
                for i in range(4):
                    for j in range(4):ax.text(j,i,f'{100*fraction[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if fraction[i,j]>.55 else 'black')
                ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=[f'{lab} ({n})' for lab,n in zip(LABELS,supports[(s,t)])],xlabel='Predicted',ylabel='True',title=f'{s} to {t} HP; ACC {100*np.trace(counts)/counts.sum():.2f}%')
            fig.suptitle(f'Single source, train73, all records; {protocol}; {method}; {title}')
            fig.tight_layout(rect=[0,.025,1,.97]);fig.text(.5,.008,'Rows sum to 100%; supports are original windows; seed/noise repeats are not new samples',ha='center',fontsize=9)
            name=f'confusion_{protocol}_{method}_{condition}.png';fig.savefig(figures/name,dpi=140);plt.close(fig);images.append(name)
    for protocol in ['clean_only','awgn_mixed']:
        for condition,title in (CONDITIONS[:1] if protocol=='clean_only' else CONDITIONS):
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            matrices=[]
            for method in ['fusion','data_only']:
                g=by_direction[(by_direction.protocol==protocol)&(by_direction.condition==condition)&(by_direction.method==method)]
                matrices.append(g.pivot(index='source_load_hp',columns='target_load_hp',values='balanced_acc_mean').reindex(index=range(4),columns=range(4)).to_numpy()*100)
            for ax,m,title2 in zip(axes,[*matrices,matrices[0]-matrices[1]],['fusion BA (%)','data_only BA (%)','fusion minus data (pp)']):
                limit=max(1,float(np.nanmax(np.abs(m))))
                ax.imshow(np.ma.masked_invalid(m),cmap='RdBu_r' if 'minus' in title2 else 'Blues',vmin=-limit if 'minus' in title2 else 0,vmax=limit if 'minus' in title2 else 100)
                for s in range(4):
                    for t in range(4):
                        if np.isfinite(m[s,t]):ax.text(t,s,f'{m[s,t]:.2f}',ha='center',va='center',fontsize=9,color='white' if ('minus' not in title2 and m[s,t]>55) or ('minus' in title2 and abs(m[s,t])>.7*limit) else 'black')
                ax.set(xticks=range(4),yticks=range(4),xlabel='Target HP',ylabel='Source HP',title=title2)
            fig.suptitle(f'{protocol}; {title}; single source, equal train73, all records');fig.tight_layout()
            name=f'direction_{protocol}_{condition}.png';fig.savefig(figures/name,dpi=170);plt.close(fig);images.insert(0,name)
    lines=['# 单一来源负载迁移，共同73窗，全部40记录','',f'完成80个随机初始化模型，240组模型与目标评价，3240组逐次指标，{len(pooled)}组方向混淆计数。未排除任何原始记录，未开展小样本实验。','',
        '## 设计与公平性','','每次仅一个来源负载参与训练和验证，目标完整负载只在80模型全部完成选模后测试。12个方向。同一来源权重直接测试另外三个目标，不按目标重新训练或选模。来源内部记录随机80/20后抽取共同73训练窗，类别配额8/17/24/24。验证保留19/39/40/40，跨来源验证预算未匹配，不能声称全部条件完全匹配。','',*table(pd.read_csv(out/'source_budgets.csv')),'',
        '两个模型、两协议、5种子1706至1710共享相同来源73窗。'+training_description+'标准化仅拟合当前训练集，物理特征在线提取。网络参数、优化器、抽样及固定选模规则见[协议](PROTOCOL.md)。','',
        'clean_only不人工加噪，只评价干净目标。awgn_mixed每次出现独立50%干净，否则AWGN SNR连续U[-4,4]dB，单批一次CE。验证用来源类别/记录均衡NLL，干净权重0.5、各噪声档0.1。AWGN测试五噪声档各5实例，各模型共享目标观测。','',
        '## 12方向等权汇总','','先平均每模型噪声实例，再平均每方向5种子，最后12方向等权平均。主指标均衡准确率，保留窗口ACC和宏F1。下表SD为方向均值之间的样本标准差，不是独立轴承置信区间。','',
        '|协议|模型|条件|窗口ACC均值±方向SD (%)|均衡ACC (%)|宏F1 (%)|','|---|---|---|---:|---:|---:|']
    for r in overall.itertuples():lines.append(f'|{r.protocol}|{r.method}|{r.condition}|{100*r.acc_mean:.2f} ± {100*r.acc_std:.2f}|{100*r.balanced_acc_mean:.2f}|{100*r.f1_macro_mean:.2f}|')
    lines+=['','## 各方向干净及强噪声结果','','|来源HP|目标HP|协议|模型|条件|ACC均值±种子SD (%)|均衡ACC (%)|','|---:|---:|---|---|---|---:|---:|']
    for r in by_direction[by_direction.condition.isin(['clean','white_-4'])].itertuples():lines.append(f'|{r.source_load_hp}|{r.target_load_hp}|{r.protocol}|{r.method}|{r.condition}|{100*r.acc_mean:.2f} ± {100*r.acc_std:.2f}|{100*r.balanced_acc_mean:.2f}|')
    negative=direction_delta[direction_delta.balanced_acc_delta.lt(-1e-12)].copy()
    lines+=['','## 融合较低的条件','','下表列出均衡准确率的负增益，单位为百分点。完整窗口ACC差异和逐种子差异分别保存在paired_direction_differences.csv和paired_seed_differences.csv，不筛除不利方向。','']
    if len(negative):
        negative=negative[['source_load_hp','target_load_hp','protocol','condition','acc_delta','balanced_acc_delta']]
        negative[['acc_delta','balanced_acc_delta']]*=100
        lines+=table(negative.round(4))
    else:lines+=['本次方向平均值中未出现均衡准确率负增益；这不等于所有训练种子均占优。']
    lines+=['','## 计算开销','',f"硬件{cfg['hardware']}，PyTorch {cfg['torch']}，CPU线程{cfg['cpu_threads']}，精度{cfg['precision']}。",'',
        budget_description+'73窗的一抽样轮为96观测、3步。训练总/轮均时间、验证前向及标准化拟合分别记录。推理batch1、10预热50重复，CUDA同步，含归一化、在线物理提取与网络，排除磁盘和H2D。每来源模型在首个目标计时一次，共80次独立计时，其他目标引用该计时，不称逐目标单独测量。峰值为allocated GPU显存。','',*table(cs.round(4)),'',
        '## 证据边界','','本实验是人工故障CWRU单一来源负载到另一负载的评价，不是独立轴承或现场泛化证明。训练验证共享来源MAT，只有一个固定内划分和一次固定73窗抽取，5训练种子反映拟合随机性，不反映数据划分的完整不确定性。','',
        '不同方向共享来源模型、目标窗口，且不同负载可能使用相同故障试件。12方向及噪声重复不作为新增独立轴承，不据此夸大统计显著性。各来源验证标注预算和类内记录组成不同，不能将方向差异全归因于负载。','',
        '全部记录均恢复参与本次实验，但历史测试曝光不消失，本次仍属于已有数据上的后续实验。两模型对照同时改变信息、结构和参数量，不能独立证明物理因果机理。合成白噪声不能代表所有现场噪声。历史实验的训练数量、验证及停止规则不同，不用于严格等条件排名。','',
        '[数据数量与原始MAT使用说明](DATA_USAGE.md) · [逐方向完整指标](per_direction_summary.csv) · [逐种子结果](per_seed_direction.csv)','','## 图表','']
    for name in images:lines += [f'![{name}](figures/{name})','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'README.md').write_text('# 单一来源负载迁移\n\n[正式报告](REPORT.md) · [协议](PROTOCOL.md) · [数据使用情况](DATA_USAGE.md)\n\n全部40记录，80来源模型，12方向，训练73窗，无小样本。\n',encoding='utf-8')
    print(overall[['protocol','method','condition','acc_mean','balanced_acc_mean']].to_string(index=False),flush=True)
