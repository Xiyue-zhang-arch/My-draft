"""从冻结预测复算并汇总 V3 跨负载方法对比（纯干净，小样本或全样本）；不训练、不改参考实验。"""
from pathlib import Path
import json
import torch  # Windows 环境须先于 numpy/pandas 导入。
import numpy as np
import pandas as pd

from context import PROJECT
from metrics import METRICS

REFERENCE = PROJECT/'experiments/CWRU/实验结果/CWRU_V3'
SMALL = REFERENCE/'load_small_20261025_2017'
FULL = REFERENCE/'load_full_20261024_0950'
CASE = ['domain','ratio','subset','protocol','variant','seed','source_load_hp','target_load_hp']
PAIR = [k for k in CASE if k != 'variant']
METHODS = ['DWCN_SDGFD','ClassBD_WDCNN']
REFERENCE_METHODS = ['baseline_data','data_aux_ce']
PROTOCOL = 'clean_only'
CONDITION = ('clean',0)
LABELS = {'DWCN_SDGFD':'DWCN-SDGFD','ClassBD_WDCNN':'ClassBD-WDCNN','baseline_data':'数据单分支','data_aux_ce':'本文融合加辅助分类',
          'WDCNN':'WDCNN','DRSN_CW':'DRSN-CW','PG_RDGFN_adapted':'PG-RDGFN-adapted',
          'AW_Hybrid_adapted':'AW_Hybrid-adapted'}


def _counts(labels, probability):
    p=np.asarray(probability)
    if p.shape != (len(labels),4) or not np.isfinite(p).all():
        raise ValueError('预测维度或数值错误')
    if not np.allclose(p.sum(1),1,atol=1e-5) or (p < -1e-7).any():
        raise ValueError('预测概率不合法')
    cm=np.bincount(np.asarray(labels,dtype=int)*4+p.argmax(1),minlength=16).reshape(4,4)
    support=cm.sum(1)
    if (support==0).any(): raise ValueError('目标缺失类别')
    rec=np.diag(cm)/support
    denom=cm.sum(0)+support
    f1=np.divide(2*np.diag(cm),denom,out=np.zeros(4,dtype=float),where=denom!=0)
    return np.r_[np.trace(cm)/cm.sum(),rec.mean(),f1.mean(),rec]


def _validate_splits(root):
    """核对保存的 40 个固定负载划分与原实验逐行一致。"""
    original=REFERENCE/'run_20261022_2105'/'splits'
    names=[]
    for source in range(4):
        for ratio in [1.,.1,.2,.5]:
            for subset in ([0] if ratio==1. else [4101,4102,4103]):
                name=f'load_source_{source}_r{round(ratio*100):03d}_s{subset}'
                a=pd.read_csv(original/name/'windows.csv')
                b=pd.read_csv(root/'splits'/name/'windows.csv')
                pd.testing.assert_frame_equal(a,b)
                names.append(name)
    return len(names)


def _verify(root, ratios, expected_models, expected_variants, split_root):
    """逐预测校验窗口、标签和保存指标；只保留纯干净且指定比例的行。"""
    folder=root/'evaluation'
    m=pd.read_csv(folder/'metrics.csv')
    e=pd.read_csv(folder/'evaluations.csv')
    m=m[m.protocol.eq(PROTOCOL) & m.ratio.isin(list(ratios))].reset_index(drop=True)
    e=e[e.protocol.eq(PROTOCOL) & e.ratio.isin(list(ratios))].reset_index(drop=True)
    expected_pairs=expected_models*3
    if len(m)!=expected_pairs or len(e)!=expected_pairs:
        raise ValueError(f'{root} 测试指标或目标组合数不符')
    if set(e.variant)!=set(expected_variants) or set(m.variant)!=set(expected_variants):
        raise ValueError('方法集合不符')
    if m.duplicated(CASE+['condition','noise_seed']).any() or e.duplicated(CASE).any():
        raise ValueError('模型目标组合或条件重复')
    mgroups={k:g for k,g in m.groupby(CASE,sort=False,dropna=False)}
    windows={}
    for row in e.itertuples(index=False):
        key=tuple(getattr(row,k) for k in CASE)
        group=mgroups.pop(key,None)
        if group is None:raise ValueError('预测缺少对应指标')
        if set(zip(group.condition,group.noise_seed))!={CONDITION} or len(group)!=1:
            raise ValueError('纯干净测试条件不完整')
        name=f'{row.domain}_r{round(row.ratio*100):03d}_s{row.subset}'
        if name not in windows: windows[name]=pd.read_csv(split_root/name/'windows.csv')
        w=windows[name]
        target=w[w.load_hp==row.target_load_hp]
        if row.target_load_hp==row.source_load_hp or target.empty:raise ValueError('目标负载错误')
        path=folder/'predictions'/f'{row.job_id}_target{row.target_load_hp}.npz'
        with np.load(path,allow_pickle=False) as pred:
            np.testing.assert_array_equal(pred['labels'],target.label.to_numpy())
            np.testing.assert_array_equal(pred['window_indices'],target.window_index.to_numpy())
            if set(pred.files)!={'labels','window_indices',f'{CONDITION[0]}_n{CONDITION[1]}'}:
                raise ValueError(f'预测条件不匹配: {path}')
            item=next(group.itertuples(index=False))
            actual=_counts(pred['labels'],pred[f'{item.condition}_n{item.noise_seed}'])
            recorded=np.array([getattr(item,k) for k in METRICS],dtype=float)
            if not np.allclose(actual,recorded,atol=1e-10,rtol=1e-10):
                raise ValueError(f'指标与预测不一致: {path}')
    if mgroups:raise ValueError('存在没有预测的指标')
    return m


def _aggregate(frame, keys):
    out=frame.groupby(keys,dropna=False)[METRICS].agg(['mean','std','count'])
    out.columns=['_'.join(i) for i in out.columns]
    return out.reset_index()


def _cell(row, metric):
    return f"{100*float(row[f'{metric}_mean']):.2f}±{100*float(row[f'{metric}_std']):.2f}"


def _performance_table(frame, metric, order, title, show_subset):
    head='### '+title+'\n\n'
    if show_subset:
        head+='| 训练比例 | 子集 | ' + ' | '.join(LABELS[v] for v in order) + ' |\n'
        head+='|:--|--:|' + '--:|'*len(order)+'\n'
    else:
        head+='| 训练比例 | ' + ' | '.join(LABELS[v] for v in order) + ' |\n'
        head+='|:--|' + '--:|'*len(order)+'\n'
    lines=[head.rstrip('\n')]
    keys=['ratio','subset'] if show_subset else ['ratio']
    for raw,group in frame.groupby(keys,sort=True):
        key=raw if isinstance(raw,tuple) else (raw,)
        cells=[]
        for variant in order:
            row=group[group.variant.eq(variant)]
            if len(row)!=1: raise ValueError(f'缺少 {variant} 的 {key} 汇总')
            cells.append(_cell(row.iloc[0], metric))
        label=f'{int(round(float(key[0])*100))}%'
        if show_subset:
            lines.append(f'| {label} | {int(key[1])} | ' + ' | '.join(cells) + ' |')
        else:
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _paired_table(frame, order, metric, show_subset):
    title='### 配对差值（本文融合加辅助分类 减 对照，单位百分点）'
    if show_subset:
        head='| 训练比例 | 子集 | ' + ' | '.join(LABELS[v] for v in order) + ' |'
        sep='|:--|--:|' + '--:|'*len(order)
        keys=['ratio','subset']
    else:
        head='| 训练比例 | ' + ' | '.join(LABELS[v] for v in order) + ' |'
        sep='|:--|' + '--:|'*len(order)
        keys=['ratio']
    lines=[title,'',head,sep]
    for raw,group in frame.groupby(keys,sort=True):
        key=raw if isinstance(raw,tuple) else (raw,)
        cells=[]
        for comparator in order:
            row=group[group.comparator.eq(comparator)]
            if len(row)!=1: raise ValueError(f'缺少 {comparator} 的 {key} 配对汇总')
            cells.append(_cell(row.iloc[0], metric))
        if show_subset:
            lines.append(f'| {int(round(float(key[0])*100))}% | {int(key[1])} | ' + ' | '.join(cells) + ' |')
        else:
            lines.append(f'| {int(round(float(key[0])*100))}% | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def finish(root, scope=None):
    root=Path(root).resolve()
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    scope=scope or cfg.get('scope')
    if not scope:
        raise ValueError('缺少实验范围记录，不能汇总')
    method=scope['method']
    sample=scope.get('sample','small')
    if sample not in ('small','full'):
        raise ValueError(f'未知样本范围: {sample}')
    ratios=[1.0] if sample=='full' else list(scope['ratios'])
    subsets=[0] if sample=='full' else list(scope.get('subsets',[4101,4102,4103]))
    models=4*len(ratios)*len(subsets)*5
    pairs=models*3
    if len(cfg['jobs'])!=models or not cfg.get('decisions_locked') or not cfg.get('target_test_accessed'):
        raise ValueError('只有训练选模锁定并完成全部测试后才能汇总')
    if (root/'report').exists() or (root/'REPORT.md').exists():
        raise FileExistsError('报告已存在，禁止覆盖')
    _validate_splits(root)
    new=_verify(root,ratios,models,{method},root/'splits')
    ref_root=FULL if sample=='full' else SMALL
    old=_verify(ref_root,ratios,4*len(ratios)*len(subsets)*2*5,set(REFERENCE_METHODS),root/'splits')
    frame=pd.concat([new,old],ignore_index=True)
    if frame.duplicated(CASE+['condition','noise_seed']).any():raise ValueError('合并结果存在重复')
    variants=set(frame.variant)
    order=[v for v in ['baseline_data','data_aux_ce',*METHODS] if v in variants]
    if 'data_aux_ce' not in variants or method not in variants:
        raise ValueError('缺少本文方法或本轮对比方法的结果')
    seed=frame.groupby(CASE+['condition'],as_index=False,dropna=False)[METRICS].mean()
    direction_keys=[k for k in CASE if k!='seed']+['condition']
    direction_summary=_aggregate(seed,direction_keys)
    if not direction_summary.acc_count.eq(5).all():raise ValueError('每个迁移方向必须有五种训练种子')
    direct=seed.groupby(['ratio','subset','protocol','variant','condition','seed'],as_index=False,dropna=False).agg(
        directions=('target_load_hp','size'),**{x:(x,'mean') for x in METRICS})
    if not direct.directions.eq(12).all():raise ValueError('等权平均必须有十二个方向')
    direct=direct.drop(columns='directions')
    keys=['ratio','subset','protocol','variant','condition']
    sd=_aggregate(direct,keys)
    if not sd.acc_count.eq(5).all():raise ValueError('每配置必须有五种训练种子')
    show_subset=sample=='small'
    subset=None
    if show_subset:
        subset=sd.groupby(['ratio','protocol','variant','condition'],dropna=False)[[x+'_mean' for x in METRICS]].agg(['mean','std','count'])
        subset.columns=['_'.join(c) for c in subset.columns]
        subset=subset.reset_index()
        if not subset.acc_mean_count.eq(3).all():raise ValueError('每小样本预算必须有三个子集')
    comparators=[m for m in METHODS+['baseline_data'] if m in variants]
    pairkeys=PAIR+['condition']
    piv=seed.pivot(index=pairkeys,columns='variant',values=METRICS)
    if piv.isna().any().any():raise ValueError('方法间配对不完整')
    diff=[]
    for method_name in comparators:
        d=pd.DataFrame({x:piv[(x,'data_aux_ce')]-piv[(x,method_name)] for x in METRICS}).reset_index()
        d['comparator']=method_name
        diff.append(d)
    diff=pd.concat(diff,ignore_index=True)
    pair_direction=diff.groupby(['ratio','subset','protocol','condition','seed','comparator'],as_index=False)[METRICS].mean()
    pair_summary=_aggregate(pair_direction,['ratio','subset','protocol','condition','comparator'])
    pair_per_direction=_aggregate(diff,[k for k in PAIR if k!='seed']+['condition','comparator'])
    out=root/'report';out.mkdir(exist_ok=False)
    seed.to_csv(out/'per_model_condition.csv',index=False)
    direction_summary.to_csv(out/'per_direction_seed_mean_sd.csv',index=False)
    direct.to_csv(out/'equal_12_directions_per_seed.csv',index=False)
    sd.to_csv(out/'equal_12_directions_seed_mean_sd.csv',index=False)
    if show_subset:
        subset.to_csv(out/'between_subset_mean_sd.csv',index=False)
    diff.to_csv(out/'paired_differences_by_direction.csv',index=False)
    pair_per_direction.to_csv(out/'paired_per_direction_seed_mean_sd.csv',index=False)
    pair_summary.to_csv(out/'paired_equal_12_directions_seed_mean_sd.csv',index=False)
    costs=pd.DataFrame(cfg['jobs']).assign(cost_source='new_comparison')
    history=json.loads((ref_root/'config.json').read_text(encoding='utf-8'))
    historic=pd.DataFrame(history['jobs'])
    if len(historic):
        historic=historic[historic.protocol.eq(PROTOCOL)&historic.ratio.isin(ratios)]
        costs=pd.concat([costs,historic.assign(cost_source=ref_root.name)],ignore_index=True)
    costcols=['cost_source','domain','ratio','subset','protocol','variant','seed','training_windows','validation_windows','epochs_trained','updates_trained','best_epoch','parameters','fit_seconds','train_seconds_per_epoch','total_train_seconds','peak_gpu_mib','latency_ms','hardware','precision','batch_size','latency_scope','peak_memory_scope']
    costs.reindex(columns=costcols).to_csv(out/'comparison_cost.csv',index=False)
    budget=', '.join(f'{int(round(r*100))}%' for r in ratios)
    if sample=='full':
        scope_text=(f'{models} 个从头训练模型（4 个来源负载 × 5 个训练种子），'
                    f'每个模型用该来源的完整训练池训练、用来源验证集选模，随后统一评价另外三个负载，'
                    f'共 {pairs} 个模型—目标组合、{pairs} 组逐条件指标。')
        data_text=('来源训练池 73/156/160/160 个窗口全部使用，来源验证 19/39/40/40 个窗口；'
                   '目标负载 0/1/2/3 HP 分别为 92/195/200/200 个窗口。目标负载只用于评价，'
                   '不参与训练、标准化拟合或选模。')
        stats_text=('每格为 12 个来源—目标方向等权后的均值±五种子样本标准差（分母 4），单位百分比。'
                    '全样本不设训练子集，因此没有子集间波动统计。')
    else:
        scope_text=(f'{models} 个从头训练模型（4 个来源负载 × 预算 {budget} × 3 个固定嵌套子集 × 5 个训练种子），'
                    f'训练时只用来源训练子集，选模只用来源验证集，随后统一评价另外三个负载，'
                    f'共 {pairs} 个模型—目标组合、{pairs} 组逐条件指标。')
        data_text=('来源训练池 73/156/160/160 个窗口，预算按来源训练池逐类向上取整抽取，'
                   '10%/20%/50% 分别为 9/16/37、17/33/78、17/34/80、17/34/80；'
                   '来源验证集 19/39/40/40 个窗口不随预算缩减。目标负载 0/1/2/3 HP 分别为 92/195/200/200 个窗口。'
                   '0 HP 的 10% 预算正常类只有 1 个训练窗口。')
        stats_text=('每格为 12 个来源—目标方向等权后的均值±五种子样本标准差（分母 4），单位百分比。'
                    '同一配置下三个子集各自汇总，`report/between_subset_mean_sd.csv` 单列三个子集均值之间的波动。')
    with (root/'REPORT.md').open('x',encoding='utf-8') as f:
        f.write(f'# V3 跨负载{"全样本" if sample=="full" else "小样本"}对比方法正式报告\n\n'
          f'本轮为 {LABELS.get(method,method)} 的跨负载{"全样本" if sample=="full" else "小样本"}纯干净实验：'
          + scope_text + '\n\n'
          '数据划分、训练种子和评价窗口与本文方法（V3 跨负载实验）完全一致；'
          '本文的数据单分支和融合加辅助分类结果从原冻结目录只读复用，未重训、未重测。'
          '因此下表可以直接对照，但仍是同一固定划分下的探索性比较。\n\n'
          '## 数据与训练\n\n' + data_text + '\n\n'
          '作者代码版本及具体适配变更见 `METHOD_ADAPTATION.md`；模型训练目标和设置见训练日志与各模型训练记录。'
          '不新增信号噪声，逐模型仅使用固定来源训练和验证窗口；方法内部的特征扰动须参照模型说明。\n\n'
          '## 统计口径\n\n' + stats_text + '\n'
          '训练种子只描述训练随机性；12 个方向复用同一来源模型和目标数据，不是独立数据集重复。'
          '配对差值按来源、目标、子集（若有）和种子配对，正值表示本文融合加辅助分类较高。\n\n'
          '## 结果\n\n'
          + _performance_table(sd,'balanced_acc',order,'均衡准确率 BA（%）',show_subset) + '\n\n'
          + _performance_table(sd,'acc',order,'窗口准确率 ACC（%）',show_subset) + '\n\n'
          + _paired_table(pair_summary,comparators,'balanced_acc',show_subset) + '\n\n'
          '逐方向结果见 `report/per_direction_seed_mean_sd.csv`，逐方向配对差值见 '
          '`report/paired_per_direction_seed_mean_sd.csv`；全部负差值都保留，不因总体均值较高而省略局部退化。\n\n'
          '## 计算成本\n\n'
          '`report/comparison_cost.csv` 保存参数量、单样本推理延迟、每 epoch 训练时间、总训练时间、'
          '峰值显存、硬件与精度（未记录者为空），并区分本轮新模型与原 V3 记录来源。'
          '延迟测量口径以模型记录为准。不同方法的精度、输入长度和额外信息不同，'
          '成本数值不能直接当作同条件效率结论。\n\n'
          '## 解释边界\n\n'
          '新模型基于作者公开实现接入，DWCN 来源验证使用学生分支，采用 Adam；ClassBD 使用分段概率汇总、修复物理项断梯度并采用 AdamW。'
          '两方法输入处理、优化器和原模型适配不同，详见 `METHOD_ADAPTATION.md`，不宣称完全复现原论文。'
          '输入、训练目标和优化规则可能存在差异，不宜将结果归因于某个模块。\n\n'
          '训练与验证窗口共享来源记录，不构成独立轴承验证；来源训练、验证预算跨来源不同，'
          'λ 与划分来自既有全负载开发，历史测试已经曝光。因此本轮是固定公开数据资源上的探索性跨负载比较，'
          '不支持严格未见目标域、独立设备泛化或确认性结论。本节报告的是单一方法的一轮实验，'
          '跨方法比较须在所需结果完整后再统一汇总。\n')
