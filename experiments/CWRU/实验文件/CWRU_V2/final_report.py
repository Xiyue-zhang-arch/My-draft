"""完成后只汇总现有预测，报告所有候选（包括失败者），不改变阶段决策。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path


METRICS=['acc','balanced_acc','f1_macro','recall_N','recall_IR','recall_OR','recall_B']
KEYS=['domain','variant','ratio','subset','protocol','seed','source_load_hp','target_load_hp','condition']


def summarize(metrics):
    per_seed=metrics.groupby(KEYS,as_index=False)[METRICS].mean()
    group=[k for k in KEYS if k!='seed']
    within=per_seed.groupby(group)[METRICS].agg(['mean','std','count'])
    within.columns=['_'.join(c) for c in within.columns]
    within=within.reset_index()
    assert within.balanced_acc_count.ge(3).all()
    within['seed_sd_over_0_5pp']=within.balanced_acc_std.gt(.005)
    keys=[k for k in group if k!='subset']
    overall=within.groupby(keys,as_index=False).agg(
        balanced_acc_mean=('balanced_acc_mean','mean'),
        subset_mean_sd=('balanced_acc_mean','std'),
        mean_within_subset_seed_sd=('balanced_acc_std','mean'),
        max_within_subset_seed_sd=('balanced_acc_std','max'),
        subset_count=('subset','nunique'),
        acc_mean=('acc_mean','mean'),recall_OR_mean=('recall_OR_mean','mean'))
    return per_seed,within,overall


def report(out):
    if (out/'REPORT.md').exists():raise FileExistsError('禁止覆盖正式报告')
    metrics=pd.read_csv(out/'metrics.csv')
    per_seed,within,overall=summarize(metrics)
    per_seed.to_csv(out/'per_seed_metrics.csv',index=False)
    within.to_csv(out/'per_subset_mean_std.csv',index=False)
    overall.to_csv(out/'hierarchical_summary.csv',index=False)
    registry=pd.read_csv(out/'models_registry.csv')
    expected_rows=int(sum((1 if r.protocol=='clean_only' else 26)*
        (1 if r.domain=='window_random' else 3) for r in registry.itertuples()))
    assert len(metrics)==expected_rows
    counts=[]
    for job in registry.itertuples():
        folder=resolve_result_path(job.folder)
        source=-1 if job.domain=='window_random' else int(job.domain[-1])
        targets=[-1] if source==-1 else [t for t in range(4) if t!=source]
        for target in targets:
            tag='test_random' if target==-1 else f'test_hp{target}'
            with np.load(folder/tag/'predictions.npz') as saved:
                labels=saved['labels']
                expected=metrics[metrics.domain.eq(job.domain)&metrics.variant.eq(job.variant)&
                    metrics.ratio.eq(job.ratio)&metrics.subset.eq(job.subset)&metrics.protocol.eq(job.protocol)&
                    metrics.seed.eq(job.seed)&metrics.target_load_hp.eq(target)]
                for r in expected.itertuples():
                    p=saved[f'{r.condition}_n{r.noise_seed}']
                    pred=p.argmax(1)
                    matrix=np.bincount(labels*4+pred,minlength=16).reshape(4,4)
                    recall=matrix.diagonal()/matrix.sum(1)
                    np.testing.assert_allclose(recall,[getattr(r,k) for k in METRICS[3:]],atol=1e-12,rtol=0)
                    assert abs(recall.mean()-r.balanced_acc)<1e-12
                    for i in range(4):
                        for j in range(4):
                            counts.append(dict(**{k:getattr(r,k) for k in KEYS},true=i,predicted=j,
                                count=int(matrix[i,j]),original_support=int((labels==i).sum())))
    counts=pd.DataFrame(counts)
    grouping=[k for k in KEYS if k!='seed']+['true','predicted']
    pooled=counts.groupby(grouping,as_index=False).agg(count=('count','sum'),original_support=('original_support','first'))
    rowkeys=[k for k in grouping if k!='predicted']
    pooled['row_fraction']=pooled['count']/pooled.groupby(rowkeys)['count'].transform('sum')
    pooled.to_csv(out/'confusion_counts.csv',index=False)
    comparisons=[]
    for variant in per_seed.variant.unique():
        if variant.startswith('baseline_'):continue
        for reference in ['baseline_fusion','baseline_data']:
            keys=[k for k in KEYS if k!='variant']
            c=per_seed[per_seed.variant.eq(variant)].set_index(keys)
            b=per_seed[per_seed.variant.eq(reference)].set_index(keys)
            common=c.index.intersection(b.index)
            if len(common)!=len(c):raise ValueError('测试对照缺失')
            d=c.loc[common][['balanced_acc']]-b.loc[common][['balanced_acc']]
            d=d.rename(columns={'balanced_acc':'gain'}).reset_index()
            d['variant']=variant;d['reference']=reference
            comparisons.append(d)
    paired=pd.concat(comparisons,ignore_index=True) if comparisons else pd.DataFrame()
    paired.to_csv(out/'paired_test_differences.csv',index=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'pdf.fonttype':42})
    figures=out/'figures';figures.mkdir(exist_ok=False)
    labels={'baseline_fusion':'原融合','baseline_data':'数据单分支','data_aux_ce':'数据辅助CE','residual_scale':'残差缩放'}
    # 每固定子集有跨种子SD；总览误差棒用子集均值之间SD，全样本退回种子SD并明确说明。
    for protocol in ['clean_only','awgn_mixed']:
        conditions=['clean'] if protocol=='clean_only' else ['clean','white_-4','white_-2','white_+0','white_+2','white_+4']
        fig,axes=plt.subplots(1,1,figsize=(8,5)) if protocol=='clean_only' else plt.subplots(3,2,figsize=(14,13))
        axlist=[axes] if protocol=='clean_only' else list(axes.flat)
        for ax,condition in zip(axlist,conditions):
            for variant,g in overall[(overall.domain=='window_random')&(overall.protocol==protocol)&
                    (overall.condition==condition)].groupby('variant'):
                g=g.sort_values('ratio')
                sd=g.subset_mean_sd.fillna(g.mean_within_subset_seed_sd)
                ax.errorbar(g.ratio*100,g.balanced_acc_mean*100,yerr=sd*100,marker='o',capsize=3,
                    label=labels.get(variant,variant))
            ax.set(xlabel='训练窗口比例 (%)',ylabel='均衡准确率 (%)',title=condition,
                xticks=[10,20,50,100],ylim=(0,105))
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle('V2基础实验：'+protocol)
        fig.tight_layout(rect=(0,.05,1,.96))
        fig.text(.5,.012,'小样本误差棒为3个子集均值的SD；每子集先平均3个训练种子。全样本为5种子SD。',ha='center',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(figures/f'window_{protocol}.{ext}',dpi=180)
        plt.close(fig)
    decision=json.loads((out/'STAGE_DECISIONS.json').read_text(encoding='utf-8'))
    table=overall[overall.domain.eq('window_random')].copy()
    for key in ['balanced_acc_mean','subset_mean_sd','mean_within_subset_seed_sd','max_within_subset_seed_sd','acc_mean','recall_OR_mean']:
        table[key]=100*table[key]
    selection='未有候选通过预定基础验证门槛，本轮停止跨负载扩展。' if decision['selected_candidate'] is None else (
        '通过基础验证门槛的候选为'+decision['selected_candidate']+'，已完成锁定参数的跨负载评价。')
    new=registry[registry.origin.eq('V2')]
    reference_count=int(registry.origin.eq('V1').sum())
    summary=new.groupby(['domain','variant','ratio','protocol'])[
        ['parameters','epochs_trained','updates_trained','total_train_seconds','train_seconds_per_epoch',
         'validation_seconds','fit_seconds','peak_gpu_mib','latency_ms']].mean().reset_index()
    summary.to_csv(out/'compute_summary.csv',index=False)
    lines=['# CWRU_V2 诊断与有限单变量优化报告','',
        f'完成{len(new)}个新模型；只读引用{reference_count}个V1模型的保存测试预测。'
        f'共{len(metrics)}组逐次指标。全部阶段决策先锁定，再统一测试。','',
        '## 阶段结论','',selection,'',
        '负结果同样保留。测试结果不能反过来更改候选是否通过验证门槛。没有强行放宽0.5个百分点波动门槛，'
        '没有选择性删除种子或困难原始记录。','',
        '[训练、物理特征与融合诊断](diagnosis/REPORT.md) · [小样本基线验证](SMALL_SAMPLE_BASELINE.md) · '
        '[完整阶段决策](STAGE_DECISIONS.json) · [数据使用](DATA_USAGE.md) · [协议](PROTOCOL.md)','',
        '## 基础实验测试结果','',
        '主指标均衡准确率。噪声实例先在每模型内平均，再在每固定子集内计算训练种子均值和样本SD，'
        '最后对3子集等权平均。subset_mean_sd描述子集均值之间差异，mean_within_subset_seed_sd为子集内种子SD的平均。'
        '这两者不合并为9个独立数据集重复，表中NaN表示只有一个全样本集合。单位均为百分数或百分点。','',
        table.round(3).to_markdown(index=False),'',
        '![纯干净基础实验](figures/window_clean_only.png)','',
        '![AWGN基础实验](figures/window_awgn_mixed.png)','',
        '## 计算成本','',summary.round(4).to_markdown(index=False),'',
        '静态FLOPs见complexity.csv，仅Conv1d和Linear，未覆盖物理FFT、归一化与逐元素算子。'
        '辅助CE增加训练前向及反向成本，推理不启用辅助头；真实成本以训练计时和单样本延迟为准。'
        'batch32，CUDA编码器BF16其余FP32；延迟batch1含归一化、在线物理特征及网络，10预热50重复，同步计时，排除磁盘和H2D。','',
        '## 研究边界','',
        '固定验证集被重复用于候选筛选，属于探索性研究。基础实验训练测试共享记录；测试集此前已有曝光，'
        '不能称独立确认性结果。103验证标签在所有小样本预算中保留。训练子集3次抽样与训练种子分开统计，'
        '仍不足以估计独立轴承泛化。跨负载方向复用模型和目标记录，也不构成独立轴承重复。'
        '不同预算的epoch批次数不同，早停实际更新量可能不同，不声称实际计算预算相同。'
        '0.5个百分点为用户稳定性筛选准则，不是显著性检验。此次两候选未检验所有可能改进，'
        '失败不等于所有融合方法无效，成功也不能证明所有噪声、类别和负载均受益。']
    if decision['selected_candidate'] is not None:
        loadtable=overall[overall.domain.ne('window_random')].copy()
        loadtable.to_csv(out/'load_transfer_summary.csv',index=False)
        lines+=['','## 锁定候选跨负载结果','',loadtable.round(5).to_markdown(index=False)]
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
