"""只汇总本轮保存的预测和指标，不重训或重新推理。"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import METRICS

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path

LIMITS = '''全部40记录和固定窗口划分已在历史开发中使用，历史测试曝光不会因新目录或重新训练消失。随机窗口训练和测试共享记录，只能评价记录内窗口识别。统一系数由全负载随机窗口验证选择，目标负载已参与方法开发，因此单来源迁移属于探索性评价，不能称严格未见域或独立轴承泛化。

本轮来源0/1/2/3 HP分别另用19/39/40/40个固定验证标签。训练样本数不是总标注预算。五训练种子描述优化随机性。噪声实例、子集和迁移方向不构成独立数据集重复。来源训练和验证共享记录，12方向复用来源模型和目标数据。

两组只能比较融合加辅助分类整体方案与数据单分支，不能分离物理融合和辅助损失各自贡献。固定训练设置不代表文献证明其最优。实际训练量受早停影响。全样本干净验证选出的系数不保证在小样本、噪声或迁移条件下最优，但不再逐条件调整。
'''


def exclusive(path, content):
    with path.open('x', encoding='utf-8') as f:
        f.write(content)


def validation_table(jobs):
    frames = []
    for j in jobs:
        frame = pd.read_csv(resolve_result_path(j['folder'])/'validation_metrics.csv')
        for key in ['job_id','domain','ratio','subset','protocol','variant','seed','aux_weight']:
            frame[key] = j[key]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def selection_report(root, jobs, rows, weight):
    out = root/'selection'
    out.mkdir(exist_ok=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'lambda_sensitivity.csv', index=False)
    val = validation_table(jobs)
    val.to_csv(out/'validation_metrics.csv', index=False)
    summary = val.groupby(['variant','aux_weight'])[METRICS+['validation_nll']].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    assert summary.acc_count.eq(5).all()
    summary.to_csv(out/'mean_std.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.errorbar(frame.aux_weight, frame.mean_best_validation_ce, yerr=frame.sample_sd, marker='o', capsize=3)
    ax.set(xscale='log', xlabel='Auxiliary CE weight', ylabel='Best validation CE (mean +/- sample SD)')
    fig.tight_layout(); fig.savefig(out/'lambda_sensitivity.png',dpi=180); plt.close(fig)
    exclusive(out/'REPORT.md', '# V3全样本纯干净验证选参\n\n'
        f'已完成30个从头训练模型。每模型最多200轮，耐心20。统一λ选为{weight}。只按五种子最佳验证窗口平均CE的均值最低选择，数值完全相同才选较小λ。本阶段未访问V3测试预测。\n\n'
        +frame.to_markdown(index=False)+'\n\n'
        '五档均保留用于验证参数敏感性分析。数据单分支为5种子共同对照，不随λ重复训练。选中λ及数据对照的10个模型复用于主实验，不重复训练。未选中四档不新增测试。\n\n'
        +summary.round(6).to_markdown(index=False)+'\n\n'+LIMITS)


def aggregate(frame, keys, out):
    summary = frame.groupby(keys)[METRICS].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    if not summary.acc_count.eq(5).all():
        raise ValueError('每配置必须完整5训练种子')
    summary.to_csv(out, index=False)
    return summary


def finish(root):
    out = root/'evaluation'
    protected = [root/'REPORT.md', root/'FIGURES.md', root/'figures', root/'validation_metrics.csv', root/'computational_cost.csv']
    protected += [out/name for name in ['per_seed_metrics.csv','mean_std.csv','paired_seed_differences.csv','paired_mean_std.csv','transfer_equal_direction_mean_std.csv','pooled_confusion_matrices.csv']]
    if any(path.exists() for path in protected):
        raise FileExistsError('报告输出已存在，禁止覆盖')
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    m = pd.read_csv(out/'metrics.csv')
    validate(root, cfg, m)
    keys = ['domain','ratio','subset','protocol','variant','source_load_hp','target_load_hp','condition']
    # 每模型先平均噪声实例，再在固定子集内汇总五训练种子。
    seed = m.groupby(keys+['seed'], as_index=False)[METRICS].mean()
    seed.to_csv(out/'per_seed_metrics.csv', index=False)
    summary = aggregate(seed, keys, out/'mean_std.csv')
    pair_keys = [k for k in keys if k!='variant']+['seed']
    paired = seed.pivot(index=pair_keys,columns='variant',values=METRICS)
    if paired.isna().any().any():
        raise ValueError('两方法配对不完整')
    differences = pd.DataFrame({metric:paired[(metric,'data_aux_ce')]-paired[(metric,'baseline_data')] for metric in METRICS}).reset_index()
    differences.to_csv(out/'paired_seed_differences.csv', index=False)
    aggregate(differences, [k for k in pair_keys if k!='seed'], out/'paired_mean_std.csv')
    # 12方向等权，保留相同seed索引，不把方向当独立重复。
    transfer = seed[seed.source_load_hp.ne(-1)].groupby(['ratio','subset','protocol','variant','condition','seed'],as_index=False)[METRICS].mean()
    aggregate(transfer,['ratio','subset','protocol','variant','condition'],out/'transfer_equal_direction_mean_std.csv')
    val = validation_table(cfg['jobs'])
    val.to_csv(root/'validation_metrics.csv',index=False)
    costs = pd.DataFrame(cfg['jobs'])
    keep = ['job_id','domain','ratio','subset','protocol','variant','seed','aux_weight',
            'training_windows','validation_windows','epochs_trained','updates_trained','best_epoch',
            'parameters','conv_linear_flops','fit_seconds','train_seconds_per_epoch','total_train_seconds',
            'validation_seconds','peak_gpu_mib','latency_ms','hardware','precision','batch_size','latency_scope']
    costs[keep].to_csv(root/'computational_cost.csv',index=False)
    # 汇总混淆计数保留支持数；重复预测总数不能称独立样本数。
    cm = pd.read_csv(out/'confusion_matrices.csv')
    pooled = cm.groupby(keys+['true_label','predicted_label'],as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby(keys+['true_label'])['count'].transform('sum')
    pooled['row_percent'] = 100*pooled['count']/pooled.prediction_support
    pooled.to_csv(out/'pooled_confusion_matrices.csv',index=False)
    plot_curves(cfg['jobs'], root)
    exclusive(root/'REPORT.md', '# CWRU_V3正式报告\n\n'
        f'完成80个从头训练的全样本来源模型。统一λ为{cfg["selected_aux_weight"]}。本轮12方向、240模型目标组合、3240逐条件指标。不开展小样本。全部训练和选模完成后统一测试。\n\n'
        '逐配置五种子均值及样本标准差见evaluation/mean_std.csv。噪声先在模型内平均；本轮只有全样本，不涉及子集间波动。两方法配对差值见paired_mean_std.csv。12方向等权汇总见transfer_equal_direction_mean_std.csv。所有ACC、BA、Macro-F1和四类召回均报告，不以总BA掩盖单类退化。原始及逐行归一化混淆矩阵均保留。\n\n'
        '模型成本见computational_cost.csv。单样本延迟包含窗口标准化、在线物理特征和网络，10次预热、50次同步测量，不含磁盘与H2D。Conv1d/Linear FLOPs不含FFT，不能作为完整FLOPs。训练梯度诊断与标准化拟合耗时单独保存在每模型parameters.json。\n\n'
        +LIMITS+'\n\n## 12方向等权结果\n\n'
        +pd.read_csv(out/'transfer_equal_direction_mean_std.csv').round(6).to_markdown(index=False)+'\n')


def validate(root,cfg,m):
    from run_load_experiment import full_cases
    assert len(cfg['jobs'])==80 and cfg['decisions_locked']
    keys=['domain','protocol','variant','seed']
    assert {tuple(j[k] for k in keys) for j in cfg['jobs']}=={tuple(c[k] for k in keys) for c in full_cases()}
    assert len(m)==3240 and not m.duplicated(['job_id','target_load_hp','condition','noise_seed']).any()
    e=pd.read_csv(root/'evaluation/evaluations.csv')
    assert len(e)==240 and not e.duplicated(['job_id','target_load_hp']).any()
    cm=pd.read_csv(root/'evaluation/confusion_matrices.csv')
    assert len(cm)==51840
    for j in cfg['jobs']:
        a=m[m.job_id.eq(j['job_id'])]
        assert set(a.target_load_hp)==set(range(4))-{j['source_load_hp']}
        for target,g in a.groupby('target_load_hp'):
            expected={('clean',0)}
            if j['protocol']=='awgn_mixed':
                expected|={(f'white_{snr:+d}',seed) for snr in [-4,-2,0,2,4] for seed in range(7201,7206)}
            assert set(zip(g.condition,g.noise_seed))==expected
    supports=pd.read_csv(root/'dataset_support.csv')
    for (domain,target,condition,noise),g in cm.groupby(['domain','target_load_hp','condition','noise_seed']):
        support=supports[(supports.domain==domain)&(supports.role==f'test_hp{target}')].iloc[0]
        for label,col in enumerate(['N','IR','OR','B']):
            assert g[g.true_label==label].support.eq(int(support[col])).all()
    counts=cm.groupby(['job_id','target_load_hp','condition','noise_seed','true_label'])
    assert (counts['count'].sum()==counts.support.first()).all()


def plot_curves(jobs,root):
    from window_report import _curves
    folders=[]
    for domain,g in pd.DataFrame(jobs).groupby('domain'):
        dest=root/'figures'/domain
        dest.mkdir(parents=True,exist_ok=False)
        folders.extend((domain,name) for name in _curves(g,dest))
    exclusive(root/'FIGURES.md','# 各来源训练轨迹\n\n'+''.join(f'## {domain}\n\n![{name}](figures/{domain}/{name})\n\n' for domain,name in folders))
