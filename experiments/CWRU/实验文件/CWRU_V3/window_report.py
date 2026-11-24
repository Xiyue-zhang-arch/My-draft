"""V3随机窗口阶段报告。仅汇总已保存结果，不训练、不推理、不修改引用结果。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import METRICS
from report import validation_table

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path


SEEDS = set(range(1706, 1711))
VARIANTS = ('baseline_data', 'data_aux_ce')
CONDITIONS = ['clean', 'white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4']
KEYS = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'source_load_hp',
        'target_load_hp', 'condition']
META = ['job_id', 'domain', 'ratio', 'subset', 'protocol', 'variant', 'seed',
        'aux_weight', 'source_load_hp', 'target_load_hp']
COLORS = {'baseline_data': '#2864a0', 'data_aux_ce': '#c44b32'}
NAMES = {'baseline_data': 'Data branch', 'data_aux_ce': 'Fusion + auxiliary CE'}


def _csv(frame, path):
    frame.to_csv(path, index=False, mode='x')


def _text(path, text):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def _mean_std(frame, keys, metrics=METRICS):
    counts = frame.groupby(keys).seed.agg(['nunique', 'count'])
    if not (counts['nunique'].eq(5).all() and counts['count'].eq(5).all()):
        raise ValueError('报告要求每配置恰好5个训练种子')
    if any(set(g.seed) != SEEDS for _, g in frame.groupby(keys)):
        raise ValueError('训练种子必须为1706至1710')
    summary = frame.groupby(keys)[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    return summary.reset_index()


def _validate(cfg, m, cm, evaluations):
    jobs = pd.DataFrame(cfg['jobs'])
    if len(jobs) != 200 or jobs.job_id.nunique() != 200:
        raise ValueError('随机窗口报告必须覆盖200个不同模型')
    cases = ['ratio', 'subset', 'protocol', 'variant', 'seed']
    expected = {(ratio, subset, protocol, variant, seed)
                for ratio in (1., .1, .2, .5)
                for subset in ([0] if ratio == 1. else [4101, 4102, 4103])
                for protocol in ('clean_only', 'awgn_mixed')
                for variant in VARIANTS for seed in SEEDS}
    if set(jobs[cases].itertuples(index=False, name=None)) != expected:
        raise ValueError('随机窗口模型配置集合不完整')
    refs = jobs.origin.eq('selection_reference') if 'origin' in jobs else pd.Series(False, index=jobs.index)
    if refs.sum() != 10 or not (jobs.loc[refs, 'ratio'].eq(1.).all()
            and jobs.loc[refs, 'protocol'].eq('clean_only').all()):
        raise ValueError('只允许复用10个全样本纯干净选参模型')
    if not jobs.loc[jobs.variant.eq('data_aux_ce'), 'aux_weight'].eq(cfg['selected_aux_weight']).all():
        raise ValueError('融合模型未使用已锁定的统一lambda')
    if not jobs.loc[jobs.variant.eq('baseline_data'), 'aux_weight'].eq(0).all():
        raise ValueError('数据单分支不能有辅助损失')
    if len(m) != 2700 or len(cm) != 43200 or len(evaluations) != 200:
        raise ValueError('要求2700逐次指标、43200混淆计数、200份模型评价')
    for frame in (jobs, m, cm, evaluations):
        if not frame.domain.eq('window_random').all():
            raise ValueError('本报告仅包含随机窗口实验')
        for key in ('source_load_hp', 'target_load_hp'):
            if key in frame and not frame[key].eq(-1).all():
                raise ValueError('随机窗口结果不能混入跨负载目标')
    if m.duplicated(['job_id', 'condition', 'noise_seed']).any():
        raise ValueError('逐次指标重复')
    if evaluations.job_id.nunique() != 200 or not evaluations.test_windows.eq(103).all():
        raise ValueError('每模型必须评价同一固定103测试窗')
    expected_ids = set(jobs.job_id)
    for frame in (m, cm, evaluations):
        if set(frame.job_id) != expected_ids:
            raise ValueError('模型登记与评价对象不一致')
        common = [c for c in META if c in frame and c in jobs and c != 'job_id']
        check = frame[['job_id', *common]].drop_duplicates().merge(
            jobs[['job_id', *common]], on='job_id', suffixes=('_result', '_job'), validate='many_to_one')
        if any(not check[c+'_result'].eq(check[c+'_job']).all() for c in common):
            raise ValueError('评价元数据与模型登记不一致')
    for job in cfg['jobs']:
        a = m[m.job_id.eq(job['job_id'])]
        expected_conditions = {('clean', 0)}
        if job['protocol'] == 'awgn_mixed':
            expected_conditions |= {(c, n) for c in CONDITIONS[1:] for n in range(7201, 7206)}
        if set(zip(a.condition, a.noise_seed)) != expected_conditions:
            raise ValueError('测试条件或噪声实例不完整')
        if int(job['validation_windows']) != 103:
            raise ValueError('随机窗口验证标签预算必须为103')
    values = m[METRICS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('分类指标非有限或不在0至1范围')
    eval_keys = ['job_id', 'condition', 'noise_seed']
    cm_keys = eval_keys + ['true_label', 'predicted_label']
    if cm.duplicated(cm_keys).any() or not cm['count'].ge(0).all() or not cm['count'].mod(1).eq(0).all():
        raise ValueError('混淆矩阵重复或计数错误')
    if set(cm.true_label) != set(range(4)) or set(cm.predicted_label) != set(range(4)):
        raise ValueError('混淆矩阵类别错误')
    support = cm.groupby(eval_keys+['true_label'])['count'].sum()
    actual_support = support.unstack('true_label')
    if not np.array_equal(actual_support.to_numpy(), np.tile([11, 30, 31, 31], (2700, 1))):
        raise ValueError('测试类别支持数必须为11/30/31/31')
    if not cm.groupby(eval_keys).size().eq(16).all():
        raise ValueError('每次评价必须保存完整4乘4混淆计数')
    # 由混淆计数复核指标，避免只检查行数而漏掉结果错配；不重新推理。
    diag = cm[cm.true_label.eq(cm.predicted_label)].set_index(eval_keys+['true_label'])['count']
    recalls = (diag/support).unstack('true_label')
    precision_denominator = cm.groupby(eval_keys+['predicted_label'])['count'].sum()
    precision_denominator.index = precision_denominator.index.set_names(eval_keys+['true_label'])
    f1 = (2*diag/(support+precision_denominator)).unstack('true_label').mean(axis=1)
    recomputed = pd.DataFrame({'acc': diag.groupby(level=eval_keys).sum()/103,
        'balanced_acc': recalls.mean(axis=1), 'f1_macro': f1,
        **{metric: recalls[i] for i, metric in enumerate(METRICS[3:])}})
    recorded = m.set_index(eval_keys)[METRICS].sort_index()
    if not recorded.index.equals(recomputed.sort_index().index) or not np.allclose(
            recorded, recomputed[METRICS].sort_index(), atol=1e-9, rtol=0):
        raise ValueError('保存指标与混淆计数不一致')
    if not all(resolve_result_path(p).is_file() for p in evaluations.prediction_path):
        raise ValueError('登记预测文件缺失')
    return jobs


def _costs(jobs):
    jobs = jobs.copy()
    if 'origin' in jobs:
        jobs['origin'] = jobs.origin.fillna('new')
    fields = ['job_id', 'origin', 'domain', 'ratio', 'subset', 'protocol', 'variant', 'seed',
        'aux_weight', 'training_windows', 'validation_windows', 'epochs_trained', 'updates_trained',
        'best_epoch', 'parameters', 'conv_linear_flops', 'fit_seconds', 'train_seconds_per_epoch',
        'total_train_seconds', 'gradient_diagnostic_seconds', 'total_training_with_diagnostics_seconds',
        'validation_seconds', 'peak_gpu_mib', 'latency_ms', 'hardware', 'precision', 'batch_size',
        'latency_scope', 'peak_memory_scope', 'folder']
    return jobs[[c for c in fields if c in jobs]]


def _curves(jobs, figures):
    paths = []
    for (protocol, ratio, subset), group in jobs.groupby(['protocol', 'ratio', 'subset']):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        noisy = protocol == 'awgn_mixed'
        columns = [('main_ce', 'Training main CE'), ('train_acc_sampled_augmented', 'Training ACC'),
            ('validation_nll', 'Checkpoint validation CE'),
            ('validation_clean_balanced_acc', 'Clean validation BA'),
            ('validation_white_-4_balanced_acc' if noisy else 'validation_clean_acc',
             '-4 dB validation BA' if noisy else 'Clean validation ACC'),
            ('validation_white_-4_recall_OR' if noisy else 'validation_clean_recall_OR',
             '-4 dB validation OR recall' if noisy else 'Clean validation OR recall')]
        for job in group.to_dict('records'):
            h = pd.read_csv(resolve_result_path(job['folder'])/'training.csv')
            required = ['epoch'] + [c for c, _ in columns]
            if not set(required).issubset(h.columns) or len(h) != job['epochs_trained']:
                raise ValueError('训练逐轮记录缺失或与完成登记不一致')
            if not np.array_equal(h.epoch, np.arange(1, len(h)+1)):
                raise ValueError('训练epoch记录不连续')
            for ax, (column, title) in zip(axes.flat, columns):
                ax.plot(h.epoch, h[column], color=COLORS[job['variant']], alpha=.45, linewidth=.9,
                    label=NAMES[job['variant']] if job['seed'] == 1706 else None)
                ax.set(xlabel='Epoch', ylabel=title)
                ax.grid(alpha=.2)
                if 'CE' not in title:
                    ax.set_ylim(-.02, 1.02)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{protocol}, train={ratio:.0%}, subset={subset}; individual seed trajectories')
        fig.tight_layout()
        path = figures/f'training_{protocol}_r{int(ratio*100):03d}_s{subset}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path.name)
    return paths


def _overview(summary, pairs, figures):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for row, (protocol, condition) in enumerate([('clean_only', 'clean'), ('awgn_mixed', 'white_-4')]):
        for col, ratio in enumerate([.1, .2, .5, 1.]):
            ax = axes[row, col]
            a = summary[summary.protocol.eq(protocol)&summary.condition.eq(condition)&summary.ratio.eq(ratio)]
            subsets = sorted(a.subset.unique())
            for offset, variant in zip([-.07, .07], VARIANTS):
                g = a[a.variant.eq(variant)].sort_values('subset')
                ax.errorbar(np.arange(len(subsets))+offset, 100*g.balanced_acc_mean,
                    yerr=100*g.balanced_acc_std, fmt='o-', capsize=3,
                    color=COLORS[variant], label=NAMES[variant])
            ax.set(xticks=np.arange(len(subsets)), xticklabels=[str(s) if s else 'full' for s in subsets],
                title=f'{protocol}, {ratio:.0%}\n{condition}', xlabel='Training subset', ylabel='Test BA (%)')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Mean +/- sample SD over five training seeds; noise averaged within each model')
    fig.tight_layout()
    fig.savefig(figures/'accuracy_overview.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharey=True)
    for ax, ((ratio, subset), a) in zip(axes.flat, summary[summary.protocol.eq('awgn_mixed')].groupby(['ratio', 'subset'])):
        for variant in VARIANTS:
            g = a[a.variant.eq(variant)&a.condition.ne('clean')].copy()
            g['snr'] = g.condition.str.removeprefix('white_').astype(int)
            g = g.sort_values('snr')
            ax.errorbar(g.snr, 100*g.balanced_acc_mean, yerr=100*g.balanced_acc_std,
                fmt='o-', capsize=2, color=COLORS[variant], label=NAMES[variant])
        ax.set(title=f'{ratio:.0%}, subset={subset}', xlabel='Test SNR (dB)', ylabel='BA (%)', xticks=[-4,-2,0,2,4])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('AWGN test curves; five-seed means and sample SD, noise averaged within each model')
    fig.tight_layout()
    fig.savefig(figures/'awgn_curves.png', dpi=180)
    plt.close(fig)
    labels = [('clean_only', 'clean')] + [('awgn_mixed', c) for c in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for ax, metric, title in zip(axes, ['acc', 'balanced_acc', 'recall_OR'], ['ACC', 'BA', 'OR recall']):
        table = pairs.pivot(index=['ratio', 'subset'], columns=['protocol', 'condition'], values=metric+'_mean')
        table = 100*table.reindex(columns=pd.MultiIndex.from_tuples(labels))
        values = table.to_numpy(float)
        lim = max(float(np.max(np.abs(values))), .5)
        im = ax.imshow(values, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
        ax.set(xticks=np.arange(len(labels)), xticklabels=['Clean train\nclean test', 'AWGN train\nclean test', '-4', '-2', '0', '2', '4'],
            yticks=np.arange(len(table)), yticklabels=[f'{r:.0%} / {s}' for r, s in table.index], title=f'{title}: fusion minus data (pp)')
        ax.tick_params(axis='x', labelsize=7)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=7,
                color='white' if abs(value) > .65*lim else 'black')
        fig.colorbar(im, ax=ax, fraction=.04)
    fig.tight_layout()
    fig.savefig(figures/'paired_difference_overview.png', dpi=180)
    plt.close(fig)


def _display(summary):
    rows = []
    for r in summary.to_dict('records'):
        row = {'训练协议': r['protocol'], '训练比例': f'{r["ratio"]:.0%}', '子集': int(r['subset']),
               '测试条件': r['condition'], '方法': '数据单分支' if r['variant']=='baseline_data' else '融合加辅助分类'}
        for key, name in zip(METRICS, ['ACC', 'BA', 'Macro-F1', '正常召回', '内圈召回', '外圈召回', '滚动体召回']):
            row[name] = f'{100*r[key+"_mean"]:.2f}±{100*r[key+"_std"]:.2f}'
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def finish(root):
    """完整200模型评价落盘后调用；重复调用或部分报告已存在时拒绝覆盖。"""
    root = Path(root).resolve()
    out = root/'evaluation'
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    m = pd.read_csv(out/'metrics.csv')
    cm = pd.read_csv(out/'confusion_matrices.csv')
    evaluations = pd.read_csv(out/'evaluations.csv')
    jobs = _validate(cfg, m, cm, evaluations)
    outputs = [root/'REPORT.md', root/'validation_metrics.csv', root/'computational_cost.csv',
        root/'figures', out/'per_seed_metrics.csv', out/'mean_std.csv', out/'between_subset_summary.csv',
        out/'paired_seed_differences.csv', out/'paired_mean_std.csv', out/'paired_between_subset_summary.csv',
        out/'pooled_confusion_matrices.csv', out/'validation_mean_std.csv']
    if any(p.exists() for p in outputs):
        raise FileExistsError('报告文件已存在，禁止覆盖或自动重复生成')
    # 先在同一模型内平均5个噪声实例，再以训练种子为单位计算样本标准差。
    seed = m.groupby(KEYS+['seed'], as_index=False)[METRICS].mean()
    summary = _mean_std(seed, KEYS)
    between_keys = [k for k in KEYS if k != 'subset']
    subset = seed[seed.subset.ne(0)].groupby(KEYS, as_index=False)[METRICS].mean()
    between = subset.groupby(between_keys)[METRICS].agg(['mean', 'std', 'count'])
    between.columns = ['_'.join(c) for c in between.columns]
    if not between.acc_count.eq(3).all():
        raise ValueError('小样本结果要求每预算完整3子集')
    pair_keys = [k for k in KEYS if k != 'variant']+['seed']
    pair_table = seed.pivot(index=pair_keys, columns='variant', values=METRICS)
    if pair_table.isna().any().any() or set(pair_table.columns.get_level_values(1)) != set(VARIANTS):
        raise ValueError('两方法配对不完整')
    differences = pd.DataFrame({k: pair_table[(k, 'data_aux_ce')]-pair_table[(k, 'baseline_data')]
        for k in METRICS}).reset_index()
    pair_summary = _mean_std(differences, pair_keys[:-1])
    pair_subset = differences[differences.subset.ne(0)].groupby(pair_keys[:-1], as_index=False)[METRICS].mean()
    pair_between = pair_subset.groupby([k for k in pair_keys[:-1] if k != 'subset'])[METRICS].agg(['mean','std','count'])
    pair_between.columns = ['_'.join(c) for c in pair_between.columns]
    val = validation_table(cfg['jobs'])
    val_seed = val.groupby(['domain','ratio','subset','protocol','variant','condition','seed'], as_index=False)[METRICS+['validation_nll']].mean()
    val_summary = _mean_std(val_seed, ['domain','ratio','subset','protocol','variant','condition'], METRICS+['validation_nll'])
    pooled = cm.groupby(KEYS+['true_label','predicted_label'], as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby(KEYS+['true_label'])['count'].transform('sum')
    pooled['row_percent'] = 100*pooled['count']/pooled.prediction_support
    pooled['unique_test_support'] = pooled.true_label.map({0:11, 1:30, 2:31, 3:31})
    pooled['prediction_repetitions'] = np.where(pooled.condition.eq('clean'), 5, 25)
    _csv(seed, out/'per_seed_metrics.csv')
    _csv(summary, out/'mean_std.csv')
    _csv(between.reset_index(), out/'between_subset_summary.csv')
    _csv(differences, out/'paired_seed_differences.csv')
    _csv(pair_summary, out/'paired_mean_std.csv')
    _csv(pair_between.reset_index(), out/'paired_between_subset_summary.csv')
    _csv(pooled, out/'pooled_confusion_matrices.csv')
    _csv(val, root/'validation_metrics.csv')
    _csv(val_summary, out/'validation_mean_std.csv')
    _csv(_costs(jobs), root/'computational_cost.csv')
    figures = root/'figures'
    figures.mkdir(exist_ok=False)
    curve_paths = _curves(jobs, figures)
    _overview(summary, pair_summary, figures)
    selected = summary[summary.condition.eq('clean') | (summary.protocol.eq('awgn_mixed')&summary.condition.eq('white_-4'))]
    p4 = pair_summary[pair_summary.protocol.eq('awgn_mixed')&pair_summary.condition.eq('white_-4')]
    pair_rows = []
    for row in p4.to_dict('records'):
        pair_rows.append({'训练比例': f'{row["ratio"]:.0%}', '子集': int(row['subset']),
            **{name: f'{100*row[key+"_mean"]:+.2f}±{100*row[key+"_std"]:.2f}'
               for key, name in [('acc','ACC差值'), ('balanced_acc','BA差值'), ('recall_OR','外圈召回差值')]}})
    support_path = root/'dataset_support.csv'
    support_text = ''
    if support_path.is_file():
        supports = pd.read_csv(support_path)
        support_text = '\n\n'+supports[['ratio','role','windows','N','IR','OR','B']].drop_duplicates().to_markdown(index=False)
    source = str(cfg.get('reference_source', '原V3选参目录'))
    text = f'''# CWRU_V3随机窗口阶段正式报告

本阶段完成190个新模型的训练和测试，并只读复用已完成选参阶段的10个全样本纯干净模型及其测试预测，共200个模型。统一λ固定为{cfg['selected_aux_weight']}。本阶段未开展按负载划分的迁移实验，也未重新选参。

## 数据和实验范围

全部40条记录切为687个完整窗口，复用固定的记录内70%/15%/15%随机划分。全样本训练481窗、验证103窗、测试103窗。训练比例10%、20%、50%对应49、98、242个训练窗，每档3个固定嵌套子集4101至4103。每子集、训练协议和方法均使用1706至1710五个训练种子。全样本使用同一完整训练池，不设置3份重复全样本子集。

纯干净和AWGN协议均比较数据单分支与原融合加数据分支辅助分类。保留原骨干、门控和共享分类头。40条原始记录没有排除，data只读。所选窗口、逐MAT用量及逐类支持数见splits、dataset_file_usage.csv、training_subset_file_usage.csv和各模型的selected_windows.csv。{support_text}

λ来自2026年10月22日完整30模型的全样本纯干净选参。五档λ各5种子，另有数据单分支5种子。选择只依据五种子最佳验证普通窗口平均CE的均值，同值选较小λ。所引用结果来源为`{source}`。本阶段未重新训练这些模型，未重新计算其测试预测。未入选的20个选参模型不计入本阶段200模型。

## 训练和测试时机

最多200epoch，patience为20，每轮验证并保存最佳权重，最小改善量1e-6。AdamW初始学习率5e-4、权重衰减1e-3、batch size为32、梯度裁剪5。使用200轮单周期余弦衰减至1e-5，无重启。普通CE，标签平滑为0。验证仅使用最终分类输出，不包含辅助损失。纯干净按窗口平均验证CE选模。AWGN使用0.5乘干净验证CE与五档噪声验证CE各0.1之和。以上为固定实验设置，不宣称文献证明其最优，实际epoch和更新量受早停影响。

AWGN每次窗口出现时以50%概率保留干净，否则SNR连续均匀采样[-4,4] dB。融合两分支使用相同观测。纯干净仅用本训练子集干净观测拟合median/IQR。AWGN融合使用本训练子集等量干净和含噪观测拟合。3次固定噪声拟合实现的种子为6301，尺度下限0.05，裁剪[-8,8]。小样本全部从头训练，未使用全样本统计或预训练权重。重复观测不增加独立训练窗口数。

本阶段190个新模型全部选模并固定后才开展新增测试。先前10模型的测试已经完成，本次仅引用。200份模型评价共2700组逐条件指标，其中100个纯干净模型各1组，100个AWGN模型各26组。AWGN测试包括干净与五档SNR各5个噪声实例。验证噪声6201，测试噪声7201至7205，二者分离。所有测试采用相同103窗，类别支持数为正常11、内圈30、外圈31、滚动体31。测试没有用于改变λ、子集、训练设置或选模规则。

## 统计口径

下表为百分比，均值±样本标准差。每个模型先平均同一SNR下的5个噪声实例，再在固定子集内对5个训练种子计算均值及样本标准差，分母为4。纯干净仅1个测试实例。三个子集各自报告，子集五种子均值之间的样本标准差单列，不将15个组合称为15个独立种子。

ACC、BA、Macro-F1和全部四类召回完整保存于evaluation/mean_std.csv。两方法在相同子集、训练协议、种子和测试条件下配对，差值为融合加辅助分类减数据单分支。配对五种子差值及其样本标准差见paired_seed_differences.csv和paired_mean_std.csv。between_subset_summary.csv及paired_between_subset_summary.csv分别报告性能和配对增益在3个子集间的变化。这里没有将SD当作置信区间，也没有根据均值排序宣称统计显著。

原始混淆计数见evaluation/confusion_matrices.csv，聚合计数与逐行百分比见pooled_confusion_matrices.csv。每行按真实类别归一化，行和100%，对角线为召回率。unique_test_support保存独立窗口的类别数，prediction_support是重复预测总数。五种子或噪声重复不会扩大独立测试样本数。

## 干净及强噪声结果

{_display(selected)}

## AWGN负4dB配对差值

下表单位为百分点。正值表示融合方案较高，负值表示融合方案较低。必须同时检查BA和外圈召回，不能用总体均值掩盖某类或某个子集退化。

{pd.DataFrame(pair_rows).to_markdown(index=False)}

![固定子集五种子准确率](figures/accuracy_overview.png)

![AWGN噪声曲线](figures/awgn_curves.png)

![配对差值概览](figures/paired_difference_overview.png)

## 训练曲线和计算成本

各模型training.csv、validation_curve.csv保留逐epoch训练和验证记录，parameters.json和training_spec.json保留设置、最优epoch及实际训练量。20组训练曲线展示全部5种子各自的轨迹，不补齐早停后的epoch，也不把仅剩运行种子的均值解释为固定5种子的变化。训练ACC对应实际抽样及增强观测，不能直接当作未经增强训练集的评价准确率。主CE与总训练损失区分，辅助CE不进入验证。

computational_cost.csv列出参数量、单样本推理延迟、每epoch训练时间、拟合时间、验证时间、梯度诊断时间、峰值显存、硬件和精度。单样本推理包括窗口标准化、在线物理特征和网络，10次预热、50次同步测量，不含磁盘和H2D；训练batch size为32，推理测量batch size为1。物理特征未缓存以隐去成本。Conv1d/Linear FLOPs不含FFT，不能宣称完整FLOPs。峰值显存按已分配显存记录，包含首批梯度诊断。10个引用模型的成本为原实测，不计作本阶段新增训练耗时。成本的逐模型测量范围以latency_scope和peak_memory_scope为准。

## 解释边界

本阶段只评价固定40条记录内随机窗口识别。训练、验证和测试共享原始记录，窗口间可能相关，不能据此声称跨记录、跨轴承或跨负载泛化。五训练种子仅描述训练随机性，三个子集仅描述同一训练池的抽样变化，噪声实例也不是独立数据集重复。

小样本每个配置仍使用103个有标签验证窗口，训练窗口数不是总标注预算。统一λ在全样本、全负载随机窗口数据上选定，本阶段的小样本结果评价固定λ下的方案性能，不能解释为完全不借助全样本开发信息的少标签学习。全样本干净验证所选λ不保证在含噪或小样本条件下最优，本阶段遵守统一λ，不逐条件调整。

历史开发已经访问过相关测试数据。2026年10月22日选参30模型还按用户要求提前测试，本阶段无法称为全部研究决策之前从未接触测试的独立确认性实验。旧测试结果只读复用，建立新目录或从头训练不能消除测试曝光。

两组对照只能评价融合加辅助分类这一整体方案与数据单分支的差别，不能分离物理融合和辅助损失各自贡献。即使平均性能提高，也不能据此证明特定物理机制或单独辅助损失带来因果改善。若某些条件下降，须保留并报告，不能用跨条件平均掩盖。

跨负载实验本阶段没有运行。统一λ由全负载随机窗口验证选定，未来目标负载已经参与方法开发。后续单一来源12方向只能按此开发背景作探索性评价，不能声称严格未见域或独立轴承泛化。

## 训练曲线索引

'''
    text += '\n'.join(f'- [{name}](figures/{name})' for name in curve_paths)+'\n'
    _text(root/'REPORT.md', text)
    return {'models': 200, 'new_models': 190, 'reference_models': 10,
            'metric_rows': 2700, 'evaluation_pairs': 200, 'report': str(root/'REPORT.md')}
