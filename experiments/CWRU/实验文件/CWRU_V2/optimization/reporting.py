"""仅从已保存指标和预测生成阶段报告，不训练、不推理、不决定候选是否采用。"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path


METRICS = ['acc', 'balanced_acc', 'f1_macro', 'recall_N', 'recall_IR', 'recall_OR', 'recall_B']
MODEL_KEYS = ['config_id', 'variant', 'domain', 'ratio', 'subset', 'protocol', 'seed',
              'source_load_hp', 'target_load_hp', 'condition']
PAIR_KEYS = ['domain', 'ratio', 'subset', 'protocol', 'seed',
             'source_load_hp', 'target_load_hp', 'condition']
META = ['aux_weight', 'physics_fit', 'aux_head', 'val_noise_count', 'epochs_limit',
        'control_config_id']
CONDITIONS = ['clean', 'white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4']
LABELS = {'baseline_data': '数据单分支', 'data_aux_ce': '融合加辅助分类'}
PROJECT = Path(__file__).resolve().parents[5]


def _validate_metrics(frame):
    frame = pd.DataFrame(frame).copy()
    required = ['job_id', 'noise_seed'] + MODEL_KEYS + METRICS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'指标缺少字段: {sorted(missing)}')
    if frame.empty:
        raise ValueError('不能为尚未评价的空指标表生成完成报告')
    if frame[required].isna().any().any():
        raise ValueError('指标主键或科学数值包含缺失值')
    if not set(frame.variant).issubset(LABELS):
        raise ValueError('后续报告只接受数据单分支与融合加辅助分类')
    values = frame[METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < -1e-12) | (values > 1 + 1e-12)).any():
        raise ValueError('准确率及召回率必须是[0,1]的有限值')
    raw_keys = ['job_id', 'source_load_hp', 'target_load_hp', 'condition', 'noise_seed']
    if frame.duplicated(raw_keys).any():
        raise ValueError('同一模型、目标和噪声实例存在重复指标，不能当作新增重复')
    if frame.groupby(MODEL_KEYS, dropna=False).job_id.nunique().gt(1).any():
        raise ValueError('同配置同种子出现多个模型，必须先核对模型登记')
    return frame


def _metadata(frame, keys):
    """只传播在科学分组内不变的配置，拒绝把不一致配置简单取首项。"""
    columns = [c for c in META if c in frame and c not in keys]
    if not columns:
        return None
    grouped = frame.groupby(keys, dropna=False)
    for column in columns:
        if grouped[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f'分组内配置不一致: {column}')
    return grouped[columns].first().reset_index()


def summarize_metrics(frame):
    """噪声在模型内平均，种子在固定子集内统计，再对各子集等权汇总。"""
    frame = _validate_metrics(frame)
    grouped = frame.groupby(MODEL_KEYS, dropna=False)
    per_seed = grouped[METRICS].mean().reset_index()
    counts = grouped.agg(job_id=('job_id', 'first'), noise_instance_count=('noise_seed', 'nunique')).reset_index()
    per_seed = per_seed.merge(counts, on=MODEL_KEYS, validate='one_to_one')
    metadata = _metadata(frame, MODEL_KEYS)
    if metadata is not None:
        per_seed = per_seed.merge(metadata, on=MODEL_KEYS, validate='one_to_one')

    subset_keys = [k for k in MODEL_KEYS if k != 'seed']
    grouped = per_seed.groupby(subset_keys, dropna=False)
    within = grouped[METRICS].agg(['mean', 'std', 'count'])
    within.columns = ['_'.join(column) for column in within.columns]
    within = within.reset_index()
    counts = grouped.agg(seed_count=('seed', 'nunique'),
                         min_noise_instances=('noise_instance_count', 'min'),
                         max_noise_instances=('noise_instance_count', 'max')).reset_index()
    within = within.merge(counts, on=subset_keys, validate='one_to_one')
    within['insufficient_training_seeds'] = within.seed_count.lt(3)
    within['seed_sd_over_0_5pp'] = within.balanced_acc_std.gt(.005)
    metadata = _metadata(per_seed, subset_keys)
    if metadata is not None:
        within = within.merge(metadata, on=subset_keys, validate='one_to_one')

    overall_keys = [k for k in subset_keys if k != 'subset']
    grouped = within.groupby(overall_keys, dropna=False)
    aggregations = dict(subset_count=('subset', 'nunique'),
                        min_seed_count=('seed_count', 'min'), max_seed_count=('seed_count', 'max'),
                        min_noise_instances=('min_noise_instances', 'min'),
                        max_noise_instances=('max_noise_instances', 'max'))
    for metric in METRICS:
        aggregations[f'{metric}_mean'] = (f'{metric}_mean', 'mean')
        aggregations[f'{metric}_between_subset_sd'] = (f'{metric}_mean', 'std')
        aggregations[f'{metric}_mean_within_subset_seed_sd'] = (f'{metric}_std', 'mean')
        aggregations[f'{metric}_max_within_subset_seed_sd'] = (f'{metric}_std', 'max')
    overall = grouped.agg(**aggregations).reset_index()
    overall['subset_mean_sd'] = overall.balanced_acc_between_subset_sd
    overall['mean_within_subset_seed_sd'] = overall.balanced_acc_mean_within_subset_seed_sd
    overall['max_within_subset_seed_sd'] = overall.balanced_acc_max_within_subset_seed_sd
    overall['insufficient_training_seeds'] = overall.min_seed_count.lt(3)
    overall['seed_sd_over_0_5pp'] = overall.max_within_subset_seed_sd.gt(.005)
    metadata = _metadata(per_seed, overall_keys)
    if metadata is not None:
        overall = overall.merge(metadata, on=overall_keys, validate='one_to_one')
    return per_seed, within, overall


def _output_directory(out, filenames):
    out = Path(out).resolve()
    if out == (PROJECT / 'data').resolve() or (PROJECT / 'data').resolve() in out.parents:
        raise ValueError('原始data目录不可写入报告')
    out.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in filenames if (out / name).exists()]
    if occupied:
        raise FileExistsError(f'报告输出已存在，禁止覆盖: {occupied}')
    return out


def _save_summaries(out, per_seed, within, overall):
    per_seed.to_csv(out / 'per_seed_metrics.csv', index=False)
    within.to_csv(out / 'per_subset_mean_std.csv', index=False)
    overall.to_csv(out / 'hierarchical_summary.csv', index=False)


def _display_table(overall):
    cols = ['config_id', 'variant', 'domain', 'ratio', 'protocol', 'source_load_hp',
            'target_load_hp', 'condition', 'balanced_acc_mean', 'subset_mean_sd',
            'mean_within_subset_seed_sd', 'max_within_subset_seed_sd',
            'subset_count', 'min_seed_count', 'acc_mean', 'recall_OR_mean', 'recall_B_mean']
    table = overall[cols].copy()
    for column in ['balanced_acc_mean', 'subset_mean_sd', 'mean_within_subset_seed_sd',
                   'max_within_subset_seed_sd', 'acc_mean', 'recall_OR_mean', 'recall_B_mean']:
        table[column] *= 100
    return table.round(3).to_markdown(index=False)


def _label(row):
    label = LABELS.get(row['variant'], row['variant'])
    if row['variant'] == 'data_aux_ce' and 'aux_weight' in row and pd.notna(row['aux_weight']):
        label += f" λ={row['aux_weight']:g}"
        label += ' 混合标准化' if row.get('physics_fit') == 'matched_noise' else ''
        label += ' 独立头' if row.get('aux_head') == 'independent' else ''
    if pd.notna(row.get('val_noise_count', np.nan)):
        label += f" V{int(row['val_noise_count'])}"
    if pd.notna(row.get('epochs_limit', np.nan)) and float(row['epochs_limit']) != 100:
        label += f" E{int(row['epochs_limit'])}"
    return label


def _figures(out, overall):
    """分别显示子集间差异和子集内种子波动，不合并成一种不确定性。"""
    dest = out / 'figures'
    dest.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    paths = []
    keys = ['domain', 'ratio', 'protocol', 'source_load_hp', 'target_load_hp']
    for key, frame in overall.groupby(keys, dropna=False):
        domain, ratio, protocol, source, target = key
        conditions = [c for c in CONDITIONS if c in set(frame.condition)]
        if not conditions:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
        for _, group in frame.groupby('config_id', sort=True):
            group = group.set_index('condition').reindex(conditions)
            valid = group.balanced_acc_mean.notna()
            if not valid.any():
                continue
            positions = np.flatnonzero(valid.to_numpy())
            g = group.loc[valid]
            first = g.iloc[0]
            means = g.balanced_acc_mean.to_numpy(dtype=float) * 100
            for axis, field in zip(axes, ['subset_mean_sd', 'mean_within_subset_seed_sd']):
                error = g[field].to_numpy(dtype=float) * 100
                if np.isfinite(error).any():
                    axis.errorbar(positions, means, yerr=np.nan_to_num(error, nan=0),
                                  marker='o', capsize=3, label=_label(first))
                else:
                    axis.plot(positions, means, marker='o', label=_label(first))
        for axis, title in zip(axes, ['子集均值之间的SD', '固定子集内种子SD的平均值']):
            axis.set(xlabel='测试条件' if out.name == 'evaluation' else '评价条件',
                     ylabel='均衡准确率 (%)', xticks=range(len(conditions)),
                     xticklabels=['干净' if c == 'clean' else c.replace('white_', '') + ' dB' for c in conditions],
                     ylim=(0, 102), title=title)
            axis.grid(alpha=.2)
            axis.tick_params(axis='x', rotation=20)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.01),
                      ncol=1 if len(labels) > 6 else 2, fontsize=8)
        figure.suptitle(f'{domain} | {protocol} | 训练比例 {100 * float(ratio):g}% | {source} HP 至 {target} HP')
        figure.text(.5, .005, '噪声实例先在模型内平均。左图单子集无SD；右图不是总体置信区间。',
                    ha='center', fontsize=9)
        figure.tight_layout(rect=(0, .035, 1, .94))
        stem = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{domain}_r{ratio}_{protocol}_s{source}_t{target}')
        for ext in ['png', 'pdf']:
            figure.savefig(dest / f'{stem}.{ext}', dpi=160, bbox_inches='tight')
        paths.append(f'figures/{stem}.png')
        plt.close(figure)
    return paths


def _lambda_figures(out, overall):
    """每个预算单独画lambda曲线，仅合并其他设置完全相同的配置。"""
    if 'aux_weight' not in overall:
        return []
    frame = overall[overall.variant.eq('data_aux_ce') & overall.aux_weight.gt(0)].copy()
    if frame.empty:
        return []
    keys = ['domain', 'ratio', 'protocol', 'source_load_hp', 'target_load_hp']
    keys += [k for k in ['physics_fit', 'aux_head', 'val_noise_count', 'epochs_limit'] if k in frame]
    paths = []
    for key, group in frame.groupby(keys, dropna=False):
        if group.aux_weight.nunique() < 2:
            continue
        settings = dict(zip(keys, key))
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for condition in CONDITIONS:
            data = group[group.condition.eq(condition)].sort_values('aux_weight')
            if data.empty:
                continue
            if data.aux_weight.duplicated().any():
                raise ValueError('lambda敏感性图同一权重包含多个未区分配置')
            label = '干净' if condition == 'clean' else condition.replace('white_', '') + ' dB'
            mean = data.balanced_acc_mean.to_numpy(float) * 100
            seed_sd = data.mean_within_subset_seed_sd.to_numpy(float) * 100
            x = data.aux_weight.to_numpy(float)
            line = axes[0].errorbar(x, mean, yerr=np.nan_to_num(seed_sd, nan=0.),
                                    marker='o', capsize=3, label=label)
            axes[1].plot(x, data.max_within_subset_seed_sd.to_numpy(float) * 100,
                         marker='o', color=line.lines[0].get_color(), label=label)
        lambdas = sorted(group.aux_weight.unique())
        for axis in axes:
            axis.set_xscale('log')
            axis.set_xticks(lambdas, labels=[f'{value:g}' for value in lambdas])
            axis.set_xlabel('辅助分类系数 λ（对数间距）')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        axes[0].set(ylabel='均衡准确率 (%)', title='均值及子集内种子SD的平均值')
        axes[1].set(ylabel='标准差（百分点）', title='各固定子集的最大种子SD', ylim=(0, None))
        mean = group.balanced_acc_mean.to_numpy(float) * 100
        error = np.nan_to_num(group.mean_within_subset_seed_sd.to_numpy(float) * 100, nan=0.)
        axes[0].set_ylim(max(0., float(np.min(mean - error)) - 3), min(102., float(np.max(mean + error)) + 3))
        title = f"λ敏感性 | {settings['protocol']} | 训练 {100 * float(settings['ratio']):g}%"
        if settings['domain'] != 'window_random':
            title += f" | {settings['source_load_hp']} 至 {settings['target_load_hp']} HP"
        figure.suptitle(title)
        figure.text(.5, .012, '每模型先平均噪声实例。误差棒为种子SD的子集平均，不是总体置信区间；不同预算分别作图。',
                    ha='center', fontsize=8)
        figure.tight_layout(rect=(0, .045, 1, .94))
        stem = 'lambda_' + '_'.join(str(value) for value in key)
        stem = re.sub(r'[^A-Za-z0-9_.-]', '_', stem)
        for ext in ['png', 'pdf']:
            figure.savefig(out / 'figures' / f'{stem}.{ext}', dpi=160, bbox_inches='tight')
        paths.append(f'figures/{stem}.png')
        plt.close(figure)
    return paths


def _pair_differences(per_seed):
    candidates = per_seed[per_seed.variant.eq('data_aux_ce')]
    columns = ['config_id', 'control_config_id'] + PAIR_KEYS + [f'{metric}_gain' for metric in METRICS]
    if candidates.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    if 'control_config_id' not in candidates or candidates.control_config_id.isna().any():
        raise ValueError('融合配置缺少明确的数据单分支对照control_config_id')
    rows = []
    for (config, control), candidate in candidates.groupby(['config_id', 'control_config_id']):
        reference = per_seed[per_seed.config_id.eq(control) & per_seed.variant.eq('baseline_data')]
        merged = candidate[PAIR_KEYS + METRICS].merge(reference[PAIR_KEYS + METRICS],
                    on=PAIR_KEYS, how='left', validate='one_to_one', suffixes=('_candidate', '_control'), indicator=True)
        if not merged['_merge'].eq('both').all():
            raise ValueError(f'{config}缺少条件匹配的数据单分支对照{control}')
        part = merged[PAIR_KEYS].copy()
        part['config_id'], part['control_config_id'] = config, control
        for metric in METRICS:
            part[f'{metric}_gain'] = merged[f'{metric}_candidate'] - merged[f'{metric}_control']
        rows.append(part)
    paired = pd.concat(rows, ignore_index=True)[columns]
    keys = ['config_id', 'control_config_id'] + [k for k in PAIR_KEYS if k != 'seed']
    within = paired.groupby(keys, dropna=False).balanced_acc_gain.agg(['mean', 'std', 'count']).reset_index()
    within = within.rename(columns={'mean': 'gain_mean', 'std': 'gain_seed_sd', 'count': 'paired_seed_count'})
    keys = [k for k in keys if k != 'subset']
    summary = within.groupby(keys, dropna=False).agg(
        gain_mean=('gain_mean', 'mean'), between_subset_gain_sd=('gain_mean', 'std'),
        mean_within_subset_gain_seed_sd=('gain_seed_sd', 'mean'),
        max_within_subset_gain_seed_sd=('gain_seed_sd', 'max'),
        subset_count=('subset', 'nunique'), min_paired_seed_count=('paired_seed_count', 'min')).reset_index()
    summary['paired_seed_sd_over_0_5pp'] = summary.max_within_subset_gain_seed_sd.gt(.005)
    summary['insufficient_training_seeds'] = summary.min_paired_seed_count.lt(3)
    return paired, summary


def write_validation_report(out, metrics, decision):
    """调用方必须传入固定验证集指标；本函数从不打开测试预测。"""
    out = _output_directory(out, ['VALIDATION_REPORT.md', 'per_seed_metrics.csv',
                                'per_subset_mean_std.csv', 'hierarchical_summary.csv', 'figures'])
    per_seed, within, overall = summarize_metrics(metrics)
    _save_summaries(out, per_seed, within, overall)
    plots = _figures(out, overall) + _lambda_figures(out, overall)
    lines = ['# 阶段验证报告', '',
             '以下仅为固定验证集结果。完成计算、采用改动、进入下一阶段是不同状态，以阶段决策为准。', '',
             '## 已锁定的阶段决策', '', '```json',
             json.dumps(decision, ensure_ascii=False, indent=2, default=str), '```', '',
             '## 验证指标', '',
             '均值先在每模型内平均噪声实例，再在固定子集内平均训练种子，最后对子集等权平均。'
             '表中数值为百分数或百分点。子集间SD与种子SD分别报告，不能合并成独立数据集重复。'
             '一个子集的子集间SD为NaN；少于三个训练种子的分组明确标记为不足。', '',
             _display_table(overall), '',
             '固定验证集被反复用于选择配置，结果属于探索性分析。2026-10-21用户已取消SD不超过0.5个百分点的硬门槛。'
             '表中的seed_sd_over_0_5pp仅为描述性波动警示，不参与采用、停止或扩展判定，也不是显著性检验。']
    for path in plots:
        lines += ['', f'![验证结果]({path})']
    (out / 'VALIDATION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return overall


def _confusions(metrics, evaluations):
    evaluation_keys = ['job_id', 'source_load_hp', 'target_load_hp']
    if evaluations.duplicated(evaluation_keys).any():
        raise ValueError('同一模型目标重复登记')
    expected = metrics[evaluation_keys].drop_duplicates()
    actual = evaluations[evaluation_keys].drop_duplicates()
    if len(expected.merge(actual, on=evaluation_keys, how='outer', indicator=True).query('_merge != "both"')):
        raise ValueError('指标与评价登记中的模型目标不一致')
    rows = []
    support_signatures = {}
    pool_keys = ['config_id', 'variant', 'domain', 'ratio', 'protocol',
                 'source_load_hp', 'target_load_hp', 'condition']
    for evaluation in evaluations.to_dict('records'):
        mask = np.ones(len(metrics), dtype=bool)
        for key in evaluation_keys:
            mask &= metrics[key].eq(evaluation[key]).to_numpy()
        matched = metrics.loc[mask]
        with np.load(resolve_result_path(evaluation['prediction_path']), allow_pickle=False) as saved:
            labels = np.asarray(saved['labels'])
            indices = np.asarray(saved['window_indices'])
            if labels.ndim != 1 or indices.shape != labels.shape or len(set(indices.tolist())) != len(indices):
                raise ValueError('预测中的窗口编号不唯一或与标签长度不符')
            if not np.isin(labels, [0, 1, 2, 3]).all() or not np.equal(labels, labels.astype(int)).all():
                raise ValueError('预测标签必须为0至3的整数')
            labels = labels.astype(int)
            if int(evaluation['test_windows']) != len(labels):
                raise ValueError('登记的测试窗数量与保存预测不一致')
            supports = np.bincount(labels, minlength=4)
            if np.any(supports == 0):
                raise ValueError('正式四类评价要求每个真实类别均有测试支持')
            for row in matched.to_dict('records'):
                prob = np.asarray(saved[f"{row['condition']}_n{int(row['noise_seed'])}"])
                if prob.shape != (len(labels), 4) or not np.isfinite(prob).all():
                    raise ValueError('预测概率形状或数值异常')
                if np.any(prob < -1e-7) or np.any(prob > 1 + 1e-7):
                    raise ValueError('预测概率超出[0,1]')
                np.testing.assert_allclose(prob.sum(1), 1., rtol=1e-4, atol=1e-5)
                prediction = prob.argmax(1)
                matrix = np.bincount(labels * 4 + prediction, minlength=16).reshape(4, 4)
                recalls = matrix.diagonal() / supports
                np.testing.assert_allclose(recalls, [row[k] for k in METRICS[3:]], rtol=0, atol=1e-10)
                np.testing.assert_allclose(recalls.mean(), row['balanced_acc'], rtol=0, atol=1e-10)
                np.testing.assert_allclose((prediction == labels).mean(), row['acc'], rtol=0, atol=1e-10)
                group = tuple(row[k] for k in pool_keys)
                signature = (tuple(indices.tolist()), tuple(labels.tolist()))
                if group in support_signatures and support_signatures[group] != signature:
                    raise ValueError('聚合混淆矩阵的模型使用了不同测试窗口，不可直接合并支持数')
                support_signatures[group] = signature
                for true in range(4):
                    for pred in range(4):
                        rows.append({**{k: row[k] for k in MODEL_KEYS}, 'job_id': row['job_id'],
                                     'noise_seed': row['noise_seed'], 'true': true, 'predicted': pred,
                                     'count': int(matrix[true, pred]), 'original_support': int(supports[true]),
                                     'row_fraction': float(matrix[true, pred] / supports[true])})
    raw = pd.DataFrame(rows)
    by_model = raw.groupby(MODEL_KEYS + ['true', 'predicted'], dropna=False).agg(
        row_fraction=('row_fraction', 'mean'), count=('count', 'sum'),
        original_support=('original_support', 'first'), noise_instances=('noise_seed', 'nunique')).reset_index()
    within_keys = [k for k in MODEL_KEYS if k != 'seed'] + ['true', 'predicted']
    within = by_model.groupby(within_keys, dropna=False).agg(
        row_fraction=('row_fraction', 'mean'), count=('count', 'sum'),
        original_support=('original_support', 'first'), model_count=('seed', 'nunique'),
        min_noise_instances=('noise_instances', 'min'), max_noise_instances=('noise_instances', 'max')).reset_index()
    final_keys = [k for k in within_keys if k != 'subset']
    pooled = within.groupby(final_keys, dropna=False).agg(
        row_fraction=('row_fraction', 'mean'), count=('count', 'sum'),
        original_support=('original_support', 'first'), model_count=('model_count', 'sum'),
        subset_count=('subset', 'nunique'), min_noise_instances=('min_noise_instances', 'min'),
        max_noise_instances=('max_noise_instances', 'max')).reset_index()
    row_keys = [k for k in final_keys if k != 'predicted']
    pooled['pooled_row_fraction'] = pooled['count'] / pooled.groupby(row_keys, dropna=False)['count'].transform('sum')
    check = pooled.groupby(row_keys, dropna=False).row_fraction.sum()
    np.testing.assert_allclose(check, 1., rtol=0, atol=1e-10)
    return pooled


def _compute_summary(evaluations):
    """同一来源模型用于三个目标时，其训练成本只计一次。"""
    jobs = evaluations.drop_duplicates('job_id').copy()
    groups = ['config_id', 'variant', 'domain', 'ratio', 'protocol']
    columns = ['parameters', 'inference_parameters', 'auxiliary_only_parameters', 'conv_linear_flops', 'epochs_trained',
               'updates_trained', 'fit_seconds', 'total_train_seconds', 'train_seconds_per_epoch',
               'validation_seconds', 'gradient_diagnostic_seconds',
               'total_training_with_diagnostics_seconds', 'peak_gpu_mib', 'latency_ms']
    columns = [c for c in columns if c in jobs]
    for column in columns:
        jobs[column] = pd.to_numeric(jobs[column], errors='coerce')
    result = jobs.groupby(groups, dropna=False).job_id.nunique().rename('model_count').reset_index()
    if columns:
        means = jobs.groupby(groups, dropna=False)[columns].mean().reset_index()
        result = result.merge(means, on=groups, validate='one_to_one')
    return result


def write_report(out):
    """只读统一测试的保存预测；不重新前向、不改变selection.json中的选择。"""
    out = _output_directory(out, ['REPORT.md', 'per_seed_metrics.csv', 'per_subset_mean_std.csv',
                                'hierarchical_summary.csv', 'confusion_counts.csv',
                                'paired_test_differences.csv', 'paired_summary.csv', 'compute_summary.csv', 'figures'])
    metrics = _validate_metrics(pd.read_csv(out / 'metrics.csv'))
    evaluations = pd.read_csv(out / 'evaluations.csv')
    selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
    is_partial = selection.get('partial_test') is True and selection.get('cohort_locked') is True
    if selection.get('decisions_locked') is not True and not is_partial:
        raise ValueError('测试前必须先锁定阶段决策')
    per_seed, within, overall = summarize_metrics(metrics)
    if 'control_config_id' not in per_seed:
        mapping = evaluations[['job_id', 'control_config_id']].drop_duplicates()
        per_seed = per_seed.merge(mapping, on='job_id', validate='many_to_one')
    paired, pair_summary = _pair_differences(per_seed)
    confusion = _confusions(metrics, evaluations)
    cost = _compute_summary(evaluations)
    _save_summaries(out, per_seed, within, overall)
    paired.to_csv(out / 'paired_test_differences.csv', index=False)
    pair_summary.to_csv(out / 'paired_summary.csv', index=False)
    confusion.to_csv(out / 'confusion_counts.csv', index=False)
    cost.to_csv(out / 'compute_summary.csv', index=False)
    plots = _figures(out, overall) + _lambda_figures(out, overall)
    lines = ['# 分阶段优化统一测试报告', '',
             f"本报告覆盖{evaluations.job_id.nunique()}个配置模型登记、{evaluations.folder.nunique()}个不同权重目录、"
             f"{len(evaluations)}组模型目标评价、{len(metrics)}行逐噪声实例指标。"
             '配置中未改变干净协议行为的情况会复用同一权重，并包含只读引用的历史模型，不能全部称为新增训练。', '',
             '候选是否采用由测试前锁定的验证决策决定。以下完整保留已评价候选及失败结果，不按测试表现重新筛选。', '',
             '## 锁定方案', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
             '## 测试结果', '',
             '主指标为均衡准确率。噪声实例先在模型内平均，再在固定子集内计算种子均值及样本SD，最后对子集等权平均。'
             'subset_mean_sd为子集均值之间的SD；mean_within_subset_seed_sd为各子集种子SD的平均，二者不合并。'
             '表中数值为百分数或百分点，一个子集时子集间SD为NaN。', '', _display_table(overall), '',
             '## 配对比较', '',
             '每个融合配置仅与预先登记的数据单分支配置比较，逐子集、种子、目标和噪声条件匹配。'
             '逐配对差值保存在paired_test_differences.csv；paired_summary.csv中的增益与SD使用比例单位，乘100后为百分点。', '',
             pair_summary.round(6).to_markdown(index=False) if not pair_summary.empty else '本次无融合配对结果。', '',
             '## 混淆矩阵', '',
             'confusion_counts.csv保留原始累计预测计数count。original_support为原始测试窗口数，未乘种子、子集或噪声重复数。'
             'row_fraction先平均噪声，再平均种子和子集，各真实类别行和为1。pooled_row_fraction另提供累计计数归一化结果。'
             '当重复数不相等时，两种比例可能不同，主报告采用前者。', '',
             '## 计算成本', '', cost.round(4).to_markdown(index=False), '',
             '训练成本按唯一job_id统计；同一来源模型评价三个目标不重复计训练量。'
             '参数与Conv1d/Linear FLOPs仅描述登记口径，FLOPs不含物理FFT、归一化和逐元素操作。'
             '单样本延迟、峰值显存、拟合与验证耗时取自训练引擎的实测记录，不在本报告中重新推理。'
             '具体硬件、精度、批量、预热、重复次数和计时范围见各模型parameters.json。', '',
             '## 证据边界', '',
             '仅保留数据单分支与融合加辅助分类两种方法。两组比较不能单独识别物理融合与辅助损失各自贡献。'
             '超参数搜索次数及候选失败均需披露。验证集被反复用于选参，且测试已有历史曝光，结果为探索性证据。'
             '训练与测试共享记录，训练子集、随机种子和噪声重复不能当作独立轴承或独立数据集。'
             '若跨负载系数来源于包含全部负载的随机窗口开发，不能声称目标负载从未参与方法开发。'
             '2026-10-21用户已取消SD不超过0.5个百分点的硬门槛。SD和配对增益SD仅描述波动，'
             '相关over_0_5pp字段作为警示，不参与采用或停止判定，也不是显著性检验。'
             '少于三个种子的分组明确标记，不能据此声称已充分验证稳定性。']
    for path in plots:
        lines += ['', f'![统一测试结果]({path})']
    if is_partial:
        lines[0] = '# 两系数阶段性测试报告'
        lines[2:2] = ['本报告由用户明确授权提前测试lambda=0.01、0.03及匹配数据单分支。'
                      '只锁定本批评价对象和已选权重，全系数网格及后续优化决策尚未完成。'
                      '本报告不产生系数采用决策。后续训练仍使用预设验证规则，必须披露本次提前测试曝光。', '']
        lines = [line.replace('候选是否采用由测试前锁定的验证决策决定。以下完整保留已评价候选及失败结果，不按测试表现重新筛选。',
                 '本批不选择最优系数，不按测试表现淘汰或增加后续候选。已完成的两系数及对照全部报告。') for line in lines]
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return overall
