"""从保存的预测复算 V5 纯干净和 AWGN 训练模型的跨负载结果。"""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from metrics import LABELS, METRICS, scores

from . import v5_report as clean_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path



CLEAN_REFERENCE = clean_report.PROJECT / 'experiments/CWRU/实验结果/CWRU_V5/load_clean_20261109_142434'
ALLOCATIONS = clean_report.ALLOCATIONS
BUDGETS = (1., .1, .2, .5)
EARLIER_BUDGETS = (1., .1)
REMAINING_BUDGETS = (.2, .5)
SOURCES = clean_report.SOURCES
SEEDS = clean_report.SEEDS
SUBSETS = clean_report.SUBSETS
META = clean_report.META
CASE = clean_report.CASE
PAIR = clean_report.PAIR
METRIC_NAMES = clean_report.METRIC_NAMES
CONDITIONS = ('clean', 'white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4')
NOISE_SEEDS = tuple(range(7201, 7206))
CLEAN_CONDITION = (('clean', 0),)
NOISE_CONDITIONS = tuple((condition, seed) for condition in CONDITIONS[1:] for seed in NOISE_SEEDS)
ALL_CONDITIONS = CLEAN_CONDITION + NOISE_CONDITIONS
FILE_PAIR = ['job_id', 'target_load_hp']
ROW_KEY = [*FILE_PAIR, 'condition', 'noise_seed']
CELL_KEY = [*ROW_KEY, 'true_label', 'predicted_label']


def _jobs(jobs, allocations, protocol, variants):
    expected = {(ratio, subset, source, seed, variant)
                for ratio, subset in allocations for source in SOURCES
                for seed in SEEDS for variant in variants}
    if (len(jobs) != len(expected) or jobs.job_id.duplicated().any()
            or set(jobs[[*CASE, 'variant']].itertuples(index=False, name=None)) != expected
            or not (jobs.completed.eq(True) & jobs.from_scratch.eq(True) & jobs.protocol.eq(protocol)
                    & jobs.domain.eq('load_source_' + jobs.source_load_hp.astype(str))
                    & jobs.aux_weight.eq(jobs.variant.map({'baseline_data': 0., 'data_aux_ce': .3}))).all()):
        raise ValueError(f'{protocol} 模型组合、模型状态或系数与预定设计不符')


def _evaluation_index(root, jobs, evaluations):
    expected_pairs = {(row.job_id, target) for row in jobs.itertuples(index=False)
                      for target in SOURCES if target != row.source_load_hp}
    if (len(evaluations) != len(expected_pairs)
            or set(evaluations[FILE_PAIR].itertuples(index=False, name=None)) != expected_pairs):
        raise ValueError(f'{root.name} 模型目标组合不完整或重复')
    joined = evaluations.merge(jobs[['job_id', *META]], on='job_id', validate='many_to_one',
                               suffixes=('_result', '_job'))
    if any(not joined[f'{key}_result'].eq(joined[f'{key}_job']).all() for key in META):
        raise ValueError(f'{root.name} 评价元数据与模型登记不符')
    directory = (root / 'evaluation' / 'predictions').resolve()
    paths = [resolve_result_path(path).resolve() for path in evaluations.prediction_path]
    if len(paths) != len(set(paths)) or any(path.parent != directory or not path.is_file() for path in paths):
        raise ValueError(f'{root.name} 预测路径越界、重复或缺失')


def _frozen_v3():
    entries = []
    for root, allocations in ((clean_report.FULL_REFERENCE, clean_report.FULL),
                              (clean_report.SMALL_REFERENCE, clean_report.SMALL)):
        state = clean_report._state(root)
        if (state['stage'] != 'complete' or state['decisions_locked'] is not True
                or state['target_test_accessed'] is not True or state['selected_aux_weight'] != .3):
            raise ValueError(f'{root.name} 的冻结状态或系数不符')
        jobs = pd.read_csv(root / 'models_registry.csv')
        jobs = jobs[jobs.protocol.eq('awgn_mixed')]
        _jobs(jobs, allocations, 'awgn_mixed', clean_report.V3_VARIANTS)
        evaluations = pd.read_csv(root / 'evaluation' / 'evaluations.csv')
        evaluations = evaluations[evaluations.protocol.eq('awgn_mixed')]
        _evaluation_index(root, jobs, evaluations)
        for filename, fields in (('metrics.csv', METRICS),
                                 ('confusion_matrices.csv', ['true_label', 'predicted_label', 'count', 'support', 'row_percent'])):
            required = {'job_id', 'target_load_hp', 'condition', 'noise_seed', *META, *fields}
            if not required.issubset(pd.read_csv(root / 'evaluation' / filename, nrows=0).columns):
                raise ValueError(f'{root.name} {filename} 缺少复算字段')
        entries.append((root, allocations, jobs, evaluations))
    return entries


def _frozen_v5():
    state = clean_report._state(CLEAN_REFERENCE)
    if (state['stage'] != 'complete' or state['decisions_locked'] is not True
            or state['weights_verified'] is not True or state['target_test_accessed'] is not True
            or state['selected_aux_weight'] != .3 or state['trained_models'] != 200 or state['evaluated_pairs'] != 600):
        raise ValueError('V5 纯干净对照权重或评估状态不符')
    jobs, evaluations = clean_report._read_structure(CLEAN_REFERENCE, ALLOCATIONS, .3)
    model_dir = (CLEAN_REFERENCE / 'models').resolve()
    if any(folder.parent != model_dir or folder.name != job_id or not (folder / 'model.pt').is_file()
           for job_id, folder in zip(jobs.job_id, map(lambda path: resolve_result_path(path).resolve(), jobs.folder), strict=True)):
        raise ValueError('V5 纯干净对照权重缺失或路径不符')
    return jobs, evaluations


def preflight():
    """在训练前核对冻结对照状态、权重路径和评价结构，不打开预测概率。"""
    jobs, evaluations = _frozen_v5()
    refs = _frozen_v3()
    return dict(v5_clean_models=len(jobs), v5_clean_pairs=len(evaluations),
                v3_awgn_models=sum(len(models) for _, _, models, _ in refs),
                v3_awgn_pairs=sum(len(index) for _, _, _, index in refs))


def _verify_predictions(root, jobs, evaluations, conditions):
    """逐保存概率复算七项指标和混淆矩阵，校验对应的固定窗口。"""
    _evaluation_index(root, jobs, evaluations)
    directory = root / 'evaluation'
    keys = set(jobs.job_id)
    metrics = pd.read_csv(directory / 'metrics.csv')
    metrics = metrics[metrics.job_id.isin(keys)].sort_values(ROW_KEY).reset_index(drop=True)
    confusion = pd.read_csv(directory / 'confusion_matrices.csv')
    confusion = confusion[confusion.job_id.isin(keys)].sort_values(CELL_KEY).reset_index(drop=True)
    if (len(metrics) != len(evaluations) * len(conditions)
            or len(confusion) != len(metrics) * len(LABELS)**2
            or metrics.duplicated(ROW_KEY).any() or confusion.duplicated(CELL_KEY).any()
            or not confusion[ROW_KEY].iloc[::16].reset_index(drop=True).equals(metrics[ROW_KEY])):
        raise ValueError(f'{root.name} 的评价条件或混淆矩阵不完整')
    for frame in (metrics, confusion):
        joined = frame[['job_id', *META]].drop_duplicates().merge(jobs[['job_id', *META]],
                     on='job_id', validate='many_to_one', suffixes=('_result', '_job'))
        if any(not joined[f'{key}_result'].eq(joined[f'{key}_job']).all() for key in META):
            raise ValueError(f'{root.name} 的指标元数据与模型登记不符')
    cells = confusion[['true_label', 'predicted_label']].to_numpy().reshape(-1, 16, 2)
    cell_order = np.array([(a, b) for a in range(4) for b in range(4)])
    if not np.all(cells == cell_order):
        raise ValueError('混淆矩阵类别次序或个数异常')
    saved_values = metrics[METRICS].to_numpy(float)
    saved_cells = confusion[['count', 'support', 'row_percent']].to_numpy(float).reshape(-1, 16, 3)
    row_groups = metrics.groupby(FILE_PAIR, sort=False).indices
    windows = {}
    expected_conditions = set(conditions)
    for row in evaluations.itertuples(index=False):
        indices = row_groups[(row.job_id, row.target_load_hp)]
        items = list(metrics.loc[indices, ['condition', 'noise_seed']].itertuples(index=False, name=None))
        if set(items) != expected_conditions:
            raise ValueError(f'{row.job_id} 的噪声条件或种子缺失')
        split = (row.ratio, row.subset, row.source_load_hp, row.target_load_hp)
        if split not in windows:
            table = pd.read_csv(clean_report._split_path(root, *split[:3]))
            expected = table[table.load_hp.eq(row.target_load_hp)]
            windows[split] = expected.label.to_numpy(), expected.window_index.to_numpy()
        labels, window_ids = windows[split]
        if len(labels) != row.test_windows:
            raise ValueError(f'{row.job_id} 的固定目标窗口数不符')
        names = [f'{condition}_n{seed}' for condition, seed in items]
        with np.load(resolve_result_path(row.prediction_path), allow_pickle=False) as prediction:
            if set(prediction.files) != {'labels', 'window_indices', *names}:
                raise ValueError(f'{row.job_id} 的保存预测条件不符')
            np.testing.assert_array_equal(prediction['labels'], labels)
            np.testing.assert_array_equal(prediction['window_indices'], window_ids)
            for index, name in zip(indices, names, strict=True):
                probability = prediction[name]
                if (probability.shape != (len(labels), len(LABELS)) or not np.isfinite(probability).all()
                        or (probability < 0).any() or not np.allclose(probability.sum(1), 1., atol=1e-5, rtol=0)):
                    raise ValueError(f'{row.job_id}/{name} 保存的概率无效')
                values, counts = scores(labels, probability)
                expected_values = [values[metric] for metric in METRICS]
                support = np.repeat(counts.sum(1), 4)
                expected_cells = np.stack((counts.ravel(), support, 100 * counts.ravel() / support), axis=-1)
                if (not np.allclose(saved_values[index], expected_values, atol=1e-10, rtol=0)
                        or not np.allclose(saved_cells[index], expected_cells, atol=1e-9, rtol=0)):
                    raise ValueError(f'{row.job_id}/{row.target_load_hp}/{name} 的保存指标或混淆计数不一致')
    return metrics


METHODS = {'v5_clean': 'V5 39维纯干净训练', 'v5_awgn': 'V5 39维混合训练',
           'v3_212': 'V3 212维混合训练', 'v3_data': 'V3 数据单分支混合训练'}
CONDITION_NAMES = dict(zip(CONDITIONS, ('干净', '−4 dB', '−2 dB', '0 dB', '+2 dB', '+4 dB'), strict=True))


def _aggregate(frame, group):
    directions = ['ratio', 'subset', 'source_load_hp', 'target_load_hp', 'condition', group]
    by_direction = clean_report._summary(frame, directions, len(SEEDS))
    direction_groups = frame.groupby(['ratio', 'subset', 'condition', group, 'seed'], as_index=False)
    if not direction_groups.size()['size'].eq(12).all():
        raise ValueError('每个种子及子集必须覆盖 12 个方向')
    by_seed = direction_groups[METRICS].mean()
    budget_seed = by_seed.groupby(['ratio', 'condition', group, 'seed'], as_index=False)[METRICS].mean()
    budget = clean_report._summary(budget_seed, ['ratio', 'condition', group], len(SEEDS))
    subset_mean = by_seed[by_seed.ratio.lt(1)].groupby(['ratio', 'subset', 'condition', group], as_index=False)[METRICS].mean()
    between = clean_report._summary(subset_mean, ['ratio', 'condition', group], len(SUBSETS))
    return by_direction, budget, between


def _display(frame, group, fields, delta=False):
    rows = []
    for ratio in [ratio for ratio in BUDGETS if ratio in set(frame.ratio)]:
        for condition in CONDITIONS:
            for record in frame[frame.ratio.eq(ratio) & frame.condition.eq(condition)].to_dict('records'):
                row = {'训练预算': '全样本' if ratio == 1 else f'{ratio:.0%}',
                       '目标条件': CONDITION_NAMES[condition], '比较组': METHODS[record[group]]}
                for metric in fields:
                    mean, std = (100 * record[f'{metric}_{stat}'] for stat in ('mean', 'std'))
                    row[METRIC_NAMES[METRICS.index(metric)]] = f'{mean:+.2f} ± {std:.2f}' if delta else f'{mean:.2f} ± {std:.2f}'
                rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def _markdown(budget, difference, direction, models=200, merged_sources=()):
    # 主表以 BA 表示四组曲线。详细指标和每方向不确定性保存在 CSV。
    ba = budget.pivot(index=['ratio', 'condition'], columns='method', values='balanced_acc_mean')
    rows = []
    for ratio in [ratio for ratio in BUDGETS if ratio in set(budget.ratio)]:
        for condition in CONDITIONS:
            rows.append({'训练预算': '全样本' if ratio == 1 else f'{ratio:.0%}',
                         '目标条件': CONDITION_NAMES[condition],
                         **{title: f'{100 * ba.loc[(ratio, condition), method]:.2f}' for method, title in METHODS.items()}})
    critical = direction[direction.source_load_hp.eq(3) & direction.target_load_hp.eq(0)
                         & direction.condition.isin(['clean', 'white_-4'])]
    critical = critical.groupby(['ratio', 'condition', 'method'], as_index=False)[
        ['balanced_acc_mean', 'recall_B_mean', 'recall_OR_mean']].mean()
    critical[['balanced_acc_mean', 'recall_B_mean', 'recall_OR_mean']] *= 100
    critical['method'] = critical.method.map(METHODS)
    critical['condition'] = critical.condition.map(CONDITION_NAMES)
    critical['ratio'] = critical.ratio.map(lambda ratio: '全样本' if ratio == 1 else f'{ratio:.0%}')
    critical.columns = ['预算', '目标条件', '比较组', 'BA', '滚动体召回', '外圈召回']
    if merged_sources:
        earlier, remaining = merged_sources
        coverage = (f'本报告合并全样本与 10% 阶段 `{earlier}` 和 20% 与 50% 阶段 `{remaining}`。'
                    f'两阶段已分别从保存概率复算，共 {312 * models} 行指标及对应混淆计数。'
                    '本次合并只读取已核验指标，重新计算汇总与配对差。\n\n')
    else:
        coverage = f'报告生成前已从保存概率复算全部 {312 * models} 行指标及混淆计数。\n\n'
        if models == 80:
            coverage += '本次阶段报告仅覆盖全样本与 10% 档。\n\n'
        elif models == 120:
            coverage += '本次阶段报告仅覆盖 20% 与 50% 档。\n\n'
    return (
        '# CWRU V5 39 维 AWGN 跨负载实验\n\n'
        f'按已确认的 λ=0.3 从头训练 {models} 个混合训练模型，复用 {models} 个已核验的 V5 纯干净权重。'
        f'{3 * models} 个混合组目标组合各完成 26 次评价，{3 * models} 个纯干净组目标组合各补测 25 次噪声。'
        f'引用 V5 已保存的 {3 * models} 次干净预测和 V3 的 {156 * models} 次混合训练参照预测。'
        + coverage +
        '## 四组目标 BA\n\n'
        '单位为百分比。同一模型和 SNR 先平均五个固定噪声实例，再等权平均方向与固定子集，最后平均五个种子。\n\n'
        + pd.DataFrame(rows).to_markdown(index=False) + '\n\n'
        '## −4 dB 下 V5 混合训练相对配对参照\n\n'
        '数值为 V5 混合组减对应比较组的五种子均值 ± 样本标准差，单位为百分点。\n\n'
        + _display(difference[difference.condition.eq('white_-4')], 'reference', METRICS, delta=True) + '\n\n'
        '## 3 HP 到 0 HP 的干净与 −4 dB 结果\n\n'
        + critical.round(2).to_markdown(index=False) + '\n\n'
        '## 解释边界\n\n'
        'V5 混合训练同时改变训练观测、标准化统计和验证选模规则。对照差值评价该整套策略。'
        '固定目标记录已有历史测试曝光，完整来源验证集使用额外标签。五种子、噪声实例、固定子集及方向共享记录或模型。'
        '标准差描述本设计内的波动。结果适用于当前人工 AWGN、可用记录转速和本数据集。\n'
    )


def _phase(state, stage):
    budgets = tuple(state['ratios'])
    if budgets not in (EARLIER_BUDGETS, REMAINING_BUDGETS, BUDGETS):
        raise ValueError('不支持的跨负载 AWGN 预算范围')
    allocations = tuple(group for group in ALLOCATIONS if group[0] in budgets)
    models = len(allocations) * len(SOURCES) * len(SEEDS)
    if (state['stage'] != stage or state['selected_aux_weight'] != .3
            or not all(state[key] is True for key in ('weights_verified', 'clean_weights_verified',
                                                       'decisions_locked', 'target_test_accessed'))
            or state['trained_models'] != models or state['referenced_models'] != models
            or state['evaluated_pairs'] != 6 * models or state['metric_rows'] != 153 * models):
        raise ValueError('报告只接受全部权重核验和正式评价完成的 V5 AWGN 结果')
    return budgets, allocations


def finish(root):
    root = Path(root).resolve()
    budgets, allocations = _phase(clean_report._state(root), 'reporting')
    models = len(allocations) * len(SOURCES) * len(SEEDS)
    mixed = pd.read_csv(root / 'models_registry.csv')
    clean = pd.read_csv(root / 'referenced_models.csv')
    _jobs(mixed, allocations, 'awgn_mixed', ('data_aux_ce',))
    _jobs(clean, allocations, 'clean_only', ('data_aux_ce',))
    old_clean, old_clean_eval = _frozen_v5()
    old_clean = old_clean[old_clean.ratio.isin(budgets)]
    old_clean_eval = old_clean_eval[old_clean_eval.job_id.isin(old_clean.job_id)]
    refs = clean.rename(columns={'job_id': 'new_id', 'reference_job_id': 'job_id'})
    paired = refs.merge(old_clean, on='job_id', validate='one_to_one', suffixes=('_new', '_old'))
    if len(paired) != models or any(not paired[f'{key}_new'].eq(paired[f'{key}_old']).all() for key in (*META, 'folder')):
        raise ValueError('纯干净复用权重与冻结 V5 登记不一致')
    evaluations = pd.read_csv(root / 'evaluation' / 'evaluations.csv')
    if len(evaluations) != 6 * models or set(evaluations.job_id) != set(mixed.job_id) | set(clean.job_id):
        raise ValueError('新结果的模型目标组合不完整')
    new_groups = [(mixed, ALL_CONDITIONS, 'v5_awgn'), (clean, NOISE_CONDITIONS, 'v5_clean')]
    verified = [_verify_predictions(root, jobs, evaluations[evaluations.job_id.isin(jobs.job_id)], conditions)
                .assign(method=method, experiment=root.name) for jobs, conditions, method in new_groups]
    clean_report._same_splits(root, CLEAN_REFERENCE, allocations)
    verified.append(_verify_predictions(CLEAN_REFERENCE, old_clean, old_clean_eval, CLEAN_CONDITION)
                    .assign(method='v5_clean', experiment=CLEAN_REFERENCE.name))
    for reference, reference_allocations, jobs, index in _frozen_v3():
        in_scope = tuple(group for group in reference_allocations if group[0] in budgets)
        if not in_scope:
            continue
        clean_report._same_splits(root, reference, in_scope)
        jobs = jobs[jobs.ratio.isin(budgets)]
        index = index[index.job_id.isin(jobs.job_id)]
        metrics = _verify_predictions(reference, jobs, index, ALL_CONDITIONS)
        verified.append(metrics.assign(method=metrics.variant.map({'baseline_data': 'v3_data', 'data_aux_ce': 'v3_212'}),
                                       experiment=reference.name))
    outputs, report = _summaries(pd.concat(verified, ignore_index=True), allocations)
    _write_report(root / 'REPORT.md', root / 'evaluation', outputs, report)


def _check_metrics(raw, allocations):
    """要求每个预算、子集、方向、种子、噪声实例在四组中严格配对。"""
    keys = [*PAIR, 'condition', 'method', 'noise_seed']
    actual = pd.MultiIndex.from_frame(raw[keys])
    expected = pd.MultiIndex.from_tuples(
        [(ratio, subset, source, seed, target, condition, method, noise_seed)
         for ratio, subset in allocations for source in SOURCES for seed in SEEDS
         for target in SOURCES if target != source for condition, noise_seed in ALL_CONDITIONS
         for method in METHODS], names=keys)
    values = raw[METRICS].to_numpy(float)
    if (len(actual) != len(expected) or not actual.is_unique or not expected.difference(actual).empty
            or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()):
        raise ValueError('四组逐噪声实例的配对、范围或指标不符')


def _summaries(raw, allocations, merged_sources=()):
    _check_metrics(raw, allocations)
    models = len(allocations) * len(SOURCES) * len(SEEDS)
    keys = [*PAIR, 'condition', 'method']
    per_seed = raw.groupby(keys, as_index=False)[METRICS].mean()
    direction, budget, between = _aggregate(per_seed, 'method')
    wide = per_seed.pivot(index=[*PAIR, 'condition'], columns='method', values=METRICS)
    if len(wide) != 18 * models or wide.isna().any().any():
        raise ValueError('V5 与三个参照的配对不完整')
    candidate = wide.xs('v5_awgn', axis=1, level='method')
    differences = pd.concat([candidate.sub(wide.xs(method, axis=1, level='method')).reset_index().assign(reference=method)
                             for method in ('v5_clean', 'v3_212', 'v3_data')], ignore_index=True)
    pair_direction, pair_budget, pair_between = _aggregate(differences, 'reference')
    outputs = {'comparison_metrics.csv': raw, 'comparison_per_seed_snr.csv': per_seed,
               'comparison_direction_mean_std.csv': direction, 'comparison_budget_mean_std.csv': budget,
               'comparison_between_subset.csv': between, 'paired_seed_differences.csv': differences,
               'paired_direction_mean_std.csv': pair_direction, 'paired_budget_mean_std.csv': pair_budget,
               'paired_between_subset.csv': pair_between}
    return outputs, _markdown(budget, pair_budget, direction, models, merged_sources)


def _write_report(report_path, evaluation, outputs, report):
    paths = {evaluation / filename: frame for filename, frame in outputs.items()}
    for path in (report_path, *paths):
        if path.exists():
            raise FileExistsError(path)
    for path, frame in paths.items():
        frame.to_csv(path, index=False, mode='x')
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(report)


def _read_completed_metrics(root, allocations):
    frame = pd.read_csv(root / 'evaluation' / 'comparison_metrics.csv')
    _check_metrics(frame, allocations)
    expected_protocol = np.where(frame.method.eq('v5_clean'), 'clean_only', 'awgn_mixed')
    expected_variant = np.where(frame.method.eq('v3_data'), 'baseline_data', 'data_aux_ce')
    expected_experiment = np.where(frame.ratio.eq(1), clean_report.FULL_REFERENCE.name, clean_report.SMALL_REFERENCE.name)
    from_v5 = frame.method.isin(('v5_clean', 'v5_awgn'))
    expected_experiment = np.where(from_v5, root.name, expected_experiment)
    expected_experiment = np.where(frame.method.eq('v5_clean') & frame.condition.eq('clean'),
                                   CLEAN_REFERENCE.name, expected_experiment)
    if (not (root / 'REPORT.md').is_file()
            or not frame.domain.eq('load_source_' + frame.source_load_hp.astype(str)).all()
            or not frame.protocol.eq(expected_protocol).all() or not frame.variant.eq(expected_variant).all()
            or not frame.aux_weight.eq(np.where(frame.method.eq('v3_data'), 0., .3)).all()
            or not frame.experiment.eq(expected_experiment).all()):
        raise ValueError(f'{root.name} 的已核验指标与阶段来源不符')
    return frame


def combine(earlier, remaining):
    """只读合并两个已完成阶段的指标，生成全预算汇总。"""
    earlier, remaining = Path(earlier).resolve(), Path(remaining).resolve()
    old_state, new_state = clean_report._state(earlier), clean_report._state(remaining)
    old_budgets, old_allocations = _phase(old_state, 'complete')
    new_budgets, new_allocations = _phase(new_state, 'complete')
    if (old_budgets != EARLIER_BUDGETS or new_budgets != REMAINING_BUDGETS
            or old_state['test_snr_db'] != [-4, -2, 0, 2, 4]
            or tuple(old_state['test_noise_seeds']) != NOISE_SEEDS
            or any(old_state[key] != new_state[key] for key in ('model_config', 'training_spec', 'clean_reference',
                                                                'test_snr_db', 'test_noise_seeds', 'seeds', 'subsets'))):
        raise ValueError('两阶段预算或模型与噪声协议不一致')
    destination = remaining / 'combined'
    if destination.exists():
        raise FileExistsError(destination)
    raw = pd.concat([_read_completed_metrics(earlier, old_allocations),
                     _read_completed_metrics(remaining, new_allocations)], ignore_index=True)
    outputs, report = _summaries(raw, ALLOCATIONS, (earlier.name, remaining.name))
    staging = remaining / f'.combined_{uuid4().hex}'
    staging.mkdir(exist_ok=False)
    try:
        _write_report(staging / 'REPORT.md', staging, outputs, report)
        staging.rename(destination)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
    return destination / 'REPORT.md'
