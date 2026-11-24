"""复算 V5 随机窗口预测，并与两组冻结 V3 结果作同配置配对。"""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import LABELS, METRICS, scores
from . import v5_report as report
from .v5_noise_report import ALL_CONDITIONS, CLEAN_CONDITION, CONDITION_NAMES

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path



REFERENCE = report.PROJECT / 'experiments/CWRU/实验结果/CWRU_V3/window_random_20261023_0853'
SELECTION_REFERENCE = report.PROJECT / 'experiments/CWRU/实验结果/CWRU_V3/run_20261022_2105/selection_evaluation/predictions'
TRAIN_COUNTS = {1.: 481, .1: 49, .2: 98, .5: 242}
ALLOCATIONS = ((1., 0), *report.SMALL)
PROTOCOLS = {'clean_only': CLEAN_CONDITION, 'awgn_mixed': ALL_CONDITIONS}
CASE = ['ratio', 'subset', 'protocol', 'seed']
META = [*CASE, 'variant', 'domain', 'source_load_hp', 'aux_weight', 'origin']
ROW = ['job_id', 'condition', 'noise_seed']
CELL = [*ROW, 'true_label', 'predicted_label']
METHODS = {'v5_39': 'V5 39 维融合', 'v3_212': 'V3 212 维融合', 'v3_data': 'V3 数据单分支'}


def _read_structure(root, variants, frozen):
    jobs = pd.read_csv(root / 'models_registry.csv')
    if frozen:
        # V3 新训练的登记留空，评价时统一写成 new。
        jobs['origin'] = jobs.origin.fillna('new')
    combos = {(ratio, subset, protocol, seed, variant)
              for ratio, subset in ALLOCATIONS for protocol in PROTOCOLS for seed in report.SEEDS for variant in variants}
    referenced = jobs.origin.eq('selection_reference')
    if (len(jobs) != len(combos) or jobs.job_id.duplicated().any()
            or set(jobs[[*CASE, 'variant']].itertuples(index=False, name=None)) != combos
            or not (jobs.completed.eq(True) & jobs.from_scratch.eq(True) & jobs.domain.eq('window_random')
                    & jobs.source_load_hp.eq(-1) & jobs.aux_weight.eq(jobs.variant.map({'data_aux_ce': .3, 'baseline_data': 0.}))
                    & jobs.training_windows.eq(jobs.ratio.map(TRAIN_COUNTS)) & jobs.validation_windows.eq(103)).all()
            or not set(jobs.origin).issubset({'new', 'selection_reference'})
            or not referenced.eq(jobs.ratio.eq(1.) & jobs.protocol.eq('clean_only')).all()):
        raise ValueError(f'{root.name} 模型登记、固定窗口配额或系数不完整')
    evaluations = pd.read_csv(root / 'evaluation/evaluations.csv')
    _verify_meta(root, evaluations, jobs)
    if (len(evaluations) != len(jobs) or evaluations.job_id.duplicated().any()
            or not evaluations.test_windows.eq(103).all()):
        raise ValueError(f'{root.name} 评价登记不完整')
    directory = (root / 'evaluation/predictions').resolve()
    paths = set()
    for row in evaluations.itertuples(index=False):
        path = resolve_result_path(row.prediction_path).resolve()
        expected_parent = SELECTION_REFERENCE if frozen and row.origin == 'selection_reference' else directory
        if path.parent != expected_parent or path in paths or not path.is_file():
            raise ValueError(f'{root.name} 预测路径重复、越界或缺失 {path}')
        paths.add(path)
    return jobs, evaluations


def _verify_meta(root, frame, jobs):
    if set(frame.job_id) != set(jobs.job_id) or not frame.target_load_hp.eq(-1).all():
        raise ValueError(f'{root.name} 评价与登记模型集合不匹配')
    joined = frame[['job_id', *META]].drop_duplicates().merge(jobs[['job_id', *META]], on='job_id',
                 suffixes=('_result', '_job'), validate='many_to_one')
    if any(not joined[f'{key}_result'].eq(joined[f'{key}_job']).all() for key in META):
        raise ValueError(f'{root.name} 评价元数据与模型登记不符')


def _frozen_reference():
    state = report._state(REFERENCE)
    if (state['stage'] != 'complete' or state['selected_aux_weight'] != .3
            or state['decisions_locked'] is not True or state['target_test_accessed'] is not True
            or state['evaluated_pairs'] != 200 or state['metric_rows'] != 2700):
        raise ValueError('V3 冻结参照状态与随机窗口协议不符')
    return _read_structure(REFERENCE, ('baseline_data', 'data_aux_ce'), frozen=True)


def preflight():
    """检查冻结参照的结构与路径，不打开保存概率。"""
    jobs, evaluations = _frozen_reference()
    return dict(v3_reference_models=len(jobs), v3_reference_pairs=len(evaluations))


def _test_windows(root, compare):
    labels = window_ids = None
    for ratio, subset in ALLOCATIONS:
        name = f'window_random_r{int(ratio * 100):03d}_s{subset}'
        relative = Path('splits') / name
        if compare:
            for filename in ('windows.csv', 'train.txt', 'validation.txt', 'test.txt'):
                if (root / relative / filename).read_bytes() != (REFERENCE / relative / filename).read_bytes():
                    raise ValueError(f'V3 与 V5 固定划分不一致 {name}/{filename}')
        table = pd.read_csv(root / relative / 'windows.csv')
        test = table[table.role.eq('test')]
        current_labels, current_ids = test.label.to_numpy(), test.window_index.to_numpy()
        np.testing.assert_array_equal(current_ids, np.loadtxt(root / relative / 'test.txt', dtype=int, ndmin=1))
        np.testing.assert_array_equal(np.bincount(current_labels, minlength=4), [11, 30, 31, 31])
        if labels is None:
            labels, window_ids = current_labels, current_ids
        else:
            np.testing.assert_array_equal(current_labels, labels)
            np.testing.assert_array_equal(current_ids, window_ids)
    return labels, window_ids


def _verify_predictions(root, jobs, evaluations, labels, window_ids):
    evaluation = root / 'evaluation'
    metrics = pd.read_csv(evaluation / 'metrics.csv').sort_values(ROW).reset_index(drop=True)
    confusion = pd.read_csv(evaluation / 'confusion_matrices.csv').sort_values(CELL).reset_index(drop=True)
    rows = sum(len(PROTOCOLS[protocol]) for protocol in jobs.protocol)
    cells = len(LABELS)**2
    if (len(metrics) != rows or len(confusion) != rows * cells
            or metrics.duplicated(ROW).any() or confusion.duplicated(CELL).any()
            or not confusion[ROW].iloc[::cells].reset_index(drop=True).equals(metrics[ROW])):
        raise ValueError(f'{root.name} 评价指标、混淆计数或条件范围不完整')
    for frame in (metrics, confusion):
        _verify_meta(root, frame, jobs)
    matrix_order = np.array([(a, b) for a in range(len(LABELS)) for b in range(len(LABELS))])
    if not (confusion[['true_label', 'predicted_label']].to_numpy().reshape(-1, cells, 2) == matrix_order).all():
        raise ValueError('混淆矩阵类别不符')
    values = metrics[METRICS].to_numpy(float)
    matrices = confusion[['count', 'support', 'row_percent']].to_numpy(float).reshape(-1, cells, 3)
    groups = metrics.groupby('job_id', sort=False).indices
    for row in evaluations.itertuples(index=False):
        indices = groups[row.job_id]
        conditions = list(metrics.loc[indices, ['condition', 'noise_seed']].itertuples(index=False, name=None))
        if set(conditions) != set(PROTOCOLS[row.protocol]):
            raise ValueError(f'{root.name}/{row.job_id} 固定噪声条件或种子异常')
        names = [f'{condition}_n{seed}' for condition, seed in conditions]
        with np.load(resolve_result_path(row.prediction_path), allow_pickle=False) as predictions:
            if set(predictions.files) != {'labels', 'window_indices', *names}:
                raise ValueError(f'{root.name}/{row.job_id} 预测条件不完整')
            np.testing.assert_array_equal(predictions['labels'], labels)
            np.testing.assert_array_equal(predictions['window_indices'], window_ids)
            for index, name in zip(indices, names, strict=True):
                p = predictions[name]
                if (p.shape != (len(labels), len(LABELS)) or not np.isfinite(p).all()
                        or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5, rtol=0)):
                    raise ValueError(f'{root.name}/{row.job_id}/{name} 概率无效')
                result, counts = scores(labels, p)
                support = np.repeat(counts.sum(1), len(LABELS))
                matrix = np.stack((counts.ravel(), support, 100 * counts.ravel() / support), axis=-1)
                if (not np.allclose(values[index], [result[metric] for metric in METRICS], atol=1e-10, rtol=0)
                        or not np.allclose(matrices[index], matrix, atol=1e-9, rtol=0)):
                    raise ValueError(f'{root.name}/{row.job_id}/{name} 复算指标或混淆矩阵不一致')
    return metrics


def _summaries(frame, group):
    within = report._summary(frame, ['ratio', 'subset', 'protocol', 'condition', group], len(report.SEEDS))
    budget_keys = ['ratio', 'protocol', 'condition', group]
    by_seed = frame.groupby([*budget_keys, 'seed'], as_index=False)
    counts = by_seed.size()
    if not counts['size'].eq(np.where(counts.ratio.eq(1.), 1, len(report.SUBSETS))).all():
        raise ValueError('种子内的固定子集不完整')
    budget = report._summary(by_seed[METRICS].mean(), budget_keys, len(report.SEEDS))
    small = frame[frame.ratio.lt(1.)].groupby([*budget_keys, 'subset'], as_index=False)[METRICS].mean()
    between = report._summary(small, budget_keys, len(report.SUBSETS))
    return within, budget, between


def _table(frame, group, difference=False):
    rows = []
    for record in frame.sort_values(['ratio', 'protocol', 'condition', group]).to_dict('records'):
        label = METHODS[record[group]]
        row = {'预算': '全样本' if record['ratio'] == 1 else f'{record["ratio"]:.0%}',
               '训练': '纯干净' if record['protocol'] == 'clean_only' else 'AWGN 混合',
               '目标条件': CONDITION_NAMES[record['condition']],
               '方法': f'V5 减 {label}' if difference else label}
        for metric, title in zip(METRICS, report.METRIC_NAMES, strict=True):
            mean, sd = (100 * record[f'{metric}_{stat}'] for stat in ('mean', 'std'))
            row[title] = f'{mean:+.2f} ± {sd:.2f}' if difference else f'{mean:.2f} ± {sd:.2f}'
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def finish(root):
    root = Path(root).resolve()
    state = report._state(root)
    if (state['stage'] != 'reporting' or state['selected_aux_weight'] != .3
            or not all(state[key] is True for key in ('weights_verified', 'decisions_locked', 'target_test_accessed'))
            or (state['trained_models'], state['referenced_models'], state['evaluated_pairs'], state['metric_rows'])
               != (95, 5, 100, 1350)):
        raise ValueError('V5 随机窗口报告仅接受全部模型核验及测试完成的结果')
    jobs, evaluations = _read_structure(root, ('data_aux_ce',), frozen=False)
    old_jobs, old_evaluations = _frozen_reference()
    labels, windows = _test_windows(root, compare=True)
    metrics = _verify_predictions(root, jobs, evaluations, labels, windows).assign(method='v5_39', experiment=root.name)
    old = _verify_predictions(REFERENCE, old_jobs, old_evaluations, labels, windows)
    old = old.assign(method=old.variant.map({'data_aux_ce': 'v3_212', 'baseline_data': 'v3_data'}), experiment=REFERENCE.name)
    raw = pd.concat([metrics, old], ignore_index=True)
    keys = [*CASE, 'condition', 'method']
    groups = raw.groupby(keys, as_index=False)
    sizes = groups.size()
    if len(raw) != 4050 or not sizes['size'].eq(np.where(sizes.condition.eq('clean'), 1, 5)).all():
        raise ValueError('三组指标或噪声重复数不完整')
    per_seed = groups[METRICS].mean()
    within, budget, between = _summaries(per_seed, 'method')
    wide = per_seed.pivot(index=[*CASE, 'condition'], columns='method', values=METRICS)
    if len(wide) != 350 or wide.isna().any().any():
        raise ValueError('V5 与两个 V3 方法的同条件配对不完整')
    candidate = wide.xs('v5_39', axis=1, level='method')
    differences = pd.concat([candidate.sub(wide.xs(method, axis=1, level='method')).reset_index().assign(reference=method)
                             for method in ('v3_212', 'v3_data')], ignore_index=True)
    pair_within, pair_budget, pair_between = _summaries(differences, 'reference')
    frames = {'comparison_metrics.csv': raw, 'comparison_per_seed_snr.csv': per_seed,
              'comparison_subset_mean_std.csv': within, 'comparison_budget_mean_std.csv': budget,
              'comparison_between_subset.csv': between, 'paired_seed_differences.csv': differences,
              'paired_subset_mean_std.csv': pair_within, 'paired_budget_mean_std.csv': pair_budget,
              'paired_between_subset.csv': pair_between}
    output = root / 'REPORT.md'
    paths = {root / 'evaluation' / filename: frame for filename, frame in frames.items()}
    for path in (output, *paths):
        if path.exists():
            raise FileExistsError(path)
    highlighted = budget[budget.condition.isin(['clean', 'white_-4'])]
    paired_highlight = pair_budget[pair_budget.condition.isin(['clean', 'white_-4'])]
    markdown = (
        '# CWRU V5 39 维记录内随机窗口实验\n\n'
        '采用此前确认的 λ = 0.3。95 个模型从头训练，5 个全样本纯干净模型引用已核验的选参权重。'
        '100 个 V5 模型均在本轮首次进行目标评价，生成 1350 行指标和 21600 个混淆计数。'
        '从保存的概率复算 V5 1350 行及冻结 V3 两组 2700 行指标与对应混淆矩阵。\n\n'
        '固定记录内划分覆盖 40 条记录及 687 个不重叠窗口。全样本训练池为 481 窗，'
        '10%、20%、50% 各子集为 49、98、242 窗。验证与测试各 103 窗。'
        '小样本比例对应训练窗口，另需完整验证集标签。两组融合均采用 λ = 0.3。\n\n'
        '## 干净目标与 −4 dB 目标\n\n'
        '先在各模型内平均同 SNR 的五次噪声，再对小样本三个固定子集等权汇总。'
        '表内为五种子均值 ± 样本标准差，单位为百分比。其余 SNR 和子集间波动见 `evaluation/`。\n\n'
        + _table(highlighted, 'method') + '\n\n'
        '## 同训练协议的配对差值\n\n'
        '按同一预算、固定子集、训练种子及目标条件配对，单位为百分点。\n\n'
        + _table(paired_highlight, 'reference', difference=True) + '\n\n'
        '## 解释范围\n\n'
        '训练和测试窗口共享原始记录。结果衡量记录内窗口分类能力。'
        'V3 冻结结果已包含相同测试窗，存在历史目标结果曝光。种子、子集和噪声实例是当前设计下的重复，'
        '样本标准差描述训练波动。AWGN 结果适用于本实验的固定噪声。'
        'V5 与 V3 212 维的比较包括物理描述符和输入层规模的改变。与数据单分支的比较包含融合和辅助分类的整体效果。\n'
    )
    for path, frame in paths.items():
        frame.to_csv(path, index=False, mode='x')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(markdown)
