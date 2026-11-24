"""复算 V5 与冻结 V3 的纯干净跨负载预测，并生成配对报告。"""

import json
from pathlib import Path

import pandas as pd

from metrics import LABELS, METRICS
from experiments.CWRU.实验文件.CWRU_V4.v4_report import META, _validate_predictions

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path



PROJECT = Path(__file__).resolve().parents[4]
V3_RESULTS = PROJECT / 'experiments/CWRU/实验结果' / 'CWRU_V3'
FULL_REFERENCE = V3_RESULTS / 'load_full_20261024_0950'
SMALL_REFERENCE = V3_RESULTS / 'load_small_20261025_2017'
SEEDS = (1706, 1707, 1708, 1709, 1710)
SUBSETS = (4101, 4102, 4103)
SOURCES = range(4)
FULL = ((1., 0),)
SMALL = tuple((ratio, subset) for ratio in (.1, .2, .5) for subset in SUBSETS)
ALLOCATIONS = FULL + SMALL
CASE = ['ratio', 'subset', 'source_load_hp', 'seed']
PAIR = [*CASE, 'target_load_hp']
DIRECTION = ['ratio', 'subset', 'source_load_hp', 'target_load_hp']
METHODS = {'v3_data': 'V3 数据单分支', 'v3_212': 'V3 212 维融合', 'v5_39': 'V5 39 维融合'}
METRIC_NAMES = ['ACC', 'BA', 'Macro-F1', '正常召回', '内圈召回', '外圈召回', '滚动体召回']
V3_VARIANTS = ('baseline_data', 'data_aux_ce')
GRID = (.01, .03, .1, .3, 1.)


def _state(root):
    return json.loads((root / 'config.json').read_text(encoding='utf-8'))


def _split_path(root, ratio, subset, source):
    return root / 'splits' / f'load_source_{source}_r{int(ratio * 100):03d}_s{subset}' / 'windows.csv'


def _read_structure(root, allocations, weight, historical=False):
    jobs = pd.read_csv(root / 'models_registry.csv')
    if historical:
        jobs = jobs[jobs.protocol.eq('clean_only')]
        variants = V3_VARIANTS
    else:
        jobs = jobs[jobs.domain.str.startswith('load_source_')]
        variants = ('data_aux_ce',)

    expected = {(ratio, subset, source, seed, variant)
                for ratio, subset in allocations for source in SOURCES
                for seed in SEEDS for variant in variants}
    if (len(jobs) != len(expected) or jobs.job_id.duplicated().any()
            or set(jobs[[*CASE, 'variant']].itertuples(index=False, name=None)) != expected):
        raise ValueError(f'{root.name} 纯干净迁移模型组合不完整或重复')
    if not (jobs.completed.eq(True) & jobs.from_scratch.eq(True) & jobs.protocol.eq('clean_only')
            & jobs.domain.eq('load_source_' + jobs.source_load_hp.astype(str))
            & jobs.aux_weight.eq(jobs.variant.map({'baseline_data': 0., 'data_aux_ce': weight}))).all():
        raise ValueError(f'{root.name} 模型状态、来源、协议或系数异常')

    evaluation = root / 'evaluation'
    evaluations = pd.read_csv(evaluation / 'evaluations.csv')
    if historical:
        evaluations = evaluations[evaluations.protocol.eq('clean_only')]
    expected_pairs = {(row.job_id, target) for row in jobs.itertuples(index=False)
                      for target in SOURCES if target != row.source_load_hp}
    if (len(evaluations) != len(expected_pairs)
            or set(evaluations[['job_id', 'target_load_hp']].itertuples(index=False, name=None)) != expected_pairs):
        raise ValueError(f'{root.name} 模型与目标负载组合不完整或重复')
    joined = evaluations.merge(jobs[['job_id', *META]], on='job_id', validate='many_to_one',
                               suffixes=('_result', '_job'))
    if any(not joined[f'{key}_result'].eq(joined[f'{key}_job']).all() for key in META):
        raise ValueError(f'{root.name} 评价元数据与模型登记不一致')

    prediction_dir = (evaluation / 'predictions').resolve()
    paths = set()
    for row in evaluations.itertuples(index=False):
        path = resolve_result_path(row.prediction_path).resolve()
        if path.parent != prediction_dir or path in paths or not path.is_file():
            raise ValueError(f'{root.name} 预测路径不唯一、越界或缺失 {path}')
        paths.add(path)
    for ratio, subset in allocations:
        for source in SOURCES:
            split = _split_path(root, ratio, subset, source)
            if not split.is_file():
                raise FileNotFoundError(split)
    return jobs, evaluations


def _references():
    refs = []
    for root, allocations in ((FULL_REFERENCE, FULL), (SMALL_REFERENCE, SMALL)):
        state = _state(root)
        if (state['stage'] != 'complete' or state['decisions_locked'] is not True
                or state['target_test_accessed'] is not True or state['selected_aux_weight'] != .3):
            raise ValueError(f'{root.name} 不是已完成并锁定的 V3 对照')
        jobs, evaluations = _read_structure(root, allocations, .3, historical=True)
        for filename in ('metrics.csv', 'confusion_matrices.csv'):
            fields = pd.read_csv(root / 'evaluation' / filename, nrows=0).columns
            required = {'job_id', *META, 'target_load_hp', 'condition', 'noise_seed'}
            required.update(METRICS if filename == 'metrics.csv' else
                            ['true_label', 'predicted_label', 'count', 'support', 'row_percent'])
            if not required.issubset(fields):
                raise ValueError(f'{root.name} {filename} 缺少复算所需字段')
        refs.append((root, allocations, jobs, evaluations))
    return refs


def preflight():
    """核对冻结对照的文件和配对设计，不读取预测值或写文件。"""
    refs = _references()
    return {'v3_reference_models': sum(len(jobs) for _, _, jobs, _ in refs),
            'v3_reference_pairs': sum(len(evaluations) for _, _, _, evaluations in refs)}


def _same_splits(root, reference, allocations):
    fields = ['window_index', 'record_index', 'record_id', 'role', 'label', 'rpm',
              'start_sample', 'end_sample', 'historical_role', 'load_hp', 'split_seed']
    for ratio, subset in allocations:
        for source in SOURCES:
            current = pd.read_csv(_split_path(root, ratio, subset, source))
            previous = pd.read_csv(_split_path(reference, ratio, subset, source))
            pd.testing.assert_frame_equal(current[fields], previous[fields], check_dtype=False)


def _verified_results(root, jobs, evaluations, methods, historical=False):
    directory = root / 'evaluation'
    metrics = pd.read_csv(directory / 'metrics.csv')
    confusion = pd.read_csv(directory / 'confusion_matrices.csv')
    if historical:
        metrics = metrics[metrics.protocol.eq('clean_only')]
        confusion = confusion[confusion.protocol.eq('clean_only')]
    _validate_predictions(root, jobs, metrics, confusion, evaluations)
    return (metrics.assign(method=metrics.variant.map(methods), experiment=root.name),
            confusion.assign(method=confusion.variant.map(methods), experiment=root.name))


def _summary(frame, keys, repeats):
    groups = frame.groupby(keys, sort=True, as_index=True)
    if not groups.size().eq(repeats).all():
        raise ValueError('方向、种子或子集的汇总重复数不完整')
    result = groups[METRICS].agg(['mean', 'std'])
    result.columns = [f'{metric}_{stat}' for metric, stat in result.columns]
    return result.reset_index()


def _aggregate(frame, name):
    by_direction = _summary(frame, [*DIRECTION, name], len(SEEDS))
    direction_groups = frame.groupby(['ratio', 'subset', name, 'seed'], as_index=False)
    if not direction_groups.size()['size'].eq(12).all():
        raise ValueError('各训练种子在每个子集须覆盖12个方向')
    by_seed = direction_groups[METRICS].mean()
    by_subset = _summary(by_seed, ['ratio', 'subset', name], len(SEEDS))
    budget_seed = by_seed.groupby(['ratio', name, 'seed'], as_index=False)[METRICS].mean()
    by_budget = _summary(budget_seed, ['ratio', name], len(SEEDS))
    subset_means = by_seed[by_seed.ratio.lt(1)].groupby(['ratio', 'subset', name], as_index=False)[METRICS].mean()
    between_subsets = _summary(subset_means, ['ratio', name], len(SUBSETS))
    return by_direction, by_subset, by_budget, between_subsets


def _pair_differences(frame):
    wide = frame.pivot(index=PAIR, columns='method', values=METRICS)
    if len(wide) != 600 or wide.isna().any().any():
        raise ValueError('V5 与两个 V3 对照的配对不完整')
    candidate = wide.xs('v5_39', axis=1, level='method')
    rows = [candidate.sub(wide.xs(method, axis=1, level='method')).reset_index().assign(reference=method)
            for method in ('v3_212', 'v3_data')]
    return pd.concat(rows, ignore_index=True)


def _pooled_confusion(confusion):
    matrix_keys = [*DIRECTION, 'method', 'true_label']
    pooled = confusion.groupby([*matrix_keys, 'predicted_label'], as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby(matrix_keys)['count'].transform('sum')
    pooled['unique_test_support'] = pooled.prediction_support // len(SEEDS)
    pooled['row_percent'] = 100 * pooled['count'] / pooled.prediction_support
    return pooled


def _table(frame, name, delta=False):
    rows = []
    for row in frame.to_dict('records'):
        label = METHODS[row[name]]
        if delta:
            label = f'V5 减 {label}'
        values = {}
        for metric, title in zip(METRICS, METRIC_NAMES):
            mean, sd = (100 * row[f'{metric}_{stat}'] for stat in ('mean', 'std'))
            values[title] = f'{mean:+.2f} ± {sd:.2f}' if delta else f'{mean:.2f} ± {sd:.2f}'
        rows.append({'训练预算': '全样本' if row['ratio'] == 1 else f'{row["ratio"]:.0%}',
                     '模型': label, **values})
    return pd.DataFrame(rows).to_markdown(index=False)


def _report(weight, budget, pair_budget, direction):
    critical = direction[direction.source_load_hp.eq(3) & direction.target_load_hp.eq(0)]
    critical_by_budget = critical.groupby(['ratio', 'method'], as_index=False)[
        [f'{metric}_mean' for metric in METRICS]].mean()
    rows = []
    for row in critical_by_budget.to_dict('records'):
        rows.append({'训练预算': '全样本' if row['ratio'] == 1 else f'{row["ratio"]:.0%}',
                     '模型': METHODS[row['method']],
                     **{name: f'{100 * row[f"{metric}_mean"]:.2f}'
                        for metric, name in zip(METRICS, METRIC_NAMES)}})
    critical_table = pd.DataFrame(rows).to_markdown(index=False)
    weight_note = ('两个融合模型使用相同的辅助系数。' if weight == .3 else
                   '两个融合模型的辅助系数不同，配对差值反映描述符与辅助系数共同改变后的差异。')
    decision_note = (
        '查看原始五档验证结果后，明确融合分类损失系数为 1，原始波形辅助损失系数须大于 0 且小于 1。'
        '在已测试且满足该约束的候选中，λ = 0.3 的平均验证交叉熵最低，为 6.11391×10⁻⁶。'
        '用户确认正式跨负载实验使用 λ = 0.3。' if weight == .3 else ''
    )
    return (
        '# CWRU V5 39 维物理描述符纯干净跨负载报告\n\n'
        'V5 在固定记录内随机窗口划分上完成 25 个纯干净融合选参模型。'
        '五档系数各用五个训练种子。原始五档中，λ = 1.0 的五种子最佳验证交叉熵均值最低，'
        '为 5.43963×10⁻⁶。原始选参报告保留全部五档结果。'
        f'{decision_note}选参阶段只使用验证指标。\n\n'
        '本轮用锁定系数从头训练 200 个来源模型。全部来源选模并核验权重后，'
        '各模型在另外三个负载上评价，共 600 个模型与目标负载组合。'
        '对照只读引用冻结 V3 的 212 维融合和数据单分支，共 1200 个组合。'
        '三组模型的目标窗口、保存指标和混淆计数均已从预测重新核对。\n\n'
        f'V5 固定辅助系数 λ={weight:g}，V3 212 维融合采用其冻结的 λ=0.3。{weight_note}'
        '两个融合模型的物理输入层宽度也不同。V3 数据单分支与融合的对照包含物理融合与辅助分类的整体差异。\n\n'
        '## 预算等权汇总\n\n'
        '每个种子内先等权平均 12 个方向。小样本再等权平均三个固定子集。'
        '表内为五个训练种子的均值与样本标准差，单位为百分比。\n\n'
        + _table(budget, 'method') + '\n\n'
        '## V5 相对冻结 V3 的配对差值\n\n'
        '同一来源、目标、预算、子集与种子配对后按上述规则汇总，单位为百分点。\n\n'
        + _table(pair_budget, 'reference', delta=True) + '\n\n'
        '## 3 HP 到 0 HP\n\n'
        '下表为五种子和固定子集等权均值，单位为百分比。各预算各子集的单独结果和逐种子配对差值见 `evaluation/`。\n\n'
        + critical_table + '\n\n'
        '## 汇总表与解释范围\n\n'
        '`comparison_direction_mean_std.csv` 按子集和方向报告五种子波动。'
        '`comparison_subset_mean_std.csv` 给出各子集的 12 方向等权结果。'
        '`comparison_between_subset.csv` 先求各子集的五种子均值，再报告三个子集间标准差。'
        '`pooled_confusion_matrices.csv` 汇总五次预测，`unique_test_support` 记录各方向各类的不同目标窗口数。\n\n'
        '全负载选参使用 103 个带标签验证窗口。四来源分别另用 19、39、40、40 个带标签验证窗口选模。'
        '小样本比例针对训练窗口，仍使用完整的来源验证集。方向、种子和子集共享记录或模型，'
        '汇总标准差描述本设计内的波动。同批目标记录已有历史测试曝光，结果属于本数据集上的探索性跨负载比较。'
        '物理特征在测试时依赖记录转速，本次评价限于干净观测。\n'
    )


def finish(root):
    """只在目标评价完成后从保存文件复算，并独占写入报告和汇总表。"""
    root = Path(root).resolve()
    cfg = _state(root)
    if (cfg['stage'] != 'reporting' or cfg['decisions_locked'] is not True
            or cfg['weights_verified'] is not True or cfg['target_test_accessed'] is not True
            or cfg['evaluated_pairs'] != 600 or cfg['selected_aux_weight'] not in GRID):
        raise ValueError('报告要求 V5 选模和权重已锁定，600 个目标组合全部保存')
    weight = cfg['selected_aux_weight']
    jobs, evaluations = _read_structure(root, ALLOCATIONS, weight)
    verified = [_verified_results(root, jobs, evaluations, {'data_aux_ce': 'v5_39'})]
    for reference, allocations, old_jobs, old_evaluations in _references():
        _same_splits(root, reference, allocations)
        verified.append(_verified_results(reference, old_jobs, old_evaluations,
                                          {'data_aux_ce': 'v3_212', 'baseline_data': 'v3_data'}, historical=True))
    metrics = pd.concat([part[0] for part in verified], ignore_index=True)
    confusion = pd.concat([part[1] for part in verified], ignore_index=True)
    if len(metrics) != 1800 or len(confusion) != len(metrics) * len(LABELS)**2:
        raise ValueError('三组复算指标或混淆矩阵的总数不完整')

    direction, subset, budget, between_subsets = _aggregate(metrics, 'method')
    paired = _pair_differences(metrics)
    pair_dir, pair_subset, pair_budget, pair_between_subsets = _aggregate(paired, 'reference')
    outputs = {
        'comparison_metrics.csv': metrics,
        'comparison_confusion_matrices.csv': confusion,
        'comparison_direction_mean_std.csv': direction,
        'comparison_subset_mean_std.csv': subset,
        'comparison_budget_mean_std.csv': budget,
        'comparison_between_subset.csv': between_subsets,
        'paired_seed_differences.csv': paired,
        'paired_direction_mean_std.csv': pair_dir,
        'paired_subset_mean_std.csv': pair_subset,
        'paired_budget_mean_std.csv': pair_budget,
        'paired_between_subset.csv': pair_between_subsets,
        'pooled_confusion_matrices.csv': _pooled_confusion(confusion),
    }
    report = _report(weight, budget, pair_budget, direction)
    report_path = root / 'REPORT.md'
    outputs = {root / 'evaluation' / filename: frame for filename, frame in outputs.items()}
    for path in [report_path, *outputs]:
        if path.exists():
            raise FileExistsError(path)
    for path, frame in outputs.items():
        frame.to_csv(path, index=False, mode='x')
    with report_path.open('x', encoding='utf-8') as stream:
        stream.write(report)
