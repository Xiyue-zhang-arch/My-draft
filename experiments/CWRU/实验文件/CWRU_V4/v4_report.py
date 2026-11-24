"""核验 V4 保存的预测，并汇总同一轮纯干净跨负载评价。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import LABELS, METRICS, scores

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path



SEEDS = (1706, 1707, 1708, 1709, 1710)
SUBSETS = (4101, 4102, 4103)
BUDGETS = [(1.0, 0), *((ratio, subset) for ratio in (.1, .2, .5) for subset in SUBSETS)]
SMALL_RATIOS = (.1, .2)
SMALL_TRAIN_WINDOWS = {.1: '9、17、17、17', .2: '16、33、34、34', .5: '37、78、80、80'}
GRID = (.01, .03, .1, .3, 1.)
VARIANTS = ('baseline_data', 'data_aux_ce')
METHOD_NAMES = dict(zip(VARIANTS, ('数据单分支', '融合加辅助分类')))
METRIC_NAMES = ['ACC', 'BA', 'Macro-F1', '正常召回', '内圈召回', '外圈召回', '滚动体召回']
CASE = ['domain', 'ratio', 'subset', 'variant', 'seed', 'aux_weight']
META = [*CASE, 'protocol', 'source_load_hp']
PAIR = ['job_id', 'target_load_hp']
DIRECTION = ['ratio', 'subset', 'variant', 'source_load_hp', 'target_load_hp']
TRANSFER = ['ratio', 'subset', 'variant']


def _validate_jobs(root, cfg, jobs, selection):
    if not cfg['decisions_locked'] or not cfg['target_test_accessed']:
        raise ValueError('报告须在全部选模锁定及目标测试后生成')
    weight = selection['selected_aux_weight']
    if cfg['selected_aux_weight'] != weight or weight not in GRID or selection['test_accessed'] is not False:
        raise ValueError('选参与实验登记不一致，或选参访问了测试')
    if len(jobs) != 430 or jobs.job_id.nunique() != 430 or jobs.duplicated(CASE).any():
        raise ValueError('要求30个选参模型与400个迁移模型')
    if not (jobs.completed.eq(True) & jobs.from_scratch.eq(True) & jobs.protocol.eq('clean_only')).all():
        raise ValueError('模型须从头完成纯干净训练')
    tuning = jobs[jobs.stage.eq('lambda_selection')]
    expected_tuning = {('window_random', 1., 0, variant, seed, value)
                       for variant in VARIANTS for seed in SEEDS
                       for value in (GRID if variant == 'data_aux_ce' else (0.,))}
    if set(tuning[CASE].itertuples(index=False, name=None)) != expected_tuning:
        raise ValueError('选参模型的候选值、种子或划分不完整')
    observed_ce = tuning[tuning.variant.eq('data_aux_ce')].groupby('aux_weight').validation_nll
    candidates = pd.DataFrame(selection['candidates']).set_index('aux_weight').sort_index()
    if (list(candidates.index) != list(GRID) or not candidates.seeds.eq(5).all()
            or not np.allclose(candidates.mean_best_validation_ce, observed_ce.mean(), atol=1e-12, rtol=0)
            or not np.allclose(candidates.sample_sd, observed_ce.std(), atol=1e-12, rtol=0)):
        raise ValueError('保存选参统计与逐模型验证CE不一致')
    winner = candidates.sort_values(['mean_best_validation_ce', 'aux_weight']).index[0]
    if weight != winner:
        raise ValueError('统一λ与预定验证规则不符')
    transfer = jobs[jobs.stage.isin(['load_transfer_full', 'load_transfer_small'])]
    expected_transfer = {(f'load_source_{source}', ratio, subset, variant, seed,
                          weight if variant == 'data_aux_ce' else 0.)
                         for source in range(4) for ratio, subset in BUDGETS
                         for variant in VARIANTS for seed in SEEDS}
    if set(transfer[CASE].itertuples(index=False, name=None)) != expected_transfer:
        raise ValueError('跨负载配置全集不完整')
    if (not transfer.domain.eq('load_source_' + transfer.source_load_hp.astype(str)).all()
            or not transfer.stage.eq(transfer.ratio.map(
                lambda ratio: 'load_transfer_full' if ratio == 1 else 'load_transfer_small')).all()):
        raise ValueError('迁移模型的来源或阶段错误')
    for folder in jobs.folder:
        if not Path(folder).resolve().is_relative_to(root / 'models'):
            raise ValueError('V4报告只能引用本轮模型')
    return transfer


def _validate_predictions(root, jobs, metrics, confusion, evaluations):
    expected_pairs = {(row.job_id, target) for row in jobs.itertuples(index=False)
                      for target in range(4) if target != row.source_load_hp}
    n_pairs = len(expected_pairs)
    for frame, expected_rows in ((metrics, n_pairs), (evaluations, n_pairs),
                                 (confusion, n_pairs * len(LABELS)**2)):
        if len(frame) != expected_rows or set(frame[PAIR].itertuples(index=False, name=None)) != expected_pairs:
            raise ValueError('跨负载评价行数或目标组合不完整')
        joined = frame[['job_id', *META]].drop_duplicates().merge(
            jobs[['job_id', *META]], on='job_id', validate='many_to_one', suffixes=('_result', '_job'))
        if any(not joined[f'{key}_result'].eq(joined[f'{key}_job']).all() for key in META):
            raise ValueError('评价元数据与模型登记不一致')
    for frame in (metrics, confusion):
        if not (frame.condition.eq('clean') & frame.noise_seed.eq(0)).all():
            raise ValueError('V4只接受纯干净目标评价')
    if metrics.duplicated(PAIR).any() or evaluations.duplicated(PAIR).any():
        raise ValueError('模型目标组合重复')

    metric_rows = metrics.set_index(PAIR)
    matrices = confusion.groupby(PAIR)
    splits = {}
    used_paths = set()
    prediction_dir = root / 'evaluation' / 'predictions'
    for row in evaluations.itertuples(index=False):
        key = (row.domain, row.ratio, row.subset)
        if key not in splits:
            path = root / 'splits' / f'{row.domain}_r{int(row.ratio*100):03d}_s{row.subset}' / 'windows.csv'
            splits[key] = pd.read_csv(path)
        windows = splits[key]
        expected = windows[windows.load_hp.eq(row.target_load_hp)]
        if len(expected) != row.test_windows:
            raise ValueError('评价登记的目标窗口数与固定划分不一致')
        prediction = resolve_result_path(row.prediction_path).resolve()
        if not prediction.is_relative_to(prediction_dir) or prediction in used_paths:
            raise ValueError('预测路径超出本轮目录或重复')
        used_paths.add(prediction)
        with np.load(prediction, allow_pickle=False) as saved:
            labels, indices, probabilities = saved['labels'], saved['window_indices'], saved['clean_n0']
        np.testing.assert_array_equal(indices, expected.window_index)
        np.testing.assert_array_equal(labels, expected.label)
        if (probabilities.shape != (len(labels), len(LABELS))
                or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1., atol=1e-5, rtol=0)):
            raise ValueError('保存的预测概率不合法')
        recomputed, counts = scores(labels, probabilities)
        pair = (row.job_id, row.target_load_hp)
        np.testing.assert_allclose(metric_rows.loc[pair, METRICS].to_numpy(float),
                                   [recomputed[metric] for metric in METRICS], atol=1e-10, rtol=0)
        matrix = matrices.get_group(pair).sort_values(['true_label', 'predicted_label'])
        index = range(len(LABELS))
        if list(zip(matrix.true_label, matrix.predicted_label)) != [(a, b) for a in index for b in index]:
            raise ValueError('每个组合须包含完整4×4混淆矩阵')
        support = np.repeat(counts.sum(1), len(LABELS))
        np.testing.assert_array_equal(matrix['count'], counts.ravel())
        np.testing.assert_array_equal(matrix.support, support)
        np.testing.assert_allclose(matrix.row_percent, 100 * counts.ravel() / support, atol=1e-10, rtol=0)


def _summary(frame, keys, repeats):
    grouped = frame.groupby(keys)[METRICS].agg(['mean', 'std', 'count'])
    grouped.columns = [f'{metric}_{statistic}' for metric, statistic in grouped.columns]
    if not grouped.acc_count.eq(repeats).all():
        raise ValueError('汇总组内重复数不完整')
    return grouped.reset_index()


def _paired(frame, keys):
    index = [key for key in keys if key != 'variant'] + ['seed']
    wide = frame.pivot(index=index, columns='variant', values=METRICS)
    return pd.DataFrame({metric: wide[metric]['data_aux_ce'] - wide[metric]['baseline_data']
                         for metric in METRICS}).reset_index()


def _cell(row, metric):
    return f'{100 * row[f"{metric}_mean"]:.2f} ± {100 * row[f"{metric}_std"]:.2f}'


def _report(summary, directional, selection):
    rows = []
    for row in summary.to_dict('records'):
        rows.append({'训练预算': f'{row["ratio"]:.0%}', '子集': row['subset'] or '完整训练池',
                     '模型': METHOD_NAMES[row['variant']],
                     **{name: _cell(row, metric) for metric, name in zip(METRICS, METRIC_NAMES)}})
    full = directional[directional.ratio.eq(1)]
    direction_rows = []
    for (source, target), group in full.groupby(['source_load_hp', 'target_load_hp']):
        methods = group.set_index('variant').to_dict('index')
        direction_rows.append({'来源（HP）': source, '目标（HP）': target,
                               **{f'{METHOD_NAMES[variant]} {metric_name}': _cell(methods[variant], metric)
                                  for variant in VARIANTS for metric, metric_name in zip(METRICS[:3], METRIC_NAMES[:3])}})
    return (
        '# CWRU V4 全频四带纯干净迁移报告\n\n'
        f'用固定随机窗口验证集比较五档辅助损失系数，共训练30个选参模型。统一λ为'
        f'{selection["selected_aux_weight"]:g}。对每档先取各训练种子的最佳验证CE，再比较五种子均值。'
        '选参模型只做验证。统一系数锁定后，从头训练40个全样本模型和360个小样本模型。'
        '全部来源模型选模完成后，用各模型评价其余三个负载，共1,200次评价。'
        '逐次指标、混淆计数和目标窗口均已与保存的预测复算核对。\n\n'
        '物理分支对0–3、3–6、6–12及12–24 kHz四段分别做滤波和包络分析。'
        '数据单分支接收时域窗口。融合模型同时使用两个分支，并保留数据辅助分类损失。\n\n'
        '## 12个迁移方向的等权汇总\n\n'
        '同一训练种子先等权平均12个方向。表中报告五训练种子均值±样本标准差，单位为百分比。'
        '三个固定子集各自报告。子集间标准差另列，以各子集的五种子均值为统计单位。\n\n'
        + pd.DataFrame(rows).to_markdown(index=False) + '\n\n'
        '## 全样本的各迁移方向\n\n'
        + pd.DataFrame(direction_rows).to_markdown(index=False) + '\n\n'
        '各预算各子集的逐方向指标和四类召回率见 `evaluation/mean_std.csv`。'
        '两方法的差值按同一划分、方向和种子配对，见 `evaluation/paired_mean_std.csv`。'
        '逐种子混淆矩阵见 `evaluation/confusion_matrices.csv`。'
        '五种子汇总混淆矩阵见 `evaluation/pooled_confusion_matrices.csv`，其中同一目标窗口重复预测五次。\n\n'
        '## 验证标签预算与结论范围\n\n'
        '全负载选参使用103个验证窗口。来源0、1、2和3 HP各自另用19、39、40和40个验证窗口选模，'
        '小样本各预算仍使用完整的来源验证集。逐类训练和验证配额见 `dataset_support.csv`。\n\n'
        '全频带边界参考了包含目标负载的全数据描述性分析，统一λ也使用了全负载的随机窗口验证集。'
        '同批记录已有历史测试曝光，因此迁移结果属于本数据集内的探索性比较。'
        '方向间共享训练模型或目标窗口，三个子集来自相同训练池，五种子标准差描述训练随机性。'
        '两组比较衡量融合加辅助分类的整体效果。\n'
    )


def _report_full(summary, directional, selection):
    def format_metrics(row):
        return {'模型': METHOD_NAMES[row['variant']],
                **{name: _cell(row, metric) for metric, name in zip(METRICS, METRIC_NAMES)}}

    total = [format_metrics(row) for row in summary.to_dict('records')]
    ordered = directional.sort_values(['source_load_hp', 'target_load_hp', 'variant'])
    directions = [{'来源（HP）': row['source_load_hp'], '目标（HP）': row['target_load_hp'],
                   **format_metrics(row)} for row in ordered.to_dict('records')]
    return (
        '# CWRU V4 全频四带全样本纯干净跨负载报告\n\n'
        '四个来源负载各自用完整训练池训练数据单分支和融合加辅助分类模型，每组使用五个训练种子。'
        '本轮从头训练40个模型。全部来源模型选模并锁定后，'
        '每个模型在另外三个负载上测试，共120个模型与目标负载组合。'
        '已从保存的预测复算核对各项指标、混淆矩阵和目标窗口。\n\n'
        f'融合模型固定辅助损失系数 λ={selection["selected_aux_weight"]:g}。'
        '该系数来自此前独立的 V4 选参阶段。五档系数各使用五个训练种子，'
        '按五种子最佳验证交叉熵的均值选择，精确相同时选择较小系数。'
        '选参只使用固定验证集。本轮物理分支用0–3、3–6、6–12及12–24 kHz四带构造包络特征。\n\n'
        '## 12个迁移方向的等权结果\n\n'
        '先在各训练种子内对12个迁移方向等权平均，再统计五种子的均值和样本标准差。'
        '表内指标单位为百分比。\n\n'
        + pd.DataFrame(total).to_markdown(index=False) + '\n\n'
        '## 逐方向结果\n\n'
        '各方向均报告五种子均值和样本标准差。\n\n'
        + pd.DataFrame(directions).to_markdown(index=False) + '\n\n'
        '同方向和同种子下的融合模型减数据单分支差值见 `evaluation/paired_mean_std.csv`。'
        '逐种子混淆矩阵见 `evaluation/confusion_matrices.csv`。'
        '五种子汇总混淆矩阵见 `evaluation/pooled_confusion_matrices.csv`，每个目标窗口被重复预测五次。\n\n'
        '## 验证标签预算与结论范围\n\n'
        '此前全负载选参使用103个验证窗口。本轮来源0、1、2、3 HP的训练池分别有73、156、160、160个窗口，'
        '另分别使用19、39、40、40个带标签验证窗口选模。来源训练和验证来自同一批来源记录。\n\n'
        '四带边界参考了包含目标负载的全数据分析，统一系数也来自全负载随机窗口验证。'
        '同批记录已有历史测试曝光，因此这是本数据集内的探索性迁移比较。'
        '部分方向共享来源模型或目标窗口，五种子样本标准差反映训练随机性。'
        '两组模型的差异反映物理融合与辅助分类的整体效果。\n'
    )


def _report_small(summary, between_subset, selection):
    def table(frame, include_subset):
        rows = []
        for row in frame.to_dict('records'):
            subset = {'子集': row['subset']} if include_subset else {}
            rows.append({'训练预算': f'{row["ratio"]:.0%}', **subset, '模型': METHOD_NAMES[row['variant']],
                         **{name: _cell(row, metric) for metric, name in zip(METRICS, METRIC_NAMES)}})
        return pd.DataFrame(rows).to_markdown(index=False)

    ratios = sorted(summary.ratio.unique())
    ratio_text = '和'.join(f'{ratio:.0%}' for ratio in ratios)
    train_counts = '，'.join(f'{ratio:.0%}训练子集分别为{SMALL_TRAIN_WINDOWS[ratio]}窗' for ratio in ratios)
    n_models = 4 * len(summary) * len(SEEDS)
    return (
        '# CWRU V4 全频四带小样本纯干净跨负载报告\n\n'
        f'四个来源负载分别按{ratio_text}预算抽取三个固定子集。每个子集独立训练数据单分支和融合加辅助分类模型，'
        f'每组五个训练种子，共从头训练{n_models}个模型。全部来源模型选模锁定后，各模型在另外三个负载上测试，'
        f'共{n_models * 3}个模型与目标负载组合。目标窗口、分类指标及混淆矩阵均已与保存的预测复算核对。\n\n'
        f'融合模型固定采用此前 V4 验证选中的 λ={selection["selected_aux_weight"]:g}。'
        '五档系数的判据为各档五种子最佳验证交叉熵的均值，完全相同时选择较小系数。'
        '物理分支在0–3、3–6、6–12及12–24 kHz四带分别构造包络特征。\n\n'
        '## 各训练预算和子集的12方向等权结果\n\n'
        '同一训练种子内先平均12个方向，然后按固定子集计算五种子均值和样本标准差。单位为百分比。\n\n'
        + table(summary, include_subset=True) + '\n\n'
        '## 三个子集之间的波动\n\n'
        '以下每格先求各子集的五种子均值，再计算三个子集均值的均值和样本标准差。单位为百分比。\n\n'
        + table(between_subset, include_subset=False) + '\n\n'
        '12个方向各自的五种子均值、样本标准差和四类召回率见 `evaluation/mean_std.csv`。'
        '融合模型减数据单分支的差值按同一方向、子集和种子配对，见 `evaluation/paired_mean_std.csv`。'
        '逐种子混淆矩阵见 `evaluation/confusion_matrices.csv`，五种子汇总见'
        ' `evaluation/pooled_confusion_matrices.csv`，每个目标窗口重复预测五次。\n\n'
        '## 训练和验证标签预算\n\n'
        f'每个来源按类别在原训练池内向上取整。来源0、1、2、3 HP的{train_counts}。'
        '四来源另用19、39、40、40个带标签验证窗口选模，'
        '每个预算均使用完整的来源验证集。此前统一λ使用103个全负载随机划分验证窗口。'
        '逐类支持数见 `dataset_support.csv`。\n\n'
        '## 结论范围\n\n'
        '频带边界参考了包含目标负载的全数据分析，统一系数也来自全负载随机窗口验证。'
        '同批记录已有历史测试曝光，迁移结果属于本数据集内的探索性比较。'
        '方向之间共享训练模型或目标窗口，三个固定子集来自相同来源训练池，'
        '训练种子和子集间标准差分别描述各自的波动。两组比较衡量物理融合加辅助分类的整体效果。\n'
    )


def _finish_transfer(root, jobs, selection, scope):
    evaluation = root / 'evaluation'
    metrics = pd.read_csv(evaluation / 'metrics.csv')
    confusion = pd.read_csv(evaluation / 'confusion_matrices.csv')
    evaluations = pd.read_csv(evaluation / 'evaluations.csv')
    _validate_predictions(root, jobs, metrics, confusion, evaluations)

    directional = _summary(metrics, DIRECTION, len(SEEDS))
    pairs = _paired(metrics, DIRECTION)
    pair_summary = _summary(pairs, [key for key in DIRECTION if key != 'variant'], len(SEEDS))
    direction_groups = metrics.groupby([*TRANSFER, 'seed'])
    if not direction_groups.size().eq(12).all():
        raise ValueError('每个种子必须覆盖12个方向')
    transfer = direction_groups[METRICS].mean().reset_index()
    summary = _summary(transfer, TRANSFER, len(SEEDS))
    transfer_pairs = _paired(transfer, TRANSFER)
    pooled = confusion.groupby([*DIRECTION, 'true_label', 'predicted_label'], as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby([*DIRECTION, 'true_label'])['count'].transform('sum')
    pooled['unique_test_support'] = pooled.prediction_support // len(SEEDS)
    pooled['row_percent'] = 100 * pooled['count'] / pooled.prediction_support
    outputs = {
        'mean_std.csv': directional, 'paired_seed_differences.csv': pairs, 'paired_mean_std.csv': pair_summary,
        'transfer_equal_direction_per_seed.csv': transfer, 'transfer_equal_direction_mean_std.csv': summary,
        'transfer_paired_mean_std.csv': _summary(transfer_pairs, ['ratio', 'subset'], len(SEEDS)),
        'pooled_confusion_matrices.csv': pooled,
    }
    if scope == 'full':
        report = _report_full(summary, directional, selection)
    else:
        subset_mean = transfer[transfer.subset.ne(0)].groupby(TRANSFER)[METRICS].mean().reset_index()
        between_subset = _summary(subset_mean, ['ratio', 'variant'], len(SUBSETS))
        outputs['transfer_between_subset_summary.csv'] = between_subset
        report = (_report_small(summary, between_subset, selection) if scope == 'small'
                  else _report(summary, directional, selection))
    if any((evaluation / name).exists() for name in outputs) or (root / 'REPORT.md').exists():
        raise FileExistsError('正式报告或汇总表已经存在')
    for name, frame in outputs.items():
        frame.to_csv(evaluation / name, index=False, mode='x')
    with (root / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(report)


def finish(root):
    """通过全部完整性检查后独占写入正式报告及表格。"""
    root = Path(root).resolve()
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    jobs = _validate_jobs(root, cfg, pd.read_csv(root / 'models_registry.csv'), selection)
    _finish_transfer(root, jobs, selection, scope='combined')


def _finish_standalone(root, scope, budgets):
    root = Path(root).resolve()
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    jobs = pd.read_csv(root / 'models_registry.csv')
    n_models = 4 * len(budgets) * len(VARIANTS) * len(SEEDS)
    if (cfg['version'] != f'CWRU_V4_load_{scope}' or cfg['planned_models'] != n_models
            or not cfg['decisions_locked'] or not cfg['target_test_accessed']):
        raise ValueError('迁移报告要求独立实验目录及全部选模和测试锁')
    if scope == 'small':
        ratios = sorted({ratio for ratio, _ in budgets})
        if cfg['ratios'] != ratios or cfg['subsets'] != list(SUBSETS):
            ratio_text = '和'.join(f'{ratio:.0%}' for ratio in ratios)
            raise ValueError(f'小样本报告只接受{ratio_text}预算及三个固定子集')
    weight = selection['selected_aux_weight']
    candidates = pd.DataFrame(selection['candidates']).sort_values(['mean_best_validation_ce', 'aux_weight'])
    if (weight != cfg['selected_aux_weight'] or weight != 1.0 or selection['test_accessed'] is not False
            or sorted(candidates.aux_weight) != list(GRID) or not candidates.seeds.eq(5).all()
            or not np.isfinite(candidates[['mean_best_validation_ce', 'sample_sd']].to_numpy(float)).all()
            or candidates.iloc[0].aux_weight != weight):
        raise ValueError('本轮须采用已批准的验证选择 λ=1.0')
    expected = {(f'load_source_{source}', ratio, subset, variant, seed,
                 weight if variant == 'data_aux_ce' else 0.)
                for source in range(4) for ratio, subset in budgets
                for variant in VARIANTS for seed in SEEDS}
    if (len(jobs) != n_models or jobs.job_id.nunique() != n_models
            or set(jobs[CASE].itertuples(index=False, name=None)) != expected):
        raise ValueError(f'报告须包含预定预算和子集的全部{n_models}个迁移模型')
    if not (jobs.completed.eq(True) & jobs.from_scratch.eq(True) & jobs.protocol.eq('clean_only')
            & jobs.stage.eq(f'load_transfer_{scope}')
            & jobs.domain.eq('load_source_' + jobs.source_load_hp.astype(str))).all():
        raise ValueError('迁移模型须从相应来源从头完成纯干净训练')
    for folder in jobs.folder:
        if not Path(folder).resolve().is_relative_to(root / 'models'):
            raise ValueError('迁移报告只能引用本轮模型')
    _finish_transfer(root, jobs, selection, scope)


def finish_full(root):
    """核验独立全样本结果目录并写入正式报告及汇总表。"""
    _finish_standalone(root, 'full', [(1., 0)])


def finish_small(root):
    """核验独立10%和20%小样本结果目录并写入报告。"""
    _finish_standalone(root, 'small', [(ratio, subset) for ratio in SMALL_RATIOS for subset in SUBSETS])


def finish_small_50(root):
    """核验独立50%小样本结果目录并写入报告。"""
    _finish_standalone(root, 'small', [(.5, subset) for subset in SUBSETS])
