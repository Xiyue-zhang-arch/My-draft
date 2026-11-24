"""V3单来源跨负载小样本报告，仅汇总本轮已经保存的评价。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import METRICS
from report import validation_table
from window_report import _costs, _csv, _curves, _display, _mean_std, _text


SEEDS = set(range(1706, 1711))
SUBSETS = {4101, 4102, 4103}
RATIOS = (.1, .2, .5)
VARIANTS = ('baseline_data', 'data_aux_ce')
CONDITIONS = ('clean', 'white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4')
LABELS = ['N', 'IR', 'OR', 'B']
KEYS = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'source_load_hp',
        'target_load_hp', 'condition']
EVAL_KEYS = ['job_id', 'target_load_hp', 'condition', 'noise_seed']
TRAIN_SUPPORT = np.array([[8, 17, 24, 24], [16, 44, 48, 48],
                          [16, 48, 48, 48], [16, 48, 48, 48]])
VAL_SUPPORT = np.array([[2, 5, 6, 6], [4, 11, 12, 12],
                        [4, 12, 12, 12], [4, 12, 12, 12]])
TEST_SUPPORT = TRAIN_SUPPORT + VAL_SUPPORT
OUTPUTS = ['per_seed_metrics.csv', 'mean_std.csv', 'between_subset_summary.csv',
           'paired_seed_differences.csv', 'paired_mean_std.csv',
           'paired_between_subset_summary.csv', 'transfer_equal_direction_per_seed.csv',
           'transfer_equal_direction_mean_std.csv', 'transfer_between_subset_summary.csv',
           'transfer_paired_seed_differences.csv', 'transfer_paired_mean_std.csv',
           'transfer_paired_between_subset_summary.csv', 'pooled_confusion_matrices.csv',
           'validation_mean_std.csv', 'negative_mean_differences.csv']


def _within(path, parent):
    path = Path(path).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f'本轮报告不能引用输出目录以外的模型或预测: {path}')
    return path


def _supports(root):
    support = pd.read_csv(root/'dataset_support.csv')
    keys = ['domain', 'ratio', 'subset', 'role']
    expected = {}
    for source in range(4):
        for ratio in RATIOS:
            for subset in SUBSETS:
                prefix = (f'load_source_{source}', ratio, subset)
                expected[(*prefix, 'train')] = np.ceil(TRAIN_SUPPORT[source]*ratio).astype(int)
                expected[(*prefix, 'validation')] = VAL_SUPPORT[source]
                for target in set(range(4))-{source}:
                    expected[(*prefix, f'test_hp{target}')] = TEST_SUPPORT[target]
    if len(support) != 180 or support.duplicated(keys).any():
        raise ValueError('要求36个固定划分各自完整的训练、验证及三个目标支持数')
    indexed = support.set_index(keys)
    if set(indexed.index) != set(expected):
        raise ValueError('dataset_support的来源、预算、子集或角色不完整')
    for key, counts in expected.items():
        row = indexed.loc[key]
        if not np.array_equal(row[LABELS].to_numpy(float), counts) or row.windows != counts.sum():
            raise ValueError(f'类别配额或支持数错误: {key}')
    return support


def validate(root, cfg, m, cm, evaluations):
    """检查配置全集、评价条件、固定支持数，并由混淆计数复核所有指标。"""
    root = Path(root).resolve()
    if cfg.get('decisions_locked') is not True or cfg.get('selected_aux_weight') != .3:
        raise ValueError('报告前必须完成选模并锁定统一lambda=0.3')
    jobs = pd.DataFrame(cfg['jobs'])
    cases = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed']
    expected = {(f'load_source_{s}', r, u, p, v, seed)
                for s in range(4) for r in RATIOS for u in SUBSETS
                for p in ('clean_only', 'awgn_mixed') for v in VARIANTS for seed in SEEDS}
    if len(jobs) != 720 or jobs.job_id.nunique() != 720 or jobs.duplicated(cases).any():
        raise ValueError('报告要求720个不同的小样本来源模型')
    if set(jobs[cases].itertuples(index=False, name=None)) != expected:
        raise ValueError('四来源、三预算、三子集、两协议、两方法、五种子配置不完整')
    if not jobs.completed.eq(True).all() or not jobs.from_scratch.eq(True).all():
        raise ValueError('全部720模型必须从头训练完成')
    if not jobs.domain.eq('load_source_'+jobs.source_load_hp.astype(str)).all():
        raise ValueError('模型domain与来源负载不一致')
    if not jobs.loc[jobs.variant.eq('data_aux_ce'), 'aux_weight'].eq(.3).all():
        raise ValueError('融合模型必须使用已锁定的统一lambda')
    if not jobs.loc[jobs.variant.eq('baseline_data'), 'aux_weight'].eq(0).all():
        raise ValueError('数据单分支不能使用辅助损失')
    support = _supports(root)
    for job in cfg['jobs']:
        source = int(job['source_load_hp'])
        train = int(np.ceil(TRAIN_SUPPORT[source]*job['ratio']).sum())
        if job['training_windows'] != train or job['validation_windows'] != VAL_SUPPORT[source].sum():
            raise ValueError('模型登记的训练或验证标注预算与固定划分不一致')
        folder = _within(job['folder'], root/'models')
        history = pd.read_csv(folder/'training.csv')
        required = ['epoch', 'main_ce', 'train_acc_sampled_augmented', 'validation_nll',
                    'validation_clean_balanced_acc', 'validation_clean_acc', 'validation_clean_recall_OR']
        if job['protocol'] == 'awgn_mixed':
            required += ['validation_white_-4_balanced_acc', 'validation_white_-4_recall_OR']
        if not set(required).issubset(history.columns) or len(history) != job['epochs_trained']:
            raise ValueError('训练逐轮记录缺失或与登记不一致')
        if not np.array_equal(history.epoch, np.arange(1, len(history)+1)):
            raise ValueError('训练epoch记录不连续')
        if not np.isfinite(history[required].to_numpy(float)).all():
            raise ValueError('训练曲线中存在非有限值')
    if len(m) != 29160 or len(cm) != 466560 or len(evaluations) != 2160:
        raise ValueError('要求29160逐条件指标、466560混淆计数、2160模型目标评价')
    expected_ids = set(jobs.job_id)
    for frame in (m, cm, evaluations):
        if set(frame.job_id) != expected_ids:
            raise ValueError('模型登记与评价对象不一致')
        meta = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed', 'aux_weight', 'source_load_hp']
        check = frame[['job_id', *meta]].drop_duplicates().merge(
            jobs[['job_id', *meta]], on='job_id', suffixes=('_result', '_job'), validate='many_to_one')
        if any(not check[c+'_result'].eq(check[c+'_job']).all() for c in meta):
            raise ValueError('评价元数据与模型登记不一致')
    if m.duplicated(EVAL_KEYS).any() or evaluations.duplicated(['job_id', 'target_load_hp']).any():
        raise ValueError('逐条件指标或模型目标评价重复')
    actual_pairs = set(evaluations[['job_id', 'target_load_hp']].itertuples(index=False, name=None))
    expected_pairs = {(j['job_id'], t) for j in cfg['jobs'] for t in set(range(4))-{j['source_load_hp']}}
    if actual_pairs != expected_pairs:
        raise ValueError('每个来源模型必须且只能评价另外三个负载')
    expected_conditions = {'clean_only': {('clean', 0)},
        'awgn_mixed': {('clean', 0)} | {(c, n) for c in CONDITIONS[1:] for n in range(7201, 7206)}}
    for (_, _), group in m.groupby(['job_id', 'target_load_hp']):
        if set(zip(group.condition, group.noise_seed)) != expected_conditions[group.protocol.iloc[0]]:
            raise ValueError('测试条件或噪声实例不完整，正SNR必须保留加号')
    metric_pairs = set(m[['job_id', 'target_load_hp']].itertuples(index=False, name=None))
    if metric_pairs != expected_pairs:
        raise ValueError('逐条件指标的目标集合不正确')
    for row in evaluations.itertuples(index=False):
        if row.test_windows != TEST_SUPPORT[int(row.target_load_hp)].sum():
            raise ValueError('目标测试窗口数不正确')
        path = _within(row.prediction_path, root/'evaluation'/'predictions')
        if not path.is_file():
            raise ValueError('登记预测文件缺失')
    if evaluations.prediction_path.nunique() != 2160:
        raise ValueError('不同模型目标组合必须分别保存预测文件')
    values = m[METRICS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('分类指标非有限或超出0至1')
    cm_keys = EVAL_KEYS+['true_label', 'predicted_label']
    counts = cm['count'].to_numpy(float)
    if cm.duplicated(cm_keys).any() or not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1).any():
        raise ValueError('混淆矩阵重复或计数不正确')
    if set(cm.true_label) != set(range(4)) or set(cm.predicted_label) != set(range(4)):
        raise ValueError('混淆矩阵类别不正确')
    if not cm.groupby(EVAL_KEYS).size().eq(16).all():
        raise ValueError('每次评价必须保存完整4乘4混淆计数')
    row_support = cm.groupby(EVAL_KEYS+['true_label'])['count'].sum()
    matrix_support = row_support.unstack('true_label')
    expected_support = TEST_SUPPORT[matrix_support.index.get_level_values('target_load_hp').to_numpy(int)]
    if not np.array_equal(matrix_support.to_numpy(), expected_support):
        raise ValueError('混淆计数与固定目标类别支持数不一致')
    expected_cell_support = TEST_SUPPORT[cm.target_load_hp.to_numpy(int), cm.true_label.to_numpy(int)]
    if not np.array_equal(cm.support.to_numpy(), expected_cell_support):
        raise ValueError('混淆矩阵support列不正确')
    if not np.allclose(cm.row_percent, 100*counts/expected_cell_support, atol=1e-9, rtol=0):
        raise ValueError('混淆矩阵逐行百分比不正确')
    diag = cm[cm.true_label.eq(cm.predicted_label)].set_index(EVAL_KEYS+['true_label'])['count']
    recalls = (diag/row_support).unstack('true_label')
    predicted = cm.groupby(EVAL_KEYS+['predicted_label'])['count'].sum()
    predicted.index = predicted.index.set_names(EVAL_KEYS+['true_label'])
    f1 = (2*diag/(row_support+predicted)).unstack('true_label').mean(axis=1)
    recomputed = pd.DataFrame({'acc': diag.groupby(level=EVAL_KEYS).sum()/matrix_support.sum(axis=1),
        'balanced_acc': recalls.mean(axis=1), 'f1_macro': f1,
        **{metric: recalls[i] for i, metric in enumerate(METRICS[3:])}}).sort_index()
    recorded = m.set_index(EVAL_KEYS)[METRICS].sort_index()
    if not recorded.index.equals(recomputed.index) or not np.allclose(recorded, recomputed[METRICS], atol=1e-9, rtol=0):
        raise ValueError('保存指标与混淆计数不一致')
    return jobs, support


def _between(frame, keys):
    """每子集先平均训练种子，再计算三个子集均值之间的样本SD。"""
    subset_means = frame.groupby(keys, as_index=False)[METRICS].mean()
    group_keys = [k for k in keys if k != 'subset']
    for _, group in subset_means.groupby(group_keys):
        if len(group) != 3 or set(group.subset) != SUBSETS:
            raise ValueError('子集间统计要求完整三个固定子集')
    summary = subset_means.groupby(group_keys)[METRICS].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    return summary.reset_index()


def _paired(frame, keys):
    pair_keys = [k for k in keys if k != 'variant']+['seed']
    table = frame.pivot(index=pair_keys, columns='variant', values=METRICS)
    if table.isna().any().any() or set(table.columns.get_level_values(1)) != set(VARIANTS):
        raise ValueError('两方法配对不完整')
    differences = pd.DataFrame({metric: table[(metric, 'data_aux_ce')]-table[(metric, 'baseline_data')]
                               for metric in METRICS}).reset_index()
    return differences, _mean_std(differences, pair_keys[:-1]), _between(differences, pair_keys[:-1])


def _validation(jobs):
    val = validation_table(jobs.to_dict('records'))
    val['source_load_hp'] = val.job_id.map(jobs.set_index('job_id').source_load_hp)
    if val.duplicated(['job_id', 'condition', 'noise_seed']).any():
        raise ValueError('保存验证指标重复')
    for _, group in val.groupby('job_id'):
        expected = {('clean', 0)}
        if group.protocol.iloc[0] == 'awgn_mixed':
            expected |= {(c, 6201) for c in CONDITIONS[1:]}
        if set(zip(group.condition, group.noise_seed)) != expected:
            raise ValueError('保存验证条件不完整或误用了测试噪声种子')
    values = val[METRICS+['validation_nll']].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (val[METRICS].to_numpy() > 1).any():
        raise ValueError('保存验证指标不正确')
    keys = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'source_load_hp', 'condition']
    per_seed = val.groupby(keys+['seed'], as_index=False)[METRICS+['validation_nll']].mean()
    return val, _mean_std(per_seed, keys, METRICS+['validation_nll'])


def _pair_display(summary):
    rows = []
    for r in summary.to_dict('records'):
        row = {'训练比例': f'{r["ratio"]:.0%}', '子集': r['subset'], '训练协议': r['protocol'], '测试条件': r['condition']}
        for metric, label in zip(METRICS, ['ACC', 'BA', 'Macro-F1', '正常召回', '内圈召回', '外圈召回', '滚动体召回']):
            row[label+'差值'] = f'{100*r[metric+"_mean"]:+.2f}±{100*r[metric+"_std"]:.2f}'
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def finish(root):
    """720模型全部评价落盘后调用。已有报告输出时拒绝覆盖，不修改config。"""
    root = Path(root).resolve()
    out = root/'evaluation'
    protected = [root/name for name in ['REPORT.md', 'FIGURES.md', 'figures', 'validation_metrics.csv', 'computational_cost.csv']]
    protected += [out/name for name in OUTPUTS]
    if any(path.exists() for path in protected):
        raise FileExistsError('报告输出已存在，禁止覆盖或自动重新生成')
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    m = pd.read_csv(out/'metrics.csv')
    cm = pd.read_csv(out/'confusion_matrices.csv')
    evaluations = pd.read_csv(out/'evaluations.csv')
    jobs, support = validate(root, cfg, m, cm, evaluations)
    seed = m.groupby(KEYS+['seed'], as_index=False)[METRICS].mean()
    summary = _mean_std(seed, KEYS)
    differences, pair_summary, pair_between = _paired(seed, KEYS)
    transfer_keys = ['ratio', 'subset', 'protocol', 'variant', 'condition']
    direction_counts = seed.groupby(transfer_keys+['seed']).size()
    if not direction_counts.eq(12).all():
        raise ValueError('每个种子内必须先等权平均完整12方向')
    transfer = seed.groupby(transfer_keys+['seed'], as_index=False)[METRICS].mean()
    transfer_summary = _mean_std(transfer, transfer_keys)
    transfer_pair, transfer_pair_summary, transfer_pair_between = _paired(transfer, transfer_keys)
    val, val_summary = _validation(jobs)
    pooled = cm.groupby(KEYS+['true_label', 'predicted_label'], as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby(KEYS+['true_label'])['count'].transform('sum')
    pooled['row_percent'] = 100*pooled['count']/pooled.prediction_support
    pooled['unique_test_support'] = TEST_SUPPORT[pooled.target_load_hp.to_numpy(int), pooled.true_label.to_numpy(int)]
    pooled['prediction_repetitions'] = np.where(pooled.condition.eq('clean'), 5, 25)
    negative = pair_summary.melt(id_vars=[k for k in KEYS if k != 'variant'],
        value_vars=[metric+'_mean' for metric in METRICS], var_name='metric', value_name='mean_difference')
    negative = negative[negative.mean_difference.lt(0)].copy()
    negative['metric'] = negative.metric.str.removesuffix('_mean')
    tables = {'per_seed_metrics.csv': seed, 'mean_std.csv': summary,
        'between_subset_summary.csv': _between(seed, KEYS),
        'paired_seed_differences.csv': differences, 'paired_mean_std.csv': pair_summary,
        'paired_between_subset_summary.csv': pair_between,
        'transfer_equal_direction_per_seed.csv': transfer,
        'transfer_equal_direction_mean_std.csv': transfer_summary,
        'transfer_between_subset_summary.csv': _between(transfer, transfer_keys),
        'transfer_paired_seed_differences.csv': transfer_pair,
        'transfer_paired_mean_std.csv': transfer_pair_summary,
        'transfer_paired_between_subset_summary.csv': transfer_pair_between,
        'pooled_confusion_matrices.csv': pooled, 'validation_mean_std.csv': val_summary,
        'negative_mean_differences.csv': negative}
    for name, frame in tables.items():
        _csv(frame, out/name)
    _csv(val, root/'validation_metrics.csv')
    costs = _costs(jobs)
    costs['source_load_hp'] = costs.job_id.map(jobs.set_index('job_id').source_load_hp)
    _csv(costs, root/'computational_cost.csv')
    figures = []
    for domain, group in jobs.groupby('domain'):
        dest = root/'figures'/domain
        dest.mkdir(parents=True, exist_ok=False)
        figures.extend(f'{domain}/{name}' for name in _curves(group, dest))
    if len(figures) != 72:
        raise ValueError('必须生成72组完整训练曲线')
    _text(root/'FIGURES.md', '# 来源、预算、子集和协议训练曲线\n\n'+
        '\n\n'.join(f'## {name}\n\n![训练曲线](figures/{name})' for name in figures)+'\n')
    chosen = transfer_summary[transfer_summary.condition.isin(['clean', 'white_-4'])]
    paired_chosen = transfer_pair_summary[transfer_pair_summary.condition.isin(['clean', 'white_-4'])]
    usage = support[support.role.isin(['train', 'validation'])][['domain', 'ratio', 'role', 'windows', *LABELS]].drop_duplicates()
    declines = negative.groupby('metric').size().rename('负差值配置数').reset_index()
    recovery_note = ''
    if cfg.get('interruption_history'):
        recovery_note = ('\n本轮有显式授权的中断恢复。已完成模型只读复用，未完成尝试的原文件保留，'
            '对应配置在新文件夹从头训练，未使用缺少优化器状态的最佳权重冒充完整断点续训。'
            '中断尝试不纳入720个完成模型或预测重复数；其耗时不含在登记完成模型的成本表中，'
            '原训练记录及config.json的interruption_history保留供追溯。\n')
    report = f'''# CWRU_V3跨负载小样本正式报告

本轮完成720个从头训练的小样本来源模型，统一λ固定为{cfg['selected_aux_weight']}。全部模型完成来源验证选模并锁定后，统一评价另外三个目标负载，共2160个模型目标组合、29160组逐条件指标及466560个混淆计数单元。本报告只汇总本轮新增结果。
{recovery_note}

## 数据及评价范围

使用全部40条记录的687个完整窗口，保留既有四个来源划分。训练比例10%、20%、50%在各来源训练池内逐类向上取整抽取，每档使用4101、4102、4103三个固定嵌套子集。各方法使用相同子集、训练种子和测试观测。两种训练协议均比较数据单分支与融合加数据分支辅助分类。每配置使用1706至1710五个训练种子。

来源0/1/2/3 HP完整训练池为73/156/160/160窗。固定验证分别为19/39/40/40窗。目标0/1/2/3 HP分别包含92/195/200/200窗，不重复窗口的类别支持数保存在dataset_support.csv。训练比例是相对于各来源训练池，来源之间没有强行匹配训练窗口数。训练和验证预算如下，三个子集的数量一致，但窗口成员不同。

{usage.to_markdown(index=False)}

## 训练、标准化和测试流程

统一λ来自原全样本纯干净验证选参，本轮不重新选择。每个小样本模型从头训练，不使用全样本预训练权重，也不借用全训练池拟合的物理标准化统计。纯干净只用本训练子集干净观测拟合median/IQR，AWGN用本子集等量干净和含噪观测拟合。拟合噪声种子6301、三次重复，尺度下限0.05并裁剪到[-8,8]。这些重复不增加独立训练窗数。

最多200 epoch，patience 20，每轮验证并保存最佳权重，最小改善量1e-6。AdamW初始学习率5e-4、权重衰减1e-3、batch size 32、梯度裁剪5。200轮单周期余弦衰减至1e-5。普通CE且标签平滑为0。纯干净依据来源窗口平均验证CE选模，AWGN使用干净验证CE权重0.5、五档噪声验证CE各0.1。辅助损失不进入验证选模。相同上限和选模规则不表示实际训练更新量相同，也不证明每种方法均已充分收敛。

AWGN训练每次窗口出现时独立以50%概率保留干净，否则SNR均匀采样[-4,4] dB。融合两分支使用同一观测。验证噪声种子为6201，测试为7201至7205。纯干净模型仅评价干净目标，AWGN模型评价干净及-4、-2、0、2、4 dB各五个噪声实例。保存条件名为white_-4、white_-2、white_+0、white_+2、white_+4。

## 统计口径

每模型先平均同一SNR下的五个噪声实例，再在固定来源、目标、预算和子集内计算五训练种子的均值及样本标准差。evaluation/mean_std.csv保存ACC、BA、Macro-F1和四类召回。between_subset_summary.csv单列三个子集五种子均值之间的样本标准差。两方法按来源、目标、预算、子集、协议和种子配对，差值为融合加辅助分类减数据单分支，见paired_seed_differences.csv、paired_mean_std.csv和paired_between_subset_summary.csv。

12方向等权结果先在每个训练种子编号内平均12个方向，然后计算五种子的均值及样本标准差。该SD不是12方向间的SD，也不是置信区间。transfer_equal_direction_per_seed.csv保留种子层数据，transfer_equal_direction_mean_std.csv保留汇总。transfer_between_subset_summary.csv报告三个子集均值间变化。transfer_paired_*保存相同口径的配对差值。没有把15个种子子集组合或12个方向当作独立数据集重复，也不根据均值排序宣称统计显著。

## 12方向等权干净及强噪声结果

下表以百分比表示均值和五种子样本标准差。不能将总体均值较高解释为每个方向、类别和种子均改善。

{_display(chosen)}

## 12方向等权配对差值

下表单位为百分点，正值表示融合方案较高，负值表示较低。BA和各类召回应同时解释。所有方向和全部噪声档的完整结果保留在evaluation目录。

{_pair_display(paired_chosen)}

逐方向固定子集五种子平均差值为负的结果全部保留在evaluation/negative_mean_differences.csv。下表仅统计负差值配置的数量，包含多个相关条件，不能解释为独立失败次数或显著性检验。

{declines.to_markdown(index=False)}

## 混淆矩阵、训练曲线与计算成本

confusion_matrices.csv保存每次评价原始计数。pooled_confusion_matrices.csv在固定方向、预算、子集、协议和条件内汇总五种子及噪声重复，按真实类别逐行归一化，每行100%。unique_test_support是该目标类别去除模型和噪声重复后的窗口数，不代表窗口之间统计独立。prediction_support是重复预测总数，prediction_repetitions记录重复数。重复预测不扩大原始测试样本数。本报告生成前已根据混淆计数复核全部29160组指标及固定类别支持数，没有重新训练或重新推理。

FIGURES.md索引72组训练曲线，覆盖四来源、三预算、三子集和两协议。每图显示两方法五个种子的各自轨迹，不补齐早停后的epoch。训练ACC来自实际抽样和增强观测，不能等同于未经增强的完整训练集准确率。逐轮训练主CE、总损失、辅助CE及验证指标保留在各模型原始记录中。

computational_cost.csv保存参数量、单样本推理延迟、每epoch训练时间、总训练时间、标准化拟合、验证及梯度诊断时间、峰值显存、硬件和精度。延迟包括窗口标准化、在线物理特征和网络，batch size 1，10次预热和50次同步测量，不含磁盘与H2D。训练batch size 32。Conv1d/Linear FLOPs不含FFT，不能宣称完整FLOPs。峰值显存为已分配显存并包含首批梯度诊断，具体范围以逐模型latency_scope和peak_memory_scope为准。没有用特征缓存隐藏物理提取成本。

## 解释边界

全部40记录及固定划分已经用于历史开发，历史测试曝光不会因新目录或从头训练消失。统一λ在全负载随机窗口验证上确定，目标负载参与过方法开发。本轮只是固定该开发结果后，在单一来源上训练选模并评价其他负载的探索性实验，不能称严格未见域、独立轴承或独立确认性泛化。

来源训练与验证共享原始记录。三个子集来自同一来源训练池并在预算之间嵌套，五种子只描述训练随机性，噪声实例只描述人工加噪变化。12方向复用来源模型与目标数据，这些重复不构成独立数据集重复。AWGN结论只适用于当前人工白噪声设置，不能直接推广到真实工业噪声。

每个小样本配置仍使用来源完整的19/39/40/40个验证标签。训练窗口数不是总标注预算，当前结果也不是完全不借助全样本开发信息的少标签学习。来源之间的训练及验证预算不同，不把来源难度差异全部归因于负载变化。

来源0的10%预算只有9个训练窗，其中正常类仅1窗，不能称充分覆盖故障异质性。每轮按ceil(训练窗数/32)计算批次数，因此预算变化也改变优化更新上限，不能把预算间全部差异归因于样本量。物理分支使用记录RPM或标称转速，数据单分支不使用该元数据，本轮不单独识别转速信息带来的收益。

两组只能评价融合加辅助分类整体方案相对于数据单分支的差别，不能分离物理融合和辅助损失的独立贡献，也不能仅凭性能变化证明物理机制。全样本干净验证选择的λ不保证各小样本、噪声或来源条件最优，但本轮不逐条件调整。所有较低的BA或类别召回均保留，不以跨方向平均掩盖局部退化。
'''
    _text(root/'REPORT.md', report)
    return {'models': 720, 'new_models': 720, 'reference_models': 0,
            'metric_rows': 29160, 'evaluation_pairs': 2160, 'confusion_rows': 466560,
            'training_figures': 72, 'report': str(root/'REPORT.md')}
