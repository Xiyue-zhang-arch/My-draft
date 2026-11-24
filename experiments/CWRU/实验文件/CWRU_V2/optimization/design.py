"""后续优化的固定配置与验证集判定，不访问文件或测试集。

只有通过预设验证规则的候选才能替换当前配置；失败保留原配置。
种子SD与配对增益SD均描述性报告，不作为0.5个百分点硬门槛。
"""
from __future__ import annotations

import math
import numbers
import re


DEFAULT_SPEC = {
    'aux_weight': .1,
    'physics_fit': 'clean',
    'aux_head': 'shared',
    'val_noise_seeds': [6201],
    'max_epochs': 100,
    'schedule_epochs': 100,
    'patience': 20,
    'fit_noise_seed': 6301,
    'fit_noise_repeats': 3,
}
LAMBDA_GRID = [.01, .03, .1, .3, 1.]
SEEDS = [1706, 1707, 1708]
FULL_SEEDS = [1706, 1707, 1708, 1709, 1710]
SUBSETS = [4101, 4102, 4103]
SCREEN_RATIOS = [.1, .2]
PROTOCOLS = ['clean_only', 'awgn_mixed']
STAGES = ['lambda', 'physics_fit', 'aux_head', 'validation_noise', 'expansion', 'transfer']
TEST_NOISE_SEEDS = frozenset(range(7201, 7206))
NOISE_CONDITIONS = ['white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4']
EPS = 1e-12


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name}必须为不小于{minimum}的整数')
    return int(value)


def validate_spec(spec):
    """返回规范化的新字典；漏项、未知项与范围外参数均拒绝。"""
    if not isinstance(spec, dict):
        raise TypeError('spec必须是字典')
    expected = set(DEFAULT_SPEC)
    missing, extra = expected - set(spec), set(spec) - expected
    if missing or extra:
        raise ValueError(f'配置字段不一致，缺少{sorted(missing)}，多余{sorted(extra)}')
    result = dict(spec)
    weight = result['aux_weight']
    if (isinstance(weight, bool) or not isinstance(weight, numbers.Real)
            or not math.isfinite(float(weight)) or float(weight) not in LAMBDA_GRID):
        raise ValueError(f'aux_weight必须属于事先固定的{LAMBDA_GRID}')
    result['aux_weight'] = float(weight)
    if result['physics_fit'] not in ('clean', 'matched_noise'):
        raise ValueError('physics_fit仅允许clean或matched_noise')
    if result['aux_head'] not in ('shared', 'independent'):
        raise ValueError('aux_head仅允许shared或independent')
    seeds = result['val_noise_seeds']
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError('val_noise_seeds必须为非空列表')
    seeds = [_integer(s, 'validation noise seed') for s in seeds]
    if len(set(seeds)) != len(seeds) or TEST_NOISE_SEEDS.intersection(seeds):
        raise ValueError('验证噪声种子不能重复或与测试种子重合')
    if seeds not in ([6201], [6201, 6202, 6203]):
        raise ValueError('本轮验证方案仅允许[6201]或[6201,6202,6203]')
    result['val_noise_seeds'] = list(seeds)
    for key in ('max_epochs', 'schedule_epochs', 'patience', 'fit_noise_seed', 'fit_noise_repeats'):
        result[key] = _integer(result[key], key)
    if result['max_epochs'] not in (30, 50, 100, 200):
        raise ValueError('短程探索30/50epoch；原主实验100epoch；单独收敛诊断200epoch')
    if result['schedule_epochs'] != 100 or result['patience'] != 20:
        raise ValueError('本轮不同时改变100epoch余弦周期或20epoch早停耐心')
    if result['fit_noise_seed'] in TEST_NOISE_SEEDS or result['fit_noise_seed'] in seeds:
        raise ValueError('拟合噪声种子必须与验证及测试种子分离')
    if result['fit_noise_seed'] != 6301 or result['fit_noise_repeats'] != 3:
        raise ValueError('拟合噪声种子6301、重复3次为本轮固定设置')
    return result


def config_id(variant, spec):
    """可读配置标识；数据单分支忽略不参与其训练的物理与辅助设置。"""
    s = validate_spec(spec)
    if variant not in ('baseline_data', 'data_aux_ce'):
        raise ValueError('后续仅允许baseline_data与data_aux_ce')
    parts = [variant]
    if variant == 'data_aux_ce':
        parts += ['a' + format(s['aux_weight'], '.15g').replace('.', 'p'),
                  'fit' + s['physics_fit'], 'head' + s['aux_head']]
        if s['physics_fit'] == 'matched_noise':
            parts += [f"fitseed{s['fit_noise_seed']}", f"fitrep{s['fit_noise_repeats']}"]
    parts += ['v' + '-'.join(map(str, s['val_noise_seeds'])),
              f"e{s['max_epochs']}", f"s{s['schedule_epochs']}", f"p{s['patience']}"]
    return '_'.join(parts)


def _validation_frame(metrics):
    # 延迟导入，plan/config_id可不加载数值库，避免Windows导入顺序影响torch。
    import pandas as pd

    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError('需要非空验证指标表')
    f = metrics.copy()
    required = {'config_id', 'ratio', 'subset', 'protocol', 'seed', 'condition',
                'noise_seed', 'balanced_acc', 'recall_OR', 'validation_nll'}
    if required - set(f):
        raise ValueError(f'验证指标缺少列{sorted(required-set(f))}')
    for key in ('role', 'evaluation_set', 'data_split', 'phase', 'stage', 'split'):
        if key in f and f[key].dropna().astype(str).str.contains(
                r'(?:^|[\W_])test(?:ing)?(?:$|[\W_])', case=False, regex=True).any():
            raise ValueError(f'{key}含测试标记，禁止参与验证筛选')
    for key in ('target_test_accessed', 'is_test'):
        if key in f and f[key].map(lambda x: str(x).lower() in ('true', '1', 'yes')).any():
            raise ValueError(f'{key}表明测试数据已混入筛选表')
    if 'test_windows' in f and pd.to_numeric(f.test_windows, errors='coerce').fillna(0).gt(0).any():
        raise ValueError('含test_windows的测试评价表不能用于选参')
    if f[list(required)].isna().any().any():
        raise ValueError('验证指标关键列存在缺失值')
    if not set(f.protocol).issubset(PROTOCOLS):
        raise ValueError('未知训练协议')
    for key in ('ratio', 'balanced_acc', 'recall_OR', 'validation_nll', 'noise_seed'):
        f[key] = pd.to_numeric(f[key], errors='raise')
        if not f[key].map(lambda x: math.isfinite(float(x))).all():
            raise ValueError(f'{key}含非有限值')
    if not f.ratio.gt(0).all() or not f.ratio.le(1).all():
        raise ValueError('ratio应为(0,1]内比例')
    for key in ('balanced_acc', 'recall_OR'):
        if not f[key].between(0, 1).all():
            raise ValueError(f'{key}应为0至1比例，不能输入百分数')
    if not f.validation_nll.ge(0).all():
        raise ValueError('validation_nll不能为负数')
    if f.noise_seed.isin(TEST_NOISE_SEEDS).any():
        raise ValueError('测试噪声实例不得参与验证选参')
    keys = [k for k in ('domain', 'source_load_hp', 'target_load_hp') if k in f]
    keys += ['ratio', 'subset', 'protocol', 'seed', 'condition']
    if f[keys].isna().any().any():
        raise ValueError('验证配对键有缺失值')
    if f.duplicated(['config_id'] + keys + ['noise_seed']).any():
        raise ValueError('同配置、模型、条件和噪声实例出现重复指标')
    modelkeys = ['config_id'] + [k for k in keys if k != 'condition']
    for _, group in f.groupby(modelkeys, dropna=False, sort=False):
        expected = {'clean'} if group.protocol.iloc[0] == 'clean_only' else {'clean', *NOISE_CONDITIONS}
        if set(group.condition) != expected:
            raise ValueError('每个模型必须有完整协议条件，缺失条件不能视为通过')
        clean = group[group.condition.eq('clean')]
        if len(clean) != 1 or clean.noise_seed.iloc[0] != 0:
            raise ValueError('干净验证须恰有一条且noise_seed为0')
        if group.protocol.iloc[0] == 'awgn_mixed':
            instance_sets = [set(group[group.condition.eq(c)].noise_seed) for c in NOISE_CONDITIONS]
            if any(s != instance_sets[0] for s in instance_sets):
                raise ValueError('同模型各SNR必须使用相同完整验证噪声种子集合')
            if instance_sets[0] not in ({6201}, {6201, 6202, 6203}):
                raise ValueError('验证噪声集合不属于预设方案')
    values = ['balanced_acc', 'recall_OR', 'validation_nll']
    averaged = f.groupby(['config_id'] + keys, as_index=False, dropna=False)[values].mean()
    return f, averaged, keys


def _pair(averaged, candidate_id, reference_id, data_id, keys):
    c = averaged[averaged.config_id.eq(candidate_id)].set_index(keys)
    b = averaged[averaged.config_id.eq(reference_id)].set_index(keys)
    d = averaged[averaged.config_id.eq(data_id)].set_index(keys)
    if c.empty or b.empty or d.empty:
        raise ValueError('候选、前一方案或数据单分支验证指标缺失')
    if not c.index.is_unique or not b.index.is_unique or not d.index.is_unique:
        raise ValueError('模型条件平均后配对键仍不唯一')
    if not c.index.isin(b.index).all() or not c.index.isin(d.index).all():
        raise ValueError('候选缺少完全匹配的前一方案或数据单分支对照')
    b, d = b.loc[c.index], d.loc[c.index]
    out = c.reset_index()
    out['gain'] = c.balanced_acc.to_numpy() - b.balanced_acc.to_numpy()
    out['or_gain'] = c.recall_OR.to_numpy() - b.recall_OR.to_numpy()
    out['data_gain'] = c.balanced_acc.to_numpy() - d.balanced_acc.to_numpy()
    return out


def _per_ratio(f, column, keys):
    units = [k for k in keys if k not in ('protocol', 'seed', 'condition')]
    # 种子及噪声条件先在固定子集内平均，再对子集及预算等权汇总。
    return f.groupby(units, dropna=False)[column].mean().reset_index().groupby('ratio')[column].mean()


def assess(metrics, candidate_id, reference_id, data_id, require_gain=True, full=False):
    """仅验证集的配对判定，所有数值指标以比例输入、百分点输出。

reference_id可含额外预算/种子，但候选的每个键必须有唯一对应值。
每固定子集不足3种子时拒绝判定，不以NaN比较悄悄放行。
"""
    _, averaged, keys = _validation_frame(metrics)
    c = _pair(averaged, candidate_id, reference_id, data_id, keys)
    if set(c.protocol) != set(PROTOCOLS):
        raise ValueError('阶段判定必须同时包含纯干净和AWGN两协议')
    noisy = c[c.protocol.eq('awgn_mixed') & c.condition.isin(['white_-4', 'white_-2'])].copy()
    if noisy.empty:
        raise ValueError('缺少-4/-2dB主验证条件')
    groups = [k for k in keys if k != 'seed']
    counts = noisy.groupby(groups, dropna=False).seed.nunique()
    if counts.empty or counts.lt(3).any():
        raise ValueError('每固定子集与SNR至少需要3个不同训练种子')
    # 干净协议也必须有同等的seed/预算/子集覆盖。
    coverage_keys = [k for k in keys if k not in ('protocol', 'condition')]
    expected_models = noisy[coverage_keys].drop_duplicates().set_index(coverage_keys).index
    for protocol in PROTOCOLS:
        clean_models = c[c.protocol.eq(protocol) & c.condition.eq('clean')][coverage_keys].set_index(coverage_keys).index
        if not expected_models.isin(clean_models).all() or not clean_models.isin(expected_models).all():
            raise ValueError('干净与强噪声评价的模型覆盖不匹配')
    sd = noisy.groupby(groups, dropna=False)[['balanced_acc', 'gain', 'data_gain']].std(ddof=1)
    if sd.isna().any().any():
        raise ValueError('无法计算完整的跨种子样本标准差')
    ratio_gain = _per_ratio(noisy, 'gain', keys)
    gain = float(ratio_gain.mean())
    or_gain = float(_per_ratio(noisy, 'or_gain', keys).mean())
    noise_score = float(_per_ratio(noisy, 'balanced_acc', keys).mean())
    data_gap = float(_per_ratio(noisy, 'data_gain', keys).mean())
    modelkeys = [k for k in keys if k not in ('condition', 'protocol')]
    paired = noisy.groupby(modelkeys, dropna=False)[['gain', 'data_gain']].mean()
    positive = float(paired.gain.gt(EPS).mean())
    clean = {
        p: float(_per_ratio(c[c.protocol.eq(p) & c.condition.eq('clean')], 'gain', keys).mean())
        for p in PROTOCOLS
    }
    # 全样本还要求各协议/条件相对数据单分支都不明显退化。
    data_condition_gap = c.groupby(['protocol', 'condition'], dropna=False).data_gain.mean()
    min_data_gap = float(data_condition_gap.min())
    threshold = -.005 if full else (.005 if require_gain else 0.)
    checks = {
        'noisy_gain': gain >= threshold - EPS,
        'positive_pairs': True if (full or not require_gain) else positive >= 2/3 - EPS,
        'or_preservation': or_gain >= -.01 - EPS,
        'clean_preservation': all(v >= -.005 - EPS for v in clean.values()),
        'each_budget_preservation': bool(ratio_gain.ge(-.005-EPS).all()),
        'data_reference': min_data_gap >= -.005-EPS if full else True,
    }
    return {
        'candidate_id': candidate_id,
        'reference_id': reference_id,
        'data_id': data_id,
        'passed': bool(all(checks.values())),
        'checks': checks,
        'noisy_gain_pp': 100*gain,
        'noisy_balanced_acc': noise_score,
        'positive_pair_fraction': positive,
        'or_gain_pp': 100*or_gain,
        'clean_gain_pp': {k: 100*v for k, v in clean.items()},
        'per_budget_gain_pp': {str(k): 100*float(v) for k, v in ratio_gain.items()},
        'max_seed_sd_pp': 100*float(sd.balanced_acc.max()),
        'max_paired_sd_pp': 100*float(sd.gain.max()),
        'seed_sd_over_0_5pp': bool(sd.balanced_acc.gt(.005+EPS).any()),
        'paired_sd_over_0_5pp': bool(sd.gain.gt(.005+EPS).any()),
        'sd_is_hard_gate': False,
        'data_gap_pp': 100*data_gap,
        'min_data_condition_gap_pp': 100*min_data_gap,
        'max_data_paired_sd_pp': 100*float(sd.data_gain.max()),
        'data_paired_sd_over_0_5pp': bool(sd.data_gain.gt(.005+EPS).any()),
        'data_positive_pair_fraction': float(paired.data_gain.gt(EPS).mean()),
        'minimum_seeds_per_subset': int(counts.min()),
        'paired_model_count': len(paired),
        'require_gain': bool(require_gain),
        'full_sample_rule': bool(full),
        'decision_data': 'fixed validation only; test metrics prohibited',
        'stability_rule': 'SD is descriptive, not an acceptance gate or significance test; report paired improvement coverage separately',
    }


def _weight_for(frame, identifier):
    rows = frame[frame.config_id.eq(identifier)]
    if 'aux_weight' in rows:
        weights = rows.aux_weight.dropna().unique()
        if len(weights) != 1:
            raise ValueError('同配置aux_weight必须唯一且非缺失')
        value = float(weights[0])
    else:
        match = re.search(r'_a([^_]+)_', identifier)
        if not match:
            raise ValueError('系数选择需要aux_weight列或config_id生成的标识')
        value = float(match.group(1).replace('p', '.'))
    if not math.isfinite(value) or value not in LAMBDA_GRID:
        raise ValueError('系数不属于事先定义的网格')
    return value


def select_lambda(metrics, candidate_ids, reference_id, data_id):
    """完成整个固定网格后，仅从通过的候选中选系数；否则保留0.1。"""
    original, averaged, keys = _validation_frame(metrics)
    candidates = list(candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates) or reference_id not in candidates:
        raise ValueError('系数候选必须非空、唯一，并包含0.1参照配置')
    weights = {identifier: _weight_for(original, identifier) for identifier in candidates}
    if weights[reference_id] != .1:
        raise ValueError('本轮敏感性分析的历史参照必须为lambda=0.1')
    if sorted(weights.values()) != sorted(LAMBDA_GRID):
        raise ValueError('必须完成全部预设系数，不能提前选择或遗漏不利系数')
    reference_keys = averaged[averaged.config_id.eq(reference_id)].set_index(keys).index
    for identifier in candidates:
        current_keys = averaged[averaged.config_id.eq(identifier)].set_index(keys).index
        if (len(current_keys) != len(reference_keys)
                or not current_keys.isin(reference_keys).all()
                or not reference_keys.isin(current_keys).all()):
            raise ValueError('全部系数必须覆盖同一套预算、子集、种子和验证条件')
    assessments = {
        identifier: assess(metrics, identifier, reference_id, data_id)
        for identifier in candidates
    }
    eligible = [identifier for identifier in candidates
                if identifier != reference_id and assessments[identifier]['passed']]
    if eligible:
        best_score = max(assessments[i]['noisy_balanced_acc'] for i in eligible)
        tied = [i for i in eligible if abs(assessments[i]['noisy_balanced_acc']-best_score) <= EPS]
        selected = min(tied, key=lambda i: (abs(weights[i]-.1), weights[i], i))
    else:
        selected = reference_id
    chosen = assessments[selected]
    adopted = selected != reference_id
    return {
        'selected_config_id': selected,
        'selected_aux_weight': weights[selected],
        'assessments': assessments,
        'adopted': adopted,
        'passed': adopted,
        'reference_retained': not adopted,
        'sd_is_hard_gate': False,
        'eligible_config_ids': eligible,
        'reason': ('已在完整预设网格中选定通过全部验证规则的统一系数。'
                   if adopted else
                   '没有新系数通过全部验证规则，保留0.1，失败结果全部保留；下一阶段以原配置为参照。'),
        'decision_data': 'fixed validation only; no test metrics',
        'tie_break': 'maximum noisy validation BA, then nearest to 0.1, then smaller lambda',
    }
