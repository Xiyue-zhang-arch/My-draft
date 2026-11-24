"""只读复算两版纯干净结果，给汇总工作簿提供数值。"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'experiments/CWRU/实验文件/CWRU_V3'))
from metrics import METRICS, scores

sys.path.insert(0, str(PROJECT / 'experiments/CWRU/实验文件/CWRU_V0/shared'))
from experiment_paths import resolve_result_path


SOURCES = [
    ('V3', 'experiments/CWRU/实验结果/CWRU_V3/load_full_20261024_0950', [1.]),
    ('V3', 'experiments/CWRU/实验结果/CWRU_V3/load_small_20261025_2017', [.1, .2, .5]),
    ('V4', 'experiments/CWRU/实验结果/CWRU_V4/load_full_20261106_131617', [1.]),
    ('V4', 'experiments/CWRU/实验结果/CWRU_V4/load_small_10_20_20261106_145013', [.1, .2]),
    ('V4', 'experiments/CWRU/实验结果/CWRU_V4/load_small_50_20261106_191736', [.5]),
]
KEY = ['ratio', 'subset', 'source_load_hp', 'target_load_hp', 'seed', 'variant']
tables, evaluations, confusion, roots, sources = {}, {}, {}, {}, []
for scheme, relative, budgets in SOURCES:
    root = PROJECT / relative
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    assert cfg['stage'] == 'complete'
    sources.append(dict(scheme=scheme, budgets=budgets, source=relative,
                        lambda_value=cfg['selected_aux_weight'], model_scope=cfg['trained_models']))
    frames = {}
    for filename in ['metrics.csv', 'evaluations.csv', 'confusion_matrices.csv']:
        f = pd.read_csv(root / 'evaluation' / filename)
        mask = f.ratio.isin(budgets) & f.protocol.eq('clean_only')
        if filename != 'evaluations.csv':
            mask &= f.condition.eq('clean') & f.noise_seed.eq(0)
        frames[filename] = f.loc[mask]
    for budget in budgets:
        for filename, destination in [('metrics.csv', tables), ('evaluations.csv', evaluations)]:
            f = frames[filename]
            destination[(scheme, budget)] = f[f.ratio.eq(budget)].set_index(KEY).sort_index()
        f = frames['confusion_matrices.csv']
        confusion[(scheme, budget)] = f[f.ratio.eq(budget)].groupby(KEY)
        roots[(scheme, budget)] = root

verification = dict(metric_rows_recomputed=0, confusion_cells_recomputed=0,
                    baseline_predictions_exact=0, matched_splits=0)
max_error = 0.
for ratio in [.1, .2, .5, 1.]:
    old, new = [tables[(s, ratio)] for s in ['V3', 'V4']]
    assert old.index.equals(new.index) and old.index.is_unique
    assert len(old) == len(new) == (120 if ratio == 1. else 360)
    for idx in old.index:
        predictions = []
        for scheme in ['V3', 'V4']:
            e = evaluations[(scheme, ratio)].loc[idx]
            with np.load(resolve_result_path(e.prediction_path), allow_pickle=False) as saved:
                pred, labels, window = [saved[field].copy() for field in ['clean_n0', 'labels', 'window_indices']]
            if predictions:
                np.testing.assert_array_equal(labels, previous_labels)
                np.testing.assert_array_equal(window, previous_window)
            predictions.append(pred)
            previous_labels, previous_window = labels, window
            values, counts = scores(labels, pred)
            stored = tables[(scheme, ratio)].loc[idx, METRICS].to_numpy(float)
            error = np.max(np.abs(stored - [values[m] for m in METRICS]))
            assert error < 1e-12
            max_error = max(max_error, float(error))
            matrix = confusion[(scheme, ratio)].get_group(idx).sort_values(['true_label', 'predicted_label'])
            np.testing.assert_array_equal(matrix['count'].to_numpy().reshape(4, 4), counts)
            verification['metric_rows_recomputed'] += 1
            verification['confusion_cells_recomputed'] += 16
        if idx[-1] == 'baseline_data':
            np.testing.assert_array_equal(*predictions)
            verification['baseline_predictions_exact'] += 1
    for source in range(4):
        for subset in ([0] if ratio == 1. else [4101, 4102, 4103]):
            split = f'load_source_{source}_r{int(ratio*100):03d}_s{subset}/windows.csv'
            pair = [pd.read_csv(roots[(s, ratio)] / 'splits' / split) for s in ['V3', 'V4']]
            pd.testing.assert_frame_equal(*pair)
            verification['matched_splits'] += 1
verification['maximum_metric_absolute_error'] = max_error

def statistics(frame):
    return ({m + '_mean': float(frame[m].mean()) for m in METRICS}
            | {m + '_sd': float(frame[m].std(ddof=1)) for m in METRICS})

summary, deltas, per_seed, per_subset, subset_sd, directions = [], [], [], [], [], []
names = {'baseline': '数据单分支（两版一致）', 'V3': '原方案 V3 融合', 'V4': '新方案 V4 融合'}
for ratio in [.1, .2, .5, 1.]:
    seed_matrices = {}
    label = '全样本' if ratio == 1. else f'{ratio:.0%}'
    for method in ['baseline', 'V3', 'V4']:
        scheme = 'V3' if method == 'baseline' else method
        variant = 'baseline_data' if method == 'baseline' else 'data_aux_ce'
        f = tables[(scheme, ratio)].xs(variant, level='variant')[METRICS] * 100
        seed = f.groupby('seed')[METRICS].mean()
        assert len(seed) == 5
        seed_matrices[method] = seed
        summary.append(dict(budget=ratio, budget_label=label, method=method, method_label=names[method],
                            lambda_value=None if method == 'baseline' else .3 if method == 'V3' else 1.,
                            seeds=5, subsets=1 if ratio == 1. else 3, **statistics(seed)))
        for k, row in seed.iterrows():
            per_seed.append(dict(budget=ratio, budget_label=label, method=method, seed=int(k), **row.to_dict()))
        subset_means = []
        for subset, group in f.groupby('subset'):
            s = group.groupby('seed')[METRICS].mean()
            per_subset.append(dict(budget=ratio, budget_label=label, method=method,
                                   subset=int(subset), **statistics(s)))
            subset_means.append(s.mean().to_dict())
        if ratio < 1.:
            subset_sd.append(dict(budget=ratio, budget_label=label, method=method,
                                  **statistics(pd.DataFrame(subset_means))))
        for (source, target), group in f.groupby(['source_load_hp', 'target_load_hp']):
            s = group.groupby('seed')[METRICS].mean()
            directions.append(dict(budget=ratio, budget_label=label, method=method,
                                   source=int(source), target=int(target), **statistics(s)))
    deltas.append(dict(budget=ratio, budget_label=label,
                       **statistics(seed_matrices['V4'] - seed_matrices['V3'])))

payload = dict(metrics=METRICS, metric_labels=['ACC', 'BA', 'Macro-F1', '正常召回', '内圈召回', '外圈召回', '滚动体召回'],
               units='percent', summary=summary, deltas=deltas, per_seed=per_seed, per_subset=per_subset,
               subset_sd=subset_sd, directions=directions, sources=sources, verification=verification,
               scope='纯干净跨负载，全样本和10%/20%/50%预算，仅数据单分支与融合加数据辅助分类',
               uncertainty='总表每个训练种子先对12个方向等权，再对小样本的3个固定子集等权，最后报告5种子均值与样本标准差。子集间波动另列。')
out = Path(__file__).resolve().parent / 'summary_data.json'
with out.open('x', encoding='utf-8') as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(verification, ensure_ascii=False))
for row in summary:
    print(row['budget_label'], row['method'],
          *[f'{row[m+"_mean"]:.2f} ± {row[m+"_sd"]:.2f}' for m in METRICS[:3]])
for row in deltas:
    print('DELTA', row['budget_label'], *[round(row[m+'_mean'], 4) for m in METRICS[:3]])
