"""V4全频四带选参与纯干净跨负载实验。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from uuid import uuid4

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'CWRU_V3'))

# Windows 下先加载 Torch，沿用原 GPU 环境的 DLL 加载顺序。
import torch
import numpy as np
import pandas as pd

import engine
from context import BASE, PROJECT, R
from metrics import METRICS, scores
from report import validation_table
from run_experiment import (Experiment, GRID, RATIOS, SEEDS, SUBSETS, VARIANTS,
                            nested_subset, now, select_lambda, split_name, write_json)
from splits import read_fixed

from experiment_paths import resolve_result_path

RESULTS = PROJECT / 'experiments/CWRU/实验结果/CWRU_V4'
FIXED_SPLITS = PROJECT / 'experiments/CWRU/实验结果/CWRU_V3/run_20261022_2105/splits'
DATA = PROJECT / 'data/raw/CWRU/48k Drive End Bearking Fault Data'
BANDS = [[0, 3000], [3000, 6000], [6000, 12000], [12000, 24000]]
LABELS = ['N', 'IR', 'OR', 'B']
SOURCES = range(4)
WINDOW = 'window_random'
STAGES = ('lambda_selection', 'load_transfer_full', 'load_transfer_small')
COUNTS = (30, 40, 360)
PAIRS = 3 * sum(COUNTS[1:])
SELECTION_RULE = 'mean_of_five_best_validation_window_mean_ce; exact_tie_lower_lambda'
CASE_KEYS = ('domain', 'ratio', 'subset', 'protocol', 'variant', 'seed')


def selection_cases():
    for variant in VARIANTS:
        for weight in (GRID if variant == 'data_aux_ce' else [0.]):
            for seed in SEEDS:
                yield dict(domain=WINDOW, ratio=1., subset=0, protocol='clean_only',
                           variant=variant, seed=seed), weight


def transfer_cases(full):
    for source in SOURCES:
        for ratio in ([1.] if full else RATIOS):
            for subset in ([0] if full else SUBSETS):
                for variant in VARIANTS:
                    for seed in SEEDS:
                        yield dict(domain=f'load_source_{source}', ratio=ratio, subset=subset,
                                   protocol='clean_only', variant=variant, seed=seed)


def read_dataset():
    records, original, raw = R.load_dataset(DATA)
    if len(records) != 40 or len(original) != 687:
        raise ValueError('原始数据须包含40条记录及687个完整窗')
    original['load_hp'] = original.record_id.str.rsplit('_', n=1).str[-1].astype(int)
    return records, original, R.U.normalize(raw)


def fixed_splits(original, root, allocations=None, fixed_source=FIXED_SPLITS):
    tables, support = {}, []
    if allocations is None:
        allocations = [(WINDOW, 1., 0)]
        allocations += list(dict.fromkeys((c['domain'], c['ratio'], c['subset'])
                                          for full in (True, False) for c in transfer_cases(full)))
    keys = ['window_index', 'record_id', 'record_index', 'load_hp', 'label', 'rpm',
            'start_sample', 'end_sample']
    for domain, ratio, subset in allocations:
        name = split_name(domain, ratio, subset)
        source_dir = fixed_source / name
        w = read_fixed(source_dir)
        pd.testing.assert_frame_equal(w[keys], original[keys], check_dtype=False)
        full = read_fixed(BASE / 'splits' / domain)
        pd.testing.assert_frame_equal(w, full if ratio == 1. else nested_subset(full, ratio, subset))
        roles = {role: w.role.eq(role) for role in ['train', 'validation']}
        if domain == WINDOW:
            roles['test'] = w.role.eq('test')
            if [int(mask.sum()) for mask in roles.values()] != [481, 103, 103]:
                raise ValueError('随机窗口划分数量异常')
        else:
            source = int(domain[-1])
            if set(w.loc[roles['train'] | roles['validation'], 'load_hp']) != {source}:
                raise ValueError('目标负载进入了来源训练或验证')
            roles.update({f'test_hp{t}': w.load_hp.eq(t) for t in SOURCES if t != source})
        for role, mask in roles.items():
            labels = np.bincount(w.loc[mask, 'label'], minlength=4)
            saved_ids = np.loadtxt(source_dir / f'{role}.txt', dtype=int, ndmin=1)
            np.testing.assert_array_equal(saved_ids, w.loc[mask, 'window_index'])
            if len(labels) != 4 or min(labels) <= 0:
                raise ValueError(f'{name}/{role}未覆盖四类')
            support.append(dict(domain=domain, ratio=ratio, subset=subset, role=role,
                                windows=int(mask.sum()), **dict(zip(LABELS, labels))))
        dest = root / 'splits' / name
        dest.mkdir(parents=True, exist_ok=False)
        for file in ['windows.csv', *(f'{role}.txt' for role in roles)]:
            shutil.copyfile(source_dir / file, dest / file)
        tables[name] = w
    pd.DataFrame(support).to_csv(root / 'dataset_support.csv', index=False)
    return tables


def configure():
    from v4_physics import bind_v4_physics

    if not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA。请使用项目pytorch_GPU环境')
    device = torch.device('cuda')
    R.U.DEVICE = device
    R.U.configure()
    bind_v4_physics(engine)
    if R.U.CFG['bands_hz'] != BANDS:
        raise ValueError('实际频带与V4协议不符')
    return device


def selection_report(root, jobs, rows, weight):
    out = root / 'selection'
    out.mkdir()
    sensitivity = pd.DataFrame(rows)
    validation = validation_table(jobs)
    if len(validation) != COUNTS[0] or set(validation.condition) != {'clean'}:
        raise ValueError('选参验证结果不全')
    summary = validation.groupby(['variant', 'aux_weight'])[METRICS + ['validation_nll']].agg(
        ['mean', 'std', 'count'])
    summary.columns = ['_'.join(keys) for keys in summary.columns]
    if not summary['acc_count'].eq(len(SEEDS)).all():
        raise ValueError('每组选参验证结果须有5种子')
    sensitivity.to_csv(out / 'lambda_sensitivity.csv', index=False)
    validation.to_csv(out / 'validation_metrics.csv', index=False)
    summary.reset_index().to_csv(out / 'mean_std.csv', index=False)
    (out / 'REPORT.md').write_text(
        '# V4 全样本纯干净选参\n\n'
        f'30 个模型完成训练和验证。统一辅助系数 λ 为 {weight}。'
        '每档比较 5 个种子各自最佳验证交叉熵的均值，数值相同时取较小系数。'
        '仅使用固定 103 个验证窗口。本批模型仅报告验证结果。\n\n'
        + sensitivity.round(8).to_markdown(index=False) + '\n', encoding='utf-8')


class V4Experiment(Experiment):
    def __init__(self, root):
        self.root = Path(root).resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith('clean_'):
            raise ValueError('须使用experiments/CWRU/实验结果/CWRU_V4下全新的clean_*直接子目录')
        self.root.mkdir(parents=True, exist_ok=False)
        self.jobs = []
        self.state = dict(
            version='CWRU_V4', stage='preparing', started_at=now(), pid=os.getpid(),
            planned_models=sum(COUNTS), planned_stage_models=dict(zip(STAGES, COUNTS)),
            planned_test_pairs=PAIRS, planned_metric_rows=PAIRS, planned_confusion_cells=16*PAIRS,
            evaluated_pairs=0, metric_rows=0, lambda_grid=GRID, seeds=SEEDS, ratios=RATIOS, subsets=SUBSETS,
            selected_aux_weight=None, lambda_locked=False, selection_rule=SELECTION_RULE,
            selection_test_accessed=False, target_test_accessed=False, decisions_locked=False,
            historical_test_exposure=True, protocol='clean_only', comparison_scope=VARIANTS,
            fixed_split_source=str(FIXED_SPLITS), bands_hz=BANDS, sampling_hz=48000,
            features_per_band=53, training_spec=engine.normalize_spec(), jobs=self.jobs)
        shutil.copyfile(HERE / 'PROTOCOL.md', self.root / 'PROTOCOL.md')
        self.persist()

    def setup(self, allocations=None, fixed_source=FIXED_SPLITS):
        self.device = configure()
        records, original, self.x = read_dataset()
        self.tables = fixed_splits(original, self.root, allocations, fixed_source)
        records.to_csv(self.root / 'records.csv', index=False)
        (self.root / 'DATA_USAGE.md').write_text(
            '# V4 数据使用\n\n全部 40 条原始记录，687 个完整 0.5 秒窗口。'
            '窗口边界、转速、训练和验证索引见 splits，逐类配额见 dataset_support.csv。'
            '每个模型实际训练窗口见其 selected_windows.csv。\n', encoding='utf-8')
        self.state['hardware'] = torch.cuda.get_device_name(self.device)
        self.persist()

    def run(self):
        self.setup()
        for case, weight in selection_cases():
            self.ensure(case, weight if case['variant'] == 'data_aux_ce' else .1, STAGES[0])
        weight, rows = select_lambda(self.jobs)
        selection_report(self.root, self.jobs, rows, weight)
        write_json(self.root / 'selection.json', dict(selected_aux_weight=weight, candidates=rows,
                   rule=SELECTION_RULE, test_accessed=False, max_epochs=200))
        self.state.update(stage='lambda_locked', lambda_locked=True,
                          selected_aux_weight=weight, lambda_locked_at=now())
        self.persist()
        print(f'LAMBDA LOCKED {weight}', flush=True)
        for full, stage in zip((True, False), STAGES[1:]):
            for case in transfer_cases(full):
                self.ensure(case, weight, stage)
        expected = [(STAGES[0], case, coefficient) for case, coefficient in selection_cases()]
        expected += [(stage, case, weight if case['variant'] == 'data_aux_ce' else 0.)
                     for full, stage in zip((True, False), STAGES[1:]) for case in transfer_cases(full)]
        self.verify_models(expected)
        self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def verify_models(self, expected):
        for job, (stage, case, coefficient) in zip(self.jobs, expected, strict=True):
            if (job['stage'] != stage or job['aux_weight'] != coefficient
                    or any(job[key] != case[key] for key in CASE_KEYS) or not job['completed']):
                raise ValueError(f'模型登记与预先计划不一致 {job["job_id"]}')
            folder = resolve_result_path(job['folder'])
            for file in ('parameters.json', 'training.csv', 'validation_curve.csv', 'training_spec.json',
                         'validation_predictions.npz', 'validation_metrics.csv', 'selected_windows.csv'):
                if not (folder / file).is_file():
                    raise FileNotFoundError(folder / file)
            checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            if (checkpoint['variant'] != case['variant'] or checkpoint['model_config']['bands_hz'] != BANDS
                    or (case['variant'] == 'data_aux_ce' and checkpoint['suite_spec']['aux_weight'] != coefficient)):
                raise ValueError(f'权重配置与登记不一致 {job["job_id"]}')
            del checkpoint
        self.state['weights_verified'] = True

    def evaluate(self, finish=None):
        if finish is None:
            from v4_report import finish

        if (not self.state['decisions_locked'] or not self.state.get('weights_verified')
                or self.state['target_test_accessed']):
            raise ValueError('所有模型锁定后才可执行一次目标测试')
        transfer_jobs = [j for j in self.jobs if j['stage'] in STAGES[1:]]
        planned = self.state['planned_test_pairs']
        if 3 * len(transfer_jobs) != planned:
            raise ValueError('跨负载模型数与计划不一致')
        dest = self.root / 'evaluation'
        predictions = dest / 'predictions'
        predictions.mkdir(parents=True, exist_ok=False)
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        evaluated = 0
        for job in transfer_jobs:
            jid = job['job_id']
            source = job['source_load_hp']
            w = self.tables[split_name(job['domain'], job['ratio'], job['subset'])]
            model = engine.load(job['folder'], self.device)
            metrics, confusion, evaluations = [], [], []
            for target in SOURCES:
                if target == source:
                    continue
                ix = np.flatnonzero(w.load_hp.eq(target))
                labels = w.iloc[ix].label.to_numpy()
                p = R.predict(model, self.x[ix], w.iloc[ix].rpm.to_numpy(np.float32), self.device)
                values, counts = scores(labels, p)
                meta = dict({k: job[k] for k in ['job_id', *CASE_KEYS, 'aux_weight', 'source_load_hp']},
                            target_load_hp=target)
                condition = dict(meta, condition='clean', noise_seed=0)
                path = predictions / f'{jid}_target{target}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, labels=labels, window_indices=w.iloc[ix].window_index.to_numpy(),
                                        clean_n0=p)
                evaluations.append(dict(meta, prediction_path=str(path), test_windows=len(ix)))
                metrics.append(dict(condition, **values))
                for a in range(len(LABELS)):
                    for b in range(len(LABELS)):
                        confusion.append(dict(condition, true_label=a, predicted_label=b, count=int(counts[a, b]),
                                              support=int(counts[a].sum()),
                                              row_percent=float(100*counts[a, b]/counts[a].sum())))
            del model
            for filename, rows in [('metrics.csv', metrics), ('confusion_matrices.csv', confusion),
                                   ('evaluations.csv', evaluations)]:
                path = dest / filename
                pd.DataFrame(rows).to_csv(path, index=False, mode='a', header=not path.exists())
            evaluated += len(evaluations)
            self.state.update(evaluated_pairs=evaluated, metric_rows=evaluated)
            self.persist()
        if evaluated != planned:
            raise ValueError('目标评价数量与计划不一致')
        self.state['stage'] = 'reporting'
        self.persist()
        finish(self.root)
        self.state.update(stage='complete', completed_at=now())
        self.persist()


def check():
    device = configure()
    _, original, x = read_dataset()
    root = (RESULTS / f'_v4_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve():
        raise ValueError('功能检查目录超出V4结果范围')
    root.mkdir(parents=True, exist_ok=False)
    try:
        tables = fixed_splits(original, root)
        w = tables[split_name('load_source_0', .1, SUBSETS[0])]
        mask = w.role.isin(['train', 'validation']).to_numpy()
        normalize = engine.normalize_spec

        def one_epoch(spec=None):
            settings = normalize(dict(spec or {}, max_epochs=200))
            return dict(settings, max_epochs=1)

        with patch.object(engine, 'normalize_spec', one_epoch):
            for variant in VARIANTS:
                folder = root / variant
                info = engine.train(x[mask], w.loc[mask].reset_index(drop=True), 'clean_only', variant,
                                    SEEDS[0], SUBSETS[0], folder, device, dict(aux_weight=.3))
                if not info['completed'] or info['epochs_trained'] != 1:
                    raise ValueError(f'{variant}真实数据功能检查未完整执行')
                checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
                if checkpoint['model_config']['bands_hz'] != BANDS:
                    raise ValueError('权重未记录真实频带')
    finally:
        shutil.rmtree(root)
    print('CHECK PASSED: 41 fixed splits; both variants trained, saved, reloaded and validated on source data.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print(f'{COUNTS[0]} selection + {COUNTS[1]} full + {COUNTS[2]} small = {sum(COUNTS)} new models; '
              f'{PAIRS} clean transfer tests after all model selection; selection validation only.')
    elif args.command == 'check':
        check()
    else:
        if args.output is None:
            parser.error('run需要指定--output全新目录')
        from v4_report import finish
        if not callable(finish):
            raise TypeError('报告入口不可调用')
        RESULTS.mkdir(parents=True, exist_ok=True)
        lock = RESULTS / '.clean_run.lock'
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(f'{os.getpid()} {args.output.resolve()}\n')
        try:
            experiment = V4Experiment(args.output)
            with (experiment.root / 'training_console.log').open('x', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        experiment.run()
                    except BaseException as error:
                        experiment.state.update(stage='failed', error=str(error), error_type=type(error).__name__)
                        traceback.print_exc()
                        experiment.persist()
                        raise
        finally:
            lock.unlink()


if __name__ == '__main__':
    main()
