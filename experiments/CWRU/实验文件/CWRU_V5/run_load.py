"""V5 39 维纯干净跨负载实验。只训练经批准的 λ=0.3 融合模型。"""

import argparse
import json
import math
import os
import shutil
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch  # Windows 下先加载 PyTorch。
import numpy as np
import pandas as pd

from .run_selection import HERE, PROTOCOL, RESULTS, VARIANT, V3_SPLITS, verify_validation
from .v5_model import MODEL_CONFIG, engine, load, train
from experiments.CWRU.实验文件.CWRU_V4.run_v4 import fixed_splits, read_dataset
from metrics import LABELS, scores
from run_experiment import Experiment, now, select_lambda, split_name, write_json

# 报告复用 V4 的逐预测核验。先完成 V3/V4 路径初始化，再导入报告。
from . import v5_report

from experiment_paths import resolve_result_path


SOURCES = tuple(range(4))
BUDGETS = (1., .1, .2, .5)
SEEDS = v5_report.SEEDS
SUBSETS = v5_report.SUBSETS
WEIGHT = .3
SELECTION = RESULTS / 'selection_20261109_114155'
ALLOCATIONS = tuple((f'load_source_{source}', ratio, subset)
                    for ratio in BUDGETS for source in SOURCES
                    for subset in ((0,) if ratio == 1. else SUBSETS))
CASES = tuple(dict(domain=domain, ratio=ratio, subset=subset, seed=seed,
                   variant=VARIANT, protocol=PROTOCOL, source_load_hp=int(domain[-1]))
              for domain, ratio, subset in ALLOCATIONS for seed in SEEDS)
PAIRS = len(CASES) * 3
TRAIN_COUNTS = {1.: (73, 156, 160, 160), .1: (9, 17, 17, 17),
                .2: (16, 33, 34, 34), .5: (37, 78, 80, 80)}
VALIDATION_COUNTS = (19, 39, 40, 40)
TARGET_COUNTS = (92, 195, 200, 200)


def approved_decision():
    state = json.loads((SELECTION / 'config.json').read_text(encoding='utf-8'))
    saved = json.loads((SELECTION / 'selection/selection.json').read_text(encoding='utf-8'))
    if (state['stage'] != 'awaiting_transfer_approval' or state['lambda_locked'] is not True
            or state['weights_verified'] is not True or state['model_config'] != MODEL_CONFIG
            or state['trained_models'] != 25 or len(state['jobs']) != 25
            or state['transfer_models'] != 0 or state['selection_test_accessed'] is not False
            or state['target_test_accessed'] is not False):
        raise ValueError('选参冻结现场与审批范围不符')
    best, rows = select_lambda(state['jobs'])
    if best != 1. or state['selected_aux_weight'] != best or saved['selected_aux_weight'] != best or saved['candidates'] != rows:
        raise ValueError('原始五档选参记录与复算不一致')
    constrained = min((row for row in rows if 0 < row['aux_weight'] < 1),
                      key=lambda row: (row['mean_best_validation_ce'], row['aux_weight']))
    if constrained['aux_weight'] != WEIGHT:
        raise ValueError('符合主辅约束的已有验证结果与获批系数不符')
    return dict(approved_aux_weight=WEIGHT, original_grid_winner=best,
                approved_scope='39_dim_clean_only_full_and_10_20_50_percent',
                approval='user_confirmed_lambda_0.3_and_transfer',
                constraint='0 < auxiliary loss weight < fusion loss weight of 1',
                constraint_set_after_original_validation=True,
                source=str(SELECTION / 'selection/selection.json'), constrained_validation=constrained)


def setup(root, allocations=ALLOCATIONS):
    if not torch.cuda.is_available():
        raise RuntimeError('请在 pytorch_GPU 环境运行 V5 跨负载实验')
    device = torch.device('cuda')
    engine.R.U.configure()
    records, original, x = read_dataset()
    tables = fixed_splits(original, root, allocations, V3_SPLITS)
    budgets = {ratio for _, ratio, _ in allocations}
    v5_report._same_splits(root, v5_report.FULL_REFERENCE, tuple(group for group in v5_report.FULL if group[0] in budgets))
    v5_report._same_splits(root, v5_report.SMALL_REFERENCE, tuple(group for group in v5_report.SMALL if group[0] in budgets))
    for domain, ratio, subset in allocations:
        source = int(domain[-1])
        table = tables[split_name(domain, ratio, subset)]
        if (int(table.role.eq('train').sum()) != TRAIN_COUNTS[ratio][source]
                or int(table.role.eq('validation').sum()) != VALIDATION_COUNTS[source]
                or tuple(int(table.load_hp.eq(hp).sum()) for hp in SOURCES) != TARGET_COUNTS):
            raise ValueError(f'{domain}/{ratio}/{subset} 来源训练、验证或目标窗数异常')
    records.to_csv(root / 'records.csv', index=False)
    return device, tables, x


def source_data(x, table):
    mask = table.role.isin(['train', 'validation']).to_numpy()
    return x[mask], table.loc[mask].reset_index(drop=True)


class Transfer(Experiment):
    cases = CASES
    first_job_number = 1
    validation_predictions = ('clean_n0',)

    def __init__(self, root, decision):
        self.root = root.resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith('load_clean_'):
            raise ValueError('须使用 experiments/CWRU/实验结果/CWRU_V5 下全新的 load_clean_* 目录')
        self.root.mkdir(exist_ok=False)
        self.jobs = []
        self.spec = engine.normalize_spec(dict(aux_weight=WEIGHT, physics_fit='clean'))
        self.state = dict(version='CWRU_V5_39_load_clean', stage='preparing', started_at=now(), pid=os.getpid(),
                          planned_models=len(CASES), planned_test_pairs=PAIRS, planned_metric_rows=PAIRS,
                          planned_confusion_cells=16 * PAIRS, evaluated_pairs=0, metric_rows=0,
                          jobs=self.jobs, selected_aux_weight=WEIGHT, lambda_locked=True,
                          decisions_locked=False, weights_verified=False, target_test_accessed=False,
                          historical_test_exposure=True, selection_test_accessed=False,
                          protocol=PROTOCOL, ratios=BUDGETS, seeds=SEEDS, subsets=SUBSETS,
                          model_config=MODEL_CONFIG, training_spec=self.spec, fixed_split_source=str(V3_SPLITS),
                          automatic_resume=False, lambda_decision=decision)
        shutil.copyfile(HERE / 'PHYSICS_39.md', self.root / 'PHYSICS_39.md')
        write_json(self.root / 'lambda_decision.json', decision)
        self.persist()

    def prepare(self):
        self.state['frozen_reference_preflight'] = v5_report.preflight()
        self.device, self.tables, self.x = setup(self.root)
        self.state.update(hardware=torch.cuda.get_device_name(self.device), split_count=len(self.tables))

    def run(self):
        self.prepare()
        for index, case in enumerate(self.cases, start=self.first_job_number):
            job_id = f'm{index:04d}'
            folder = self.root / 'models' / job_id
            stage = 'load_transfer_full' if case['ratio'] == 1. else 'load_transfer_small'
            self.state.update(stage=stage, active_model=dict(job_id=job_id, **case))
            self.persist()
            table = self.tables[split_name(case['domain'], case['ratio'], case['subset'])]
            x, source = source_data(self.x, table)
            info = train(x, source, case['protocol'], case['variant'], case['seed'], case['subset'], folder, self.device, self.spec)
            self.jobs.append(dict(info, **{key: value for key, value in case.items() if key not in info},
                                  job_id=job_id, folder=str(folder), stage=stage, aux_weight=WEIGHT))
            self.state.pop('active_model')
            self.persist()
            print('MODEL COMPLETE', job_id, case, flush=True)
        self.state.update(stage='verifying_weights')
        self.persist()
        self.verify_weights()
        self.state.update(stage='decisions_locked', decisions_locked=True, weights_verified=True, decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def verify_weights(self):
        for job, case in zip(self.jobs, self.cases, strict=True):
            if (not job['completed'] or not job['from_scratch'] or job['aux_weight'] != WEIGHT
                    or any(job[key] != value for key, value in case.items())):
                raise ValueError(f'模型与计划不符 {job["job_id"]}')
            folder = resolve_result_path(job['folder'])
            for file in ('parameters.json', 'training.csv', 'training_spec.json', 'validation_curve.csv',
                         'validation_metrics.csv', 'validation_predictions.npz', 'selected_windows.csv'):
                if not (folder / file).is_file():
                    raise FileNotFoundError(folder / file)
            table = self.tables[split_name(job['domain'], job['ratio'], job['subset'])]
            expected = table.loc[table.role.eq('train')].reset_index(drop=True)
            pd.testing.assert_frame_equal(pd.read_csv(folder / 'selected_windows.csv'), expected, check_dtype=False)
            validation = table.loc[table.role.eq('validation')]
            with np.load(folder / 'validation_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['window_indices'], validation.window_index)
                np.testing.assert_array_equal(saved['labels'], validation.label)
                if (set(saved.files) != {'window_indices', 'labels', *self.validation_predictions}
                        or any(not np.isfinite(saved[key]).all() for key in self.validation_predictions)):
                    raise ValueError(f'验证预测条件异常或含非有限值 {job["job_id"]}')
            model = load(folder, 'cpu')
            if (model.variant != VARIANT or model.spec != self.spec
                    or model.network.physics[0].in_features != 39
                    or not all(torch.isfinite(value).all() for value in model.state_dict().values())
                    or not (model.evidence.scale >= .05).all()):
                raise ValueError(f'模型权重或标准化参数异常 {job["job_id"]}')

    def evaluate(self):
        if (len(self.jobs) != len(CASES) or not self.state['weights_verified']
                or not self.state['decisions_locked'] or self.state['target_test_accessed']):
            raise ValueError('须核验全部已锁定模型后，才可执行一次正式目标测试')
        destination = self.root / 'evaluation'
        predictions = destination / 'predictions'
        predictions.mkdir(parents=True, exist_ok=False)
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        evaluated = 0
        for job in self.jobs:
            table = self.tables[split_name(job['domain'], job['ratio'], job['subset'])]
            model = load(job['folder'], self.device)
            metrics, confusion, evaluations = [], [], []
            for target in SOURCES:
                if target == job['source_load_hp']:
                    continue
                indices = np.flatnonzero(table.load_hp.eq(target))
                rows = table.iloc[indices]
                labels = rows.label.to_numpy()
                probability = engine.R.predict(model, self.x[indices], rows.rpm.to_numpy(np.float32), self.device)
                values, counts = scores(labels, probability)
                meta = {key: job[key] for key in ('job_id', *v5_report.META)}
                meta['target_load_hp'] = target
                condition = dict(meta, condition='clean', noise_seed=0)
                path = predictions / f'{job["job_id"]}_target{target}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, labels=labels, window_indices=rows.window_index.to_numpy(),
                                        clean_n0=probability)
                evaluations.append(dict(meta, prediction_path=str(path), test_windows=len(rows)))
                metrics.append(dict(condition, **values))
                for a in range(len(LABELS)):
                    support = int(counts[a].sum())
                    for b in range(len(LABELS)):
                        confusion.append(dict(condition, true_label=a, predicted_label=b, count=int(counts[a, b]),
                                              support=support, row_percent=float(100 * counts[a, b] / support)))
            del model
            for filename, entries in [('evaluations.csv', evaluations), ('metrics.csv', metrics),
                                      ('confusion_matrices.csv', confusion)]:
                path = destination / filename
                pd.DataFrame(entries).to_csv(path, index=False, mode='a', header=not path.exists())
            evaluated += len(evaluations)
            self.state.update(evaluated_pairs=evaluated, metric_rows=evaluated)
            self.persist()
            print('TEST COMPLETE', job['job_id'], flush=True)
        if evaluated != PAIRS:
            raise ValueError('实际目标组合数与计划不符')
        self.state.update(stage='reporting')
        self.persist()
        v5_report.finish(self.root)
        self.state.update(stage='complete', completed_at=now())
        self.persist()


def check():
    root = (RESULTS / f'_transfer_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve() or not root.name.startswith('_transfer_check_'):
        raise ValueError('功能检查目录超出 V5 结果范围')
    root.mkdir(exist_ok=False)
    try:
        approved_decision()
        v5_report.preflight()
        device, tables, x = setup(root)
        normalize = engine.normalize_spec

        def one_epoch(spec=None):
            return dict(normalize(dict(spec or {}, max_epochs=200)), max_epochs=1)

        with patch.object(engine, 'normalize_spec', one_epoch):
            for ratio, subset in ((1., 0), (.1, SUBSETS[0])):
                table = tables[split_name('load_source_0', ratio, subset)]
                observations, source = source_data(x, table)
                folder = root / f'check_{ratio}'
                info = train(observations, source, PROTOCOL, VARIANT, SEEDS[0], subset, folder, device, dict(aux_weight=WEIGHT))
                ce = verify_validation(folder, device, observations, source, dict(aux_weight=WEIGHT))
                if (not info['completed'] or info['epochs_trained'] != 1
                        or info['training_windows'] != TRAIN_COUNTS[ratio][0]
                        or not math.isclose(ce, info['validation_nll'], rel_tol=0, abs_tol=2e-6)):
                    raise ValueError(f'{ratio:g} 真实来源单轮训练或重载核验失败')
    finally:
        shutil.rmtree(root)
    print('CHECK PASSED: 40 fixed splits, reference preflight, real 0 HP full/10% training and reload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print(f'{len(CASES)} new models, lambda {WEIGHT}, full/10/20/50%, {PAIRS} tests after all weights are locked.')
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.command == 'check':
        check()
        return
    if args.output is None:
        parser.error('run 须指定新的 --output')
    decision = approved_decision()
    lock = RESULTS / '.load_clean.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        experiment = Transfer(args.output, decision)
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
