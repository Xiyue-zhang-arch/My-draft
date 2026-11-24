"""V5 39 维记录内随机窗口实验，固定 λ=0.3。"""

import argparse
import json
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

from .run_load import (HERE, RESULTS, SELECTION, VARIANT, WEIGHT, Experiment, approved_decision,
                       engine, load, now, read_dataset, scores, source_data, split_name, train, write_json)
from .v5_model import MODEL_CONFIG
from run_experiment import BASE, RATIOS, SEEDS, SUBSETS, nested_subset
from splits import read_fixed

from experiment_paths import resolve_result_path


REFERENCE = RESULTS.parent / 'CWRU_V3/window_random_20261023_0853'
DOMAIN = 'window_random'
CLEAN = 'clean_only'
MIXED = 'awgn_mixed'
PREFIX = 'window_random_'
ALLOCATION = ((1., 0), *((ratio, subset) for ratio in RATIOS for subset in SUBSETS))
CASES = tuple(dict(domain=DOMAIN, ratio=ratio, subset=subset, protocol=protocol, variant=VARIANT, seed=seed)
              for ratio, subset in ALLOCATION for protocol in (CLEAN, MIXED) for seed in SEEDS)
NEW_CASES = CASES[5:]
TRAIN_COUNTS = {1.: 481, .1: 49, .2: 98, .5: 242}
META = ('job_id', 'domain', 'ratio', 'subset', 'protocol', 'variant', 'seed', 'aux_weight', 'source_load_hp', 'origin')
MODEL_COUNT = 100
METRIC_COUNT = 1350


def spec_for(protocol):
    return engine.normalize_spec(dict(aux_weight=WEIGHT, physics_fit='clean' if protocol == CLEAN else 'matched_noise'))


def reference_preflight():
    state = json.loads((REFERENCE / 'config.json').read_text(encoding='utf-8'))
    index = pd.read_csv(REFERENCE / 'evaluation/evaluations.csv')
    keys = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed']
    expected = {tuple(case[key] if key != 'variant' else variant for key in keys)
                for case in CASES for variant in ('baseline_data', VARIANT)}
    if (state['stage'] != 'complete' or state['selected_aux_weight'] != WEIGHT or state['metric_rows'] != 2700
            or len(index) != 200 or set(index[keys].itertuples(index=False, name=None)) != expected
            or not index.test_windows.eq(103).all() or not all(resolve_result_path(path).is_file() for path in index.prediction_path)):
        raise ValueError('冻结 V3 窗口对照不完整或与 V5 计划不匹配')
    return dict(models=len(index), metric_rows=state['metric_rows'])


def reference_jobs():
    approved_decision()
    state = json.loads((SELECTION / 'config.json').read_text(encoding='utf-8'))
    full = split_name(DOMAIN, 1., 0)
    for file in ('windows.csv', 'train.txt', 'validation.txt', 'test.txt'):
        if (SELECTION / 'splits' / full / file).read_bytes() != (REFERENCE / 'splits' / full / file).read_bytes():
            raise ValueError(f'V5 选参划分与 V3 全样本划分不一致 {file}')
    jobs = sorted((job for job in state['jobs'] if job['aux_weight'] == WEIGHT), key=lambda job: job['seed'])
    references = []
    for number, (job, case) in enumerate(zip(jobs, CASES[:5], strict=True), start=1):
        folder = resolve_result_path(job['folder']).resolve()
        if (job['job_id'] != f'm{15 + number:04d}' or any(job[key] != value for key, value in case.items())
                or folder != SELECTION / 'models' / job['job_id'] or not (folder / 'model.pt').is_file()
                or job['spec'] != spec_for(CLEAN) or not job['completed']):
            raise ValueError(f'选参引用权重不符合全样本纯干净条件 {folder}')
        references.append(dict(job, source_job_id=job['job_id'], job_id=f'r{number:04d}',
                               origin='selection_reference', source_load_hp=-1))
    return references


def setup(root):
    if not torch.cuda.is_available():
        raise RuntimeError('请在 pytorch_GPU 环境运行窗口实验')
    device = torch.device('cuda')
    engine.R.U.configure()
    records, original, x = read_dataset()
    original_fields = ['window_index', 'record_id', 'record_index', 'load_hp', 'label', 'rpm',
                       'start_sample', 'end_sample']
    full = read_fixed(BASE / 'splits' / DOMAIN)
    tables, supports = {}, []
    for ratio, subset in ALLOCATION:
        name = split_name(DOMAIN, ratio, subset)
        source = REFERENCE / 'splits' / name
        table = read_fixed(source)
        pd.testing.assert_frame_equal(table[original_fields], original[original_fields], check_dtype=False)
        pd.testing.assert_frame_equal(table, full if ratio == 1. else nested_subset(full, ratio, subset))
        destination = root / 'splits' / name
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source / 'windows.csv', destination / 'windows.csv')
        for role, size in [('train', TRAIN_COUNTS[ratio]), ('validation', 103), ('test', 103)]:
            rows = table[table.role.eq(role)]
            path = source / f'{role}.txt'
            np.testing.assert_array_equal(np.loadtxt(path, dtype=int, ndmin=1), rows.window_index)
            if len(rows) != size or set(rows.label) != set(range(4)):
                raise ValueError(f'{name}/{role} 的窗口配额或类别不完整')
            supports.append(dict(domain=DOMAIN, ratio=ratio, subset=subset, role=role, windows=len(rows),
                                 **dict(zip(('N', 'IR', 'OR', 'B'), np.bincount(rows.label, minlength=4)))))
            shutil.copyfile(path, destination / path.name)
        tables[name] = table
    records.to_csv(root / 'records.csv', index=False)
    pd.DataFrame(supports).to_csv(root / 'dataset_support.csv', index=False)
    return device, tables, x


def verify_model(job, table):
    folder = resolve_result_path(job['folder'])
    spec = spec_for(job['protocol'])
    if (job['spec'] != spec or not job['completed'] or not job['from_scratch'] or job['aux_weight'] != WEIGHT
            or job['training_windows'] != TRAIN_COUNTS[job['ratio']] or job['validation_windows'] != 103):
        raise ValueError(f'模型训练记录与计划不一致 {job["job_id"]}')
    expected = table[table.role.eq('train')].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(folder / 'selected_windows.csv'), expected, check_dtype=False)
    validation = table[table.role.eq('validation')]
    expected_keys = {'clean_n0'}
    if job['protocol'] == MIXED:
        expected_keys |= {f'white_{snr:+d}_n6201' for snr in (-4, -2, 0, 2, 4)}
    with np.load(folder / 'validation_predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['window_indices'], validation.window_index)
        np.testing.assert_array_equal(saved['labels'], validation.label)
        if (set(saved.files) != {'labels', 'window_indices', *expected_keys}
                or any(saved[key].shape != (103, 4) or not np.isfinite(saved[key]).all() for key in expected_keys)):
            raise ValueError(f'验证预测异常 {job["job_id"]}')
    model = load(folder, 'cpu')
    if (model.variant != VARIANT or model.spec != spec or model.network.physics[0].in_features != 39
            or not all(torch.isfinite(value).all() for value in model.state_dict().values())
            or not (model.evidence.scale >= .05).all()):
        raise ValueError(f'模型权重或标准化参数异常 {job["job_id"]}')


class Window(Experiment):
    def __init__(self, root, references):
        self.root = root.resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith(PREFIX):
            raise ValueError('须使用 experiments/CWRU/实验结果/CWRU_V5 下全新的 window_random_* 目录')
        self.root.mkdir(exist_ok=False)
        self.jobs = references.copy()
        self.state = dict(version='CWRU_V5_39_window_random', stage='preparing', started_at=now(), pid=os.getpid(),
                          jobs=self.jobs, planned_models=MODEL_COUNT, planned_new_models=len(NEW_CASES),
                          planned_referenced_models=len(references), planned_test_pairs=MODEL_COUNT,
                          planned_metric_rows=METRIC_COUNT, planned_confusion_cells=16 * METRIC_COUNT,
                          evaluated_pairs=0, metric_rows=0, selected_aux_weight=WEIGHT, lambda_locked=True,
                          decisions_locked=False, weights_verified=False, target_test_accessed=False,
                          historical_test_exposure=True, selection_test_already_accessed=False, automatic_resume=False,
                          ratios=[1., *RATIOS], seeds=SEEDS, subsets=SUBSETS, model_config=MODEL_CONFIG,
                          fixed_split_source=str(REFERENCE / 'splits'), reference_source=str(SELECTION),
                          comparison_source=str(REFERENCE), noise_diagnostic=False,
                          training_specs={protocol: spec_for(protocol) for protocol in (CLEAN, MIXED)})
        shutil.copyfile(HERE / 'PHYSICS_39.md', self.root / 'PHYSICS_39.md')
        shutil.copyfile(HERE / 'WINDOW_PROTOCOL.md', self.root / 'PROTOCOL.md')
        self.persist()

    def persist(self):
        references = sum(job['origin'] == 'selection_reference' for job in self.jobs)
        self.state.update(updated_at=now(), trained_models=len(self.jobs) - references,
                          referenced_models=references, completed_model_cases=len(self.jobs))
        write_json(self.root / 'config.json', self.state)
        pd.DataFrame(self.jobs).to_csv(self.root / 'models_registry.csv', index=False)

    def run(self):
        self.state['reference_preflight'] = reference_preflight()
        self.device, self.tables, self.x = setup(self.root)
        self.state.update(hardware=torch.cuda.get_device_name(self.device), split_count=len(self.tables),
                          stage='verifying_referenced_weights')
        self.persist()
        for job in self.jobs:
            verify_model(job, self.tables[split_name(DOMAIN, job['ratio'], job['subset'])])
        for number, case in enumerate(NEW_CASES, start=1):
            job_id = f'm{number:04d}'
            folder = self.root / 'models' / job_id
            self.state.update(stage='window_random_training', active_model=dict(job_id=job_id, **case))
            self.persist()
            table = self.tables[split_name(DOMAIN, case['ratio'], case['subset'])]
            x, w = source_data(self.x, table)
            info = train(x, w, case['protocol'], VARIANT, case['seed'], case['subset'], folder,
                         self.device, spec_for(case['protocol']))
            self.jobs.append(dict(info, **{key: value for key, value in case.items() if key not in info},
                                  job_id=job_id, folder=str(folder), origin='new', source_load_hp=-1,
                                  aux_weight=WEIGHT, stage='window_random_training'))
            self.state.pop('active_model')
            self.persist()
            print('MODEL COMPLETE', job_id, case, flush=True)
        self.state.update(stage='verifying_weights')
        self.persist()
        for job, case in zip(self.jobs, CASES, strict=True):
            if any(job[key] != value for key, value in case.items()):
                raise ValueError(f'窗口模型与预定计划不符 {job["job_id"]}')
            verify_model(job, self.tables[split_name(DOMAIN, job['ratio'], job['subset'])])
        self.state.update(stage='decisions_locked', decisions_locked=True, weights_verified=True, decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def evaluate(self):
        from . import v5_window_report

        if (len(self.jobs) != MODEL_COUNT or not self.state['decisions_locked'] or not self.state['weights_verified']
                or self.state['target_test_accessed']):
            raise ValueError('须核验全部 100 个权重后才可统一测试一次')
        destination = self.root / 'evaluation'
        predictions = destination / 'predictions'
        predictions.mkdir(parents=True, exist_ok=False)
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        table = self.tables[split_name(DOMAIN, 1., 0)]
        indices = np.flatnonzero(table.role.eq('test'))
        rows = table.iloc[indices]
        labels, rpm, ids = rows.label.to_numpy(), rows.rpm.to_numpy(np.float32), rows.window_index.to_numpy()
        views = list(engine.R.conditions(self.x[indices], MIXED, True))
        metric_rows = 0
        for number, job in enumerate(self.jobs, start=1):
            model = load(job['folder'], self.device)
            arrays = dict(labels=labels, window_indices=ids)
            meta = dict({key: job[key] for key in META}, target_load_hp=-1)
            metrics, confusion = [], []
            for condition, seed, observed in (views if job['protocol'] == MIXED else views[:1]):
                probability = engine.R.predict(model, observed, rpm, self.device)
                arrays[f'{condition}_n{seed}'] = probability
                values, counts = scores(labels, probability)
                row = dict(meta, condition=condition, noise_seed=seed)
                metrics.append(dict(row, **values))
                for a in range(4):
                    support = int(counts[a].sum())
                    for b in range(4):
                        confusion.append(dict(row, true_label=a, predicted_label=b, count=int(counts[a, b]),
                                              support=support, row_percent=float(100 * counts[a, b] / support)))
            del model
            path = predictions / f'{job["job_id"]}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            evaluation = dict(meta, prediction_path=str(path), test_windows=len(rows), latency_ms=job['latency_ms'])
            for filename, records in [('metrics.csv', metrics), ('confusion_matrices.csv', confusion),
                                      ('evaluations.csv', [evaluation])]:
                output = destination / filename
                pd.DataFrame(records).to_csv(output, index=False, mode='a', header=not output.exists())
            metric_rows += len(metrics)
            self.state.update(evaluated_pairs=number, metric_rows=metric_rows)
            self.persist()
            print('TEST COMPLETE', job['job_id'], flush=True)
        if metric_rows != METRIC_COUNT:
            raise ValueError('测试条件总数与计划不符')
        self.state.update(stage='reporting')
        self.persist()
        v5_window_report.finish(self.root)
        self.state.update(stage='complete', completed_at=now())
        self.persist()


def check(references):
    root = (RESULTS / f'_window_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve() or not root.name.startswith('_window_check_'):
        raise ValueError('功能检查目录超出 V5 结果范围')
    root.mkdir(exist_ok=False)
    try:
        device, tables, x = setup(root)
        for job in references:
            verify_model(job, tables[split_name(DOMAIN, 1., 0)])
        normalize = engine.normalize_spec

        def one_epoch(spec=None):
            return dict(normalize(dict(spec or {}, max_epochs=200)), max_epochs=1)

        with patch.object(engine, 'normalize_spec', one_epoch):
            for protocol in (CLEAN, MIXED):
                table = tables[split_name(DOMAIN, .1, SUBSETS[0])]
                observed, source = source_data(x, table)
                folder = root / protocol
                info = train(observed, source, protocol, VARIANT, SEEDS[0], SUBSETS[0], folder, device, spec_for(protocol))
                job = dict(info, job_id=protocol, folder=str(folder), ratio=.1, aux_weight=WEIGHT)
                verify_model(job, table)
                if info['epochs_trained'] != 1 or info['validation_calls'] != (6 if protocol == MIXED else 1):
                    raise ValueError(f'{protocol} 单轮检查的验证条件不符')
    finally:
        shutil.rmtree(root)
    print('CHECK PASSED: 10 fixed splits, 5 selection checkpoints, real data clean and AWGN training/reload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true', help='Run into a new output while retaining archived results')
    args = parser.parse_args()
    if args.command == 'plan':
        print('95 new and 5 selection weights, 100 first tests, 1350 new metrics; V3 matched comparisons.')
        return
    references = reference_jobs()
    if args.command == 'check':
        check(references)
        return
    if args.output is None:
        parser.error('run 须指定全新的 --output')
    lock = RESULTS / '.window_random.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        if any(RESULTS.glob(f'{PREFIX}*')) and not args.replay:
            raise FileExistsError('已有窗口实验结果，禁止重复启动')
        experiment = Window(args.output, references)
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
