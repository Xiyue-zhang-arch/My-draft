"""V5 跨负载 AWGN 实验。训练混合组，复用纯干净权重补测相同噪声。"""

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

from .run_load import (ALLOCATIONS, BUDGETS, CASES, HERE, LABELS, RESULTS, SEEDS, SOURCES, SUBSETS,
                       VARIANT, V3_SPLITS, WEIGHT, Transfer, approved_decision, engine,
                       load, now, scores, setup, source_data, split_name, train, v5_report,
                       write_json)
from .v5_model import MODEL_CONFIG

from experiment_paths import resolve_result_path


CLEAN_REFERENCE = RESULTS / 'load_clean_20261109_142434'
PROTOCOL = 'awgn_mixed'
NOISE_CASES = tuple(dict(case, protocol=PROTOCOL) for case in CASES)
SNRS = (-4, -2, 0, 2, 4)
NOISE_SEEDS = (7201, 7202, 7203, 7204, 7205)
CONDITIONS = tuple((f'white_{snr:+d}', seed) for snr in SNRS for seed in NOISE_SEEDS)
MODEL_TARGET_PAIRS = 6 * len(NOISE_CASES)
METRIC_ROWS = 3 * len(NOISE_CASES) * (1 + 2 * len(CONDITIONS))


def clean_references():
    state = json.loads((CLEAN_REFERENCE / 'config.json').read_text(encoding='utf-8'))
    if (state['stage'] != 'complete' or state['selected_aux_weight'] != WEIGHT
            or state['weights_verified'] is not True or state['decisions_locked'] is not True
            or state['target_test_accessed'] is not True or state['evaluated_pairs'] != 600
            or state['model_config'] != MODEL_CONFIG):
        raise ValueError('冻结的 V5 纯干净结果尚未完整核验')
    jobs, _ = v5_report._read_structure(CLEAN_REFERENCE, v5_report.ALLOCATIONS, WEIGHT)
    entries = []
    for i, job in enumerate(jobs.sort_values('job_id').to_dict('records'), start=1):
        model_id = f'm{i:04d}'
        folder = CLEAN_REFERENCE / 'models' / model_id
        if (job['job_id'] != model_id or resolve_result_path(job['folder']).resolve() != folder.resolve()
                or job['protocol'] != 'clean_only' or not (folder / 'model.pt').is_file()):
            raise ValueError(f'冻结纯干净模型登记与权重不符 {model_id}')
        entries.append(dict(job, job_id=f'c{i:04d}', reference_job_id=model_id))
    return entries


def verify_clean_weights(entries):
    clean_spec = engine.normalize_spec(dict(aux_weight=WEIGHT, physics_fit='clean'))
    for entry in entries:
        model = load(entry['folder'], 'cpu')
        if (model.variant != VARIANT or model.spec != clean_spec
                or model.network.physics[0].in_features != 39
                or not all(torch.isfinite(value).all() for value in model.state_dict().values())
                or not (model.evidence.scale >= .05).all()):
            raise ValueError(f'冻结纯干净模型权重或标准化异常 {entry["reference_job_id"]}')


class NoiseTransfer(Transfer):
    cases = NOISE_CASES
    budgets = BUDGETS
    allocations = ALLOCATIONS
    validation_predictions = ('clean_n0', *(f'white_{snr:+d}_n6201' for snr in SNRS))

    def __init__(self, root, decision):
        self.root = root.resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith('load_awgn_'):
            raise ValueError('须使用 experiments/CWRU/实验结果/CWRU_V5 下全新的 load_awgn_* 目录')
        self.root.mkdir(exist_ok=False)
        self.jobs = []
        self.spec = engine.normalize_spec(dict(aux_weight=WEIGHT, physics_fit='matched_noise'))
        planned_metrics = 3 * len(self.cases) * (1 + 2 * len(CONDITIONS))
        self.state = dict(version='CWRU_V5_39_load_awgn', stage='preparing', started_at=now(), pid=os.getpid(),
                          planned_models=len(self.cases), planned_referenced_models=len(self.cases),
                          planned_test_pairs=6 * len(self.cases), planned_metric_rows=planned_metrics,
                          planned_confusion_cells=16 * planned_metrics, evaluated_pairs=0, metric_rows=0,
                          jobs=self.jobs, selected_aux_weight=WEIGHT, lambda_locked=True,
                          decisions_locked=False, weights_verified=False, target_test_accessed=False,
                          historical_test_exposure=True, selection_test_accessed=False,
                          protocol=PROTOCOL, ratios=self.budgets, seeds=SEEDS, subsets=SUBSETS,
                          model_config=MODEL_CONFIG, training_spec=self.spec, fixed_split_source=str(V3_SPLITS),
                          clean_reference=str(CLEAN_REFERENCE), test_snr_db=SNRS, test_noise_seeds=NOISE_SEEDS,
                          automatic_resume=False, lambda_decision=decision)
        shutil.copyfile(HERE / 'PHYSICS_39.md', self.root / 'PHYSICS_39.md')
        shutil.copyfile(HERE / 'LOAD_AWGN_PROTOCOL.md', self.root / 'PROTOCOL.md')
        write_json(self.root / 'lambda_decision.json', decision)
        self.persist()

    def prepare(self):
        from . import v5_noise_report

        self.state['frozen_reference_preflight'] = v5_noise_report.preflight()
        self.references = [entry for entry in clean_references() if entry['ratio'] in self.budgets]
        self.device, self.tables, self.x = setup(self.root, self.allocations)
        groups = tuple(group for group in v5_report.ALLOCATIONS if group[0] in self.budgets)
        v5_report._same_splits(self.root, CLEAN_REFERENCE, groups)
        verify_clean_weights(self.references)
        pd.DataFrame(self.references).to_csv(self.root / 'referenced_models.csv', index=False, mode='x')
        self.state.update(hardware=torch.cuda.get_device_name(self.device), split_count=len(self.tables),
                          referenced_models=len(self.references), clean_weights_verified=True)

    def evaluate(self):
        from . import v5_noise_report

        if (len(self.jobs) != len(self.cases) or len(self.references) != len(self.cases)
                or not self.state['weights_verified'] or not self.state['clean_weights_verified']
                or not self.state['decisions_locked'] or self.state['target_test_accessed']):
            raise ValueError('须核验全部新旧权重并锁定后才可执行一次目标测试')
        destination = self.root / 'evaluation'
        predictions = destination / 'predictions'
        predictions.mkdir(parents=True, exist_ok=False)
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        evaluated = metric_rows = 0
        # 固定划分逐行共享目标窗口。每次仅缓存一个目标负载的26种观测。
        common = next(iter(self.tables.values()))
        for target in SOURCES:
            indices = np.flatnonzero(common.load_hp.eq(target))
            rows = common.iloc[indices]
            labels, rpm, window_ids = rows.label.to_numpy(), rows.rpm.to_numpy(np.float32), rows.window_index.to_numpy()
            views = list(engine.R.conditions(self.x[indices], PROTOCOL, True))
            if tuple((condition, seed) for condition, seed, _ in views) != (('clean', 0), *CONDITIONS):
                raise ValueError('固定 AWGN 测试条件与计划不符')
            for job in [*self.jobs, *self.references]:
                if job['source_load_hp'] == target:
                    continue
                model = load(job['folder'], self.device)
                is_mixed = job['protocol'] == PROTOCOL
                metrics, confusion = [], []
                arrays = dict(labels=labels, window_indices=window_ids)
                meta = {key: job[key] for key in ('job_id', *v5_report.META)}
                meta['target_load_hp'] = target
                for condition, seed, observed in (views if is_mixed else views[1:]):
                    probability = engine.R.predict(model, observed, rpm, self.device)
                    arrays[f'{condition}_n{seed}'] = probability
                    values, counts = scores(labels, probability)
                    info = dict(meta, condition=condition, noise_seed=seed)
                    metrics.append(dict(info, **values))
                    for a in range(len(LABELS)):
                        support = int(counts[a].sum())
                        for b in range(len(LABELS)):
                            confusion.append(dict(info, true_label=a, predicted_label=b,
                                                  count=int(counts[a, b]), support=support,
                                                  row_percent=float(100 * counts[a, b] / support)))
                del model
                path = predictions / f'{job["job_id"]}_target{target}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                evaluation = dict(meta, prediction_path=str(path), test_windows=len(rows))
                for filename, entries in [('evaluations.csv', [evaluation]), ('metrics.csv', metrics),
                                          ('confusion_matrices.csv', confusion)]:
                    output = destination / filename
                    pd.DataFrame(entries).to_csv(output, index=False, mode='a', header=not output.exists())
                evaluated += 1
                metric_rows += len(metrics)
                self.state.update(evaluated_pairs=evaluated, metric_rows=metric_rows, active_target=target)
                if evaluated % 10 == 0:
                    self.state['updated_at'] = now()
                    write_json(self.root / 'config.json', self.state)
            del views
            print('TARGET COMPLETE', target, evaluated, metric_rows, flush=True)
        if (evaluated, metric_rows) != (self.state['planned_test_pairs'], self.state['planned_metric_rows']):
            raise ValueError('实际目标组合或条件数与计划不符')
        self.state.pop('active_target')
        self.state.update(stage='reporting')
        self.persist()
        v5_noise_report.finish(self.root)
        self.state.update(stage='complete', completed_at=now())
        self.persist()


def check(allocations=ALLOCATIONS, samples=((1., 0), (.1, SUBSETS[0]))):
    from . import v5_noise_report

    root = (RESULTS / f'_awgn_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve() or not root.name.startswith('_awgn_check_'):
        raise ValueError('功能检查目录超出 V5 结果范围')
    root.mkdir(exist_ok=False)
    try:
        v5_noise_report.preflight()
        references = clean_references()
        device, tables, x = setup(root, allocations)
        budgets = {ratio for _, ratio, _ in allocations}
        groups = tuple(group for group in v5_report.ALLOCATIONS if group[0] in budgets)
        v5_report._same_splits(root, CLEAN_REFERENCE, groups)
        verify_clean_weights(references[:1])
        normalize = engine.normalize_spec

        def one_epoch(spec=None):
            return dict(normalize(dict(spec or {}, max_epochs=200)), max_epochs=1)

        with patch.object(engine, 'normalize_spec', one_epoch):
            for ratio, subset in samples:
                table = tables[split_name('load_source_0', ratio, subset)]
                observations, source = source_data(x, table)
                folder = root / f'check_{ratio}'
                spec = dict(aux_weight=WEIGHT, physics_fit='matched_noise')
                info = train(observations, source, PROTOCOL, VARIANT, SEEDS[0], subset, folder, device, spec)
                rows, predictions = engine.validation(load(folder, device), source, observations, PROTOCOL, device, spec)
                with np.load(folder / 'validation_predictions.npz') as saved:
                    for key in ('labels', 'window_indices'):
                        np.testing.assert_array_equal(saved[key], predictions[key])
                    for key in NoiseTransfer.validation_predictions:
                        np.testing.assert_allclose(saved[key], predictions[key], rtol=1e-6, atol=1e-7)
                if (not info['completed'] or info['epochs_trained'] != 1 or info['validation_calls'] != 6
                        or info['fit_observations'] != 6 * info['training_windows']
                        or info['fit_clean_observations'] != info['fit_noisy_observations']
                        or not math.isclose(engine.validation_score(rows, PROTOCOL), info['validation_nll'],
                                            rel_tol=0, abs_tol=2e-6)):
                    raise ValueError(f'{ratio:g} AWGN 单轮功能检查失败')
    finally:
        shutil.rmtree(root)
    print(f'CHECK PASSED: {len(allocations)} fixed splits; clean checkpoint; real 0 HP AWGN one-epoch training and reload for {samples}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print(f'{len(NOISE_CASES)} new and {len(CASES)} frozen models; {MODEL_TARGET_PAIRS} target pairs, {METRIC_ROWS} new conditions.')
        return
    if args.command == 'check':
        check()
        return
    if args.output is None:
        parser.error('run 须指定新的 --output')
    decision = dict(approved_decision(), approved_scope='39_dim_awgn_full_and_10_20_50_percent',
                    approval='user_requested_noise_transfer_after_clean_transfer',
                    clean_coefficient_reused_without_new_selection=True)
    lock = RESULTS / '.load_awgn.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        experiment = NoiseTransfer(args.output, decision)
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
