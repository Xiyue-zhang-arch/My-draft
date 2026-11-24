"""用已审批的 V4 系数运行纯干净跨负载全样本。"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from uuid import uuid4

sys.dont_write_bytecode = True

from run_v4 import (BANDS, HERE, RESULTS, SEEDS, STAGES, VARIANTS, V4Experiment,
                    configure, engine, fixed_splits, now, pd, read_dataset, split_name,
                    torch, transfer_cases)
from selection_approval import read_json, verify_boundary

SELECTION = RESULTS / 'clean_20261106_105008'
STAGE = STAGES[1]
ALLOCATIONS = [(f'load_source_{source}', 1., 0) for source in range(4)]
MODEL_COUNT = len(ALLOCATIONS) * len(VARIANTS) * len(SEEDS)
PAIR_COUNT = 3 * MODEL_COUNT


def approved_selection():
    state = read_json(SELECTION / 'config.json')
    approval = read_json(SELECTION / 'selection_approval.json')
    if (state['stage'] != 'awaiting_selection_approval' or not state['pause_boundary_verified']
            or approval['status'] != 'awaiting_approval' or approval['transfer_models'] != 0):
        raise ValueError('未找到已核验的 V4 选参停止边界')
    event = state['approval_stop_event']
    stopped = dict(state, stage=event['stage'], error=event['error'], error_type=event['error_type'],
                   active_model=event['blocked_model'])
    selection = verify_boundary(SELECTION, stopped, SELECTION / 'models' / 'm0031',
                                event['original_training_pid'])
    if (selection['selected_aux_weight'] != 1. or approval['selected_aux_weight'] != 1.
            or selection['candidates'] != approval['candidates']):
        raise ValueError('选参结果与本次获准使用的统一 λ=1.0 不一致')
    return selection['selected_aux_weight']


class ApprovedTransferExperiment(V4Experiment):
    def __init__(self, output, weight):
        self.root = Path(output).resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith(self.prefix):
            raise ValueError(f'须使用 experiments/CWRU/实验结果/CWRU_V4 下全新 {self.prefix}* 目录')
        self.root.mkdir(parents=True, exist_ok=False)
        self.jobs = []
        models = len(self.cases)
        pairs = 3 * models
        self.state = dict(
            version=self.version, stage='preparing', started_at=now(), pid=os.getpid(),
            planned_models=models, planned_stage_models={self.stage: models},
            planned_test_pairs=pairs, planned_metric_rows=pairs, planned_confusion_cells=16*pairs,
            evaluated_pairs=0, metric_rows=0, seeds=SEEDS,
            ratios=sorted({case['ratio'] for case in self.cases}),
            subsets=sorted({case['subset'] for case in self.cases}),
            approval_scope=self.scope, selected_aux_weight=weight, lambda_locked=True,
            selection_test_accessed=False, target_test_accessed=False, decisions_locked=False,
            historical_test_exposure=True, protocol='clean_only', comparison_scope=VARIANTS,
            fixed_split_source=str(SELECTION / 'splits'), lambda_selection_source=str(SELECTION / 'selection.json'),
            bands_hz=BANDS, sampling_hz=48000, features_per_band=53,
            training_spec=engine.normalize_spec(dict(aux_weight=weight)), jobs=self.jobs)
        shutil.copyfile(HERE / self.protocol_file, self.root / 'PROTOCOL.md')
        shutil.copyfile(SELECTION / 'selection.json', self.root / 'selection.json')
        self.persist()

    def run_transfer(self, finish):
        allocations = list(dict.fromkeys((c['domain'], c['ratio'], c['subset']) for c in self.cases))
        self.setup(allocations, SELECTION / 'splits')
        weight = self.state['selected_aux_weight']
        for case in self.cases:
            self.ensure(case, weight, self.stage)
        expected = [(self.stage, case, weight if case['variant'] == 'data_aux_ce' else 0.) for case in self.cases]
        self.verify_models(expected)
        self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now())
        self.persist()
        self.evaluate(finish)


class FullExperiment(ApprovedTransferExperiment):
    prefix = 'load_full_'
    version = 'CWRU_V4_load_full'
    stage = STAGE
    scope = 'load_transfer_full_clean_only'
    protocol_file = 'LOAD_FULL_PROTOCOL.md'
    cases = tuple(transfer_cases(True))


def check_one_epoch(root, device, x, w, weight, subset):
    mask = w.role.isin(['train', 'validation']).to_numpy()
    normalize = engine.normalize_spec

    def one_epoch(spec=None):
        return dict(normalize(dict(spec or {}, max_epochs=200)), max_epochs=1)

    with patch.object(engine, 'normalize_spec', one_epoch):
        for variant in VARIANTS:
            folder = root / variant
            info = engine.train(x[mask], w.loc[mask].reset_index(drop=True), 'clean_only', variant,
                                SEEDS[0], subset, folder, device, dict(aux_weight=weight))
            saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            if (not info['completed'] or info['epochs_trained'] != 1
                    or info['training_windows'] != w.role.eq('train').sum()
                    or info['validation_windows'] != w.role.eq('validation').sum()
                    or saved['model_config']['bands_hz'] != BANDS):
                raise ValueError(f'{variant} 来源 0 功能检查失败')


def check(weight):
    device = configure()
    _, original, x = read_dataset()
    root = (RESULTS / f'_v4_full_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve():
        raise ValueError('功能检查目录超出 V4 范围')
    root.mkdir(exist_ok=False)
    try:
        tables = fixed_splits(original, root, ALLOCATIONS, SELECTION / 'splits')
        usage = pd.read_csv(root / 'dataset_support.csv')
        assert usage.loc[usage.role.eq('train'), 'windows'].tolist() == [73, 156, 160, 160]
        assert usage.loc[usage.role.eq('validation'), 'windows'].tolist() == [19, 39, 40, 40]
        assert len(usage) == 20
        check_one_epoch(root, device, x, tables[split_name(*ALLOCATIONS[0])], weight, 0)
    finally:
        shutil.rmtree(root)
    print('CHECK PASSED: locked V4 selection, four fixed full splits, both real-data variants.')


def run_locked(experiment_type, output, weight, finish, *, replay=False):
    lock = RESULTS / '.clean_run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {output.resolve()}\n')
    try:
        if any(RESULTS.glob(f'{experiment_type.prefix}*')) and not replay:
            raise FileExistsError('已有同范围的 V4 迁移目录，拒绝重复启动')
        experiment = experiment_type(output, weight)
        with (experiment.root / 'training_console.log').open('x', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    experiment.run_transfer(finish)
                except BaseException as error:
                    experiment.state.update(stage='failed', error=str(error), error_type=type(error).__name__)
                    traceback.print_exc()
                    experiment.persist()
                    raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        print(f'{MODEL_COUNT} new clean full models; {PAIR_COUNT} target evaluations after model selection.')
        return
    weight = approved_selection()
    if args.command == 'check':
        check(weight)
        return
    if args.output is None:
        parser.error('run 须指定全新 --output 目录')
    from v4_report import finish_full
    run_locked(FullExperiment, args.output, weight, finish_full, replay=args.replay)


if __name__ == '__main__':
    main()
