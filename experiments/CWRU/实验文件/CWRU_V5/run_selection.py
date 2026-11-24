"""只选择 V5 的 λ。正式跨负载实验需要另行审批。"""

import argparse
import math
import os
import shutil
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch  # Windows 下先加载 PyTorch，再加载 NumPy 和 SciPy。
import numpy as np
import pandas as pd

from .v5_model import MODEL_CONFIG, engine, load, train
from experiments.CWRU.实验文件.CWRU_V4.run_v4 import fixed_splits, read_dataset
from run_experiment import GRID, SEEDS, Experiment, now, select_lambda, split_name, write_json


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[3] / 'experiments/CWRU/实验结果/CWRU_V5'
V3_SPLITS = RESULTS.parent / 'CWRU_V3/run_20261022_2105/splits'
DOMAIN = 'window_random'
SPLIT = split_name(DOMAIN, 1., 0)
VARIANT = 'data_aux_ce'
PROTOCOL = 'clean_only'
CASES = tuple((weight, seed) for weight in GRID for seed in SEEDS)


def setup(root):
    if not torch.cuda.is_available():
        raise RuntimeError('请用 pytorch_GPU 环境运行 V5 选参')
    engine.R.U.configure()
    records, original, x = read_dataset()
    table = fixed_splits(original, root, [(DOMAIN, 1., 0)], V3_SPLITS)[SPLIT]
    records.to_csv(root / 'records.csv', index=False)
    source = table.role.isin(['train', 'validation']).to_numpy()
    return torch.device('cuda'), x[source], table.loc[source].reset_index(drop=True)


def verify_validation(folder, device, x, table, spec):
    model = load(folder, device)
    rows, arrays = engine.validation(model, table, x, PROTOCOL, device, spec)
    with np.load(folder / 'validation_predictions.npz') as saved:
        for field in ('labels', 'window_indices'):
            np.testing.assert_array_equal(arrays[field], saved[field])
        np.testing.assert_allclose(arrays['clean_n0'], saved['clean_n0'], rtol=1e-6, atol=1e-7)
    return engine.validation_score(rows, PROTOCOL)


class Selection(Experiment):
    def __init__(self, root):
        self.root = root.resolve()
        if self.root.parent != RESULTS.resolve() or not self.root.name.startswith('selection_'):
            raise ValueError('须指定 experiments/CWRU/实验结果/CWRU_V5/selection_* 下的全新目录')
        self.root.mkdir(exist_ok=False)
        self.jobs = []
        self.state = dict(version='CWRU_V5_39_selection', stage='preparing', started_at=now(),
                          pid=os.getpid(), planned_models=len(CASES), trained_models=0,
                          lambda_grid=GRID, seeds=SEEDS, protocol=PROTOCOL, model_config=MODEL_CONFIG,
                          fixed_split_source=str(V3_SPLITS / SPLIT), jobs=self.jobs,
                          selected_aux_weight=None, lambda_locked=False, target_test_accessed=False,
                          selection_test_accessed=False, historical_test_exposure=True,
                          transfer_models=0, transfer_requires_explicit_approval=True)
        shutil.copyfile(HERE / 'PHYSICS_39.md', self.root / 'PHYSICS_39.md')
        self.persist()

    def run(self):
        device, x, table = setup(self.root)
        self.state['hardware'] = torch.cuda.get_device_name(device)
        for number, (weight, seed) in enumerate(CASES, start=1):
            job_id = f'm{number:04d}'
            folder = self.root / 'models' / job_id
            spec = engine.normalize_spec(dict(aux_weight=weight, physics_fit='clean'))
            self.state.update(stage='lambda_selection', active_model=dict(job_id=job_id, seed=seed, aux_weight=weight))
            self.persist()
            info = train(x, table, PROTOCOL, VARIANT, seed, 0, folder, device, spec)
            if not info['completed'] or info['training_windows'] != 481 or info['validation_windows'] != 103:
                raise ValueError(f'{job_id} 未按固定划分完整训练和验证')
            ce = verify_validation(folder, device, x, table, spec)
            if not math.isclose(ce, info['validation_nll'], abs_tol=2e-6, rel_tol=0):
                raise ValueError(f'{job_id} 的保存权重未复现所记载验证 CE')
            self.jobs.append(dict(info, job_id=job_id, folder=str(folder), domain=DOMAIN, ratio=1.,
                                  stage='lambda_selection', aux_weight=weight, reloaded_validation_ce=ce))
            self.state.pop('active_model')
            self.persist()
            print('MODEL COMPLETE', job_id, weight, seed, flush=True)
        self.finish()

    def finish(self):
        weight, sensitivity = select_lambda(self.jobs)
        out = self.root / 'selection'
        out.mkdir(exist_ok=False)
        table = pd.DataFrame(sensitivity)
        table.to_csv(out / 'lambda_sensitivity.csv', index=False)
        pd.DataFrame([{key: job[key] for key in ('job_id', 'aux_weight', 'seed', 'best_epoch',
                                               'validation_nll', 'reloaded_validation_ce')}
                      for job in self.jobs]).to_csv(out / 'per_seed_validation.csv', index=False)
        write_json(out / 'selection.json', dict(selected_aux_weight=weight, candidates=sensitivity,
                   criterion='mean_of_five_best_validation_window_mean_ce; exact_tie_lower_lambda',
                   trained_models=len(self.jobs), weights_verified=True, test_accessed=False,
                   transfer_requires_explicit_approval=True))
        (out / 'REPORT.md').write_text(
            '# V5 39 维纯干净选参\n\n'
            f'25 个模型已完成训练和固定验证。锁定 λ = {weight}。'
            '按五种子各自最佳验证窗口平均交叉熵的均值取最小，完全相同时取较小 λ。\n\n'
            + table.to_markdown(index=False, floatfmt='.8g') + '\n\n'
            '训练 481 窗，验证 103 窗。25 份最优权重均重新加载并核对验证预测及交叉熵。'
            '本阶段只计算验证指标，选参测试和跨负载测试均未执行。\n\n'
            '随机窗口的训练与验证共享原始记录，四个负载均参加验证。'
            '此处选择的统一 λ 用于后续探索性迁移评价。正式跨负载训练须经用户明确审批。\n', encoding='utf-8')
        self.state.update(stage='awaiting_transfer_approval', selected_aux_weight=weight,
                          lambda_locked=True, weights_verified=True, completed_at=now())
        self.persist()
        print('LAMBDA LOCKED', weight, 'AWAITING TRANSFER APPROVAL', flush=True)


def check():
    root = (RESULTS / f'_selection_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve() or not root.name.startswith('_selection_check_'):
        raise ValueError('功能检查目录超出 V5 结果范围')
    root.mkdir(exist_ok=False)
    try:
        device, x, table = setup(root)
        normalize_spec = engine.normalize_spec

        def one_epoch(spec=None):
            settings = normalize_spec(dict(spec or {}, max_epochs=200))
            return dict(settings, max_epochs=1)

        # 用真实的来源窗口验证完整单轮训练、选模及重载接口。
        use = table.role.eq('validation') | table.index.isin(
            table[table.role.eq('train')].groupby('label').head(2).index)
        source = table.loc[use].reset_index(drop=True)
        spec = dict(aux_weight=.3)
        with patch.object(engine, 'normalize_spec', one_epoch):
            info = train(x[use], source, PROTOCOL, VARIANT, SEEDS[0], 0, root / 'model', device, spec)
            if not info['completed'] or info['epochs_trained'] != 1:
                raise ValueError('单轮真实数据检查失败')
            ce = verify_validation(root / 'model', device, x[use], source, spec)
            if not math.isclose(ce, info['validation_nll'], abs_tol=2e-6, rel_tol=0):
                raise ValueError('单轮检查重载验证 CE 不一致')
    finally:
        shutil.rmtree(root)
    print('CHECK PASSED: fixed 481/103/103 split, real-data training, reload and validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('25 new V5 models, clean random-window validation only; lock lambda and stop.')
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.command == 'check':
        check()
        return
    if args.output is None:
        parser.error('run 须指定新的 --output')
    lock = RESULTS / '.selection.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        experiment = Selection(args.output)
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
