"""核验 80 模型停止边界，复用全样本和 10% 权重测试并报告。"""

import argparse
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import pandas as pd

from . import run_load_awgn as run
from .run_load import ALLOCATIONS

from experiment_paths import resolve_result_path


SOURCE = run.RESULTS / 'load_awgn_20261109_210313'
MARKER = SOURCE / 'models/m0081'
BUDGETS = (1., .1)
CASES = tuple(case for case in run.NOISE_CASES if case['ratio'] in BUDGETS)
SPLITS = tuple(group for group in ALLOCATIONS if group[1] in BUDGETS)


def stopped_jobs():
    state = json.loads((SOURCE / 'config.json').read_text(encoding='utf-8'))
    request = json.loads((SOURCE / 'partial_request.json').read_text(encoding='utf-8'))
    active = state.get('active_model') or {}
    if (state['stage'] != 'failed' or state['error_type'] != 'FileExistsError'
            or Path(state['error']).resolve() != MARKER.resolve() or not MARKER.is_file()
            or state['pid'] != request['original_training_pid']
            or request['stop_before'] != 'm0081' or request['expected_completed_models'] != len(CASES)
            or json.loads(MARKER.read_text(encoding='utf-8')) != request
            or state['trained_models'] != len(CASES) or state['weights_verified'] is not False
            or state['target_test_accessed'] is not False or state['decisions_locked'] is not False
            or active.get('job_id') != 'm0081' or active.get('ratio') != .2
            or state['selected_aux_weight'] != run.WEIGHT or state['model_config'] != run.MODEL_CONFIG
            or (SOURCE / 'evaluation').exists() or (run.RESULTS / '.load_awgn.lock').exists()):
        raise ValueError('现场尚未到达经核验的 80 模型停止边界')
    if {path.name for path in (SOURCE / 'models').iterdir()} != {f'm{i:04d}' for i in range(1, 82)}:
        raise ValueError('停止边界存在额外或缺少的模型目录')
    for number, (job, case) in enumerate(zip(state['jobs'], CASES, strict=True), start=1):
        job_id = f'm{number:04d}'
        folder = SOURCE / 'models' / job_id
        if (job['job_id'] != job_id or not job['completed'] or not job['from_scratch']
                or job['aux_weight'] != run.WEIGHT or any(job[key] != value for key, value in case.items())
                or resolve_result_path(job['folder']).resolve() != folder.resolve() or not (folder / 'model.pt').is_file()):
            raise ValueError(f'80 模型前缀与阶段计划不一致 {job_id}')
    return state, request


class Partial(run.NoiseTransfer):
    cases = CASES
    budgets = BUDGETS

    def __init__(self, root, decision):
        super().__init__(root, decision)
        (self.root / 'PROTOCOL.md').write_text(
            '# V5 AWGN 全样本与 10% 阶段评价\n\n'
            '按用户要求，原训练在第 80 个模型完成后、20% 首个模型构造前停止。'
            '本目录只复用原实验已完成的 20 个全样本和 60 个 10% 混合训练模型，'
            '以及同范围的 80 个纯干净权重。全程没有新增训练。\n\n'
            '全部权重核验后，对两档共 480 个模型目标组合执行一次评价。'
            '混合组每组合评价干净及 25 个噪声条件，纯干净组每组合补测 25 个噪声条件。'
            '新增 12240 行指标和 195840 个混淆单元，干净参照和 V3 预测只读复用。'
            '训练设置、噪声种子和统计规则见原训练目录的 PROTOCOL.md。20% 和 50% 留待后续。\n',
            encoding='utf-8')

    def prepare(self):
        # 原训练进程在首个 20% 模型构造前退出。这里只使用已保存权重。
        from . import v5_noise_report

        original, request = stopped_jobs()
        self.state.update(scope='full_and_10_percent_only', source_run=str(SOURCE),
                          source_stop_request=request, new_training_models=0,
                          frozen_reference_preflight=v5_noise_report.preflight())
        self.device, self.tables, self.x = run.setup(self.root, SPLITS)
        groups = tuple(group for group in run.v5_report.ALLOCATIONS if group[0] in BUDGETS)
        run.v5_report._same_splits(self.root, SOURCE, groups)
        run.v5_report._same_splits(self.root, run.CLEAN_REFERENCE, groups)
        self.references = [entry for entry in run.clean_references() if entry['ratio'] in BUDGETS]
        run.verify_clean_weights(self.references)
        pd.DataFrame(self.references).to_csv(self.root / 'referenced_models.csv', index=False, mode='x')
        self.jobs.extend(original['jobs'])
        self.state.update(stage='verifying_weights', hardware=torch.cuda.get_device_name(self.device),
                          split_count=len(self.tables), referenced_models=len(self.references), clean_weights_verified=True)
        self.persist()

    def run(self):
        self.prepare()
        self.verify_weights()
        self.state.update(stage='decisions_locked', decisions_locked=True, weights_verified=True,
                          decisions_locked_at=run.now())
        self.persist()
        self.evaluate()
        run.write_json(SOURCE / 'partial_evaluation_complete.json', dict(
            stage='paused_after_10_percent', scope='full_and_10_percent', report=str(self.root / 'REPORT.md'),
            result_dir=str(self.root), completed_at=run.now(), evaluated_pairs=self.state['evaluated_pairs'],
            metric_rows=self.state['metric_rows'], control_file=str(MARKER), remaining_budgets=[.2, .5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('80 saved AWGN weights and 80 frozen clean weights, 480 target pairs, 12240 new metrics.')
        return
    if args.output is None or not args.output.name.startswith('load_awgn_full_10_'):
        parser.error('run 须指定新的 load_awgn_full_10_* 输出目录')
    original, _ = stopped_jobs()
    if (SOURCE / 'partial_evaluation_complete.json').exists():
        raise FileExistsError('该阶段已完成正式评价')
    lock = run.RESULTS / '.load_awgn_partial.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(args.output.resolve()) + '\n')
    try:
        exp = Partial(args.output, dict(original['lambda_decision'], approved_scope='39_dim_awgn_full_and_10_percent'))
        with (exp.root / 'training_console.log').open('x', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    exp.run()
                except BaseException as error:
                    exp.state.update(stage='failed', error=str(error), error_type=type(error).__name__)
                    traceback.print_exc()
                    exp.persist()
                    raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
