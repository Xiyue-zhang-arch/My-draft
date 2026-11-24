"""续做 V5 的 20% 和 50% AWGN 跨负载实验。"""

import argparse
import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from . import run_load_awgn as run
from .run_awgn_partial import SOURCE, stopped_jobs
from . import v5_noise_report

from experiment_paths import resolve_result_path


EARLIER = run.RESULTS / 'load_awgn_full_10_20261109_231117'
BUDGETS = (.2, .5)
FIRST_JOB_NUMBER = 81
CASES = tuple(case for case in run.NOISE_CASES if case['ratio'] in BUDGETS)
SPLITS = tuple(group for group in run.ALLOCATIONS if group[1] in BUDGETS)
PREFIX = 'load_awgn_20_50_'


def verified_decision():
    original, request = stopped_jobs()
    pause = json.loads((SOURCE / 'partial_evaluation_complete.json').read_text(encoding='utf-8'))
    earlier = json.loads((EARLIER / 'config.json').read_text(encoding='utf-8'))
    if (pause['stage'] != 'paused_after_10_percent' or tuple(pause['remaining_budgets']) != BUDGETS
            or Path(pause['result_dir']).resolve() != EARLIER.resolve()
            or Path(pause['report']).resolve() != (EARLIER / 'REPORT.md').resolve()
            or earlier['stage'] != 'complete' or earlier['ratios'] != [1., .1]
            or earlier['trained_models'] != 80 or earlier['referenced_models'] != 80
            or earlier['evaluated_pairs'] != 480 or earlier['metric_rows'] != 12240
            or (pause['evaluated_pairs'], pause['metric_rows']) != (480, 12240)
            or not all(earlier[key] is True for key in ('decisions_locked', 'weights_verified',
                                                        'clean_weights_verified', 'target_test_accessed'))
            or earlier['model_config'] != run.MODEL_CONFIG or earlier['selected_aux_weight'] != run.WEIGHT
            or earlier['training_spec'] != original['training_spec'] or earlier['jobs'] != original['jobs']
            or earlier['source_stop_request'] != request
            or resolve_result_path(earlier['source_run']).resolve() != SOURCE.resolve()
            or not (EARLIER / 'REPORT.md').is_file()
            or not (EARLIER / 'evaluation/comparison_metrics.csv').is_file()):
        raise ValueError('原暂停现场与已完成的全样本和 10% 报告不一致')
    return dict(original['lambda_decision'], approved_scope='39_dim_awgn_20_and_50_percent',
                approval='user_approved_remaining_20_and_50_percent',
                earlier_report=str(EARLIER / 'REPORT.md'), earlier_target_results_exposed=True)


class Remaining(run.NoiseTransfer):
    cases = CASES
    budgets = BUDGETS
    allocations = SPLITS
    first_job_number = FIRST_JOB_NUMBER

    def __init__(self, root, decision):
        super().__init__(root, decision)
        (self.root / 'PROTOCOL.md').write_text(
            '# V5 AWGN 20% 和 50% 阶段\n\n'
            '按用户授权，以 λ=0.3 从头训练 20% 和 50% 各 60 个 AWGN 混合训练模型。'
            '全样本和 10% 已完成训练与测试，只读引用其结果。旧停止现场保留。'
            '两档共使用 24 份固定划分，训练和选模沿用同一 AWGN 协议。\n\n'
            '核验 120 个新模型和同范围的 120 个纯干净权重后，统一评价 720 个目标组合。'
            '混合组每组合测干净及 25 个固定噪声条件，纯干净组每组合补测相同的 25 个噪声条件。'
            '新增 18360 行指标、293760 个混淆计数单元。干净目标沿用冻结预测。'
            '阶段报告完成后，按预算合并两阶段已核验的预测统计。\n\n'
            '20% 和 50% 的训练规则在这两档正式目标测试之前固定。全样本和 10% 目标结果已曝光，'
            '所有档位均沿用固定频带、物理特征、辅助权重、训练及测试种子。'
            '完整来源验证标签仍额外计入。结果限于这批 CWRU 记录及设定的人工白噪声。\n', encoding='utf-8')
        self.state.update(scope='20_and_50_percent_only', earlier_run=str(EARLIER),
                          source_run=str(SOURCE), combined_report_status='pending')
        self.persist()

    def prepare(self):
        super().prepare()
        groups = tuple(group for group in run.v5_report.ALLOCATIONS if group[0] in BUDGETS)
        run.v5_report._same_splits(self.root, SOURCE, groups)

    def run(self):
        super().run()
        v5_noise_report.combine(EARLIER, self.root)
        self.state.update(combined_report_status='complete', completed_at=run.now())
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('120 new mixed models, 120 saved clean models, 720 target pairs, 18360 new metrics.')
        return
    decision = verified_decision()
    if args.command == 'check':
        run.check(SPLITS, tuple((ratio, run.SUBSETS[0]) for ratio in BUDGETS))
        return
    if args.output is None or not args.output.name.startswith(PREFIX):
        parser.error('run 须指定全新的 load_awgn_20_50_* 输出目录')
    lock = run.RESULTS / '.load_awgn.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        if (run.RESULTS / '.load_awgn_partial.lock').exists() or any(run.RESULTS.glob(f'{PREFIX}*')):
            raise FileExistsError('已有 AWGN 阶段在运行或已有 20%/50% 结果，禁止重复启动')
        exp = Remaining(args.output, decision)
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
