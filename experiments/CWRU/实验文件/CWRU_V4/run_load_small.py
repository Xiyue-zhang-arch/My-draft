"""V4 纯干净跨负载小样本，按获准预算独立运行。"""
import argparse
from pathlib import Path
import shutil
import sys
from uuid import uuid4

sys.dont_write_bytecode = True

from run_load_full import (SELECTION, ApprovedTransferExperiment, approved_selection,
                           check_one_epoch, run_locked)
from run_v4 import (RESULTS, STAGES, SUBSETS, configure, fixed_splits, pd, read_dataset,
                    split_name, transfer_cases)

TRAIN_COUNTS = {.1: (9, 17, 17, 17), .2: (16, 33, 34, 34), .5: (37, 78, 80, 80)}
VALIDATION_COUNTS = (19, 39, 40, 40)


class SmallExperiment(ApprovedTransferExperiment):
    prefix = 'load_small_10_20_'
    version = 'CWRU_V4_load_small'
    stage = STAGES[2]
    scope = 'load_transfer_small_10_20_clean_only'
    protocol_file = 'LOAD_SMALL_10_20_PROTOCOL.md'
    budgets = (.1, .2)
    cases = tuple(sorted((c for c in transfer_cases(False) if c['ratio'] in (.1, .2)),
                         key=lambda case: case['ratio']))


class HalfExperiment(SmallExperiment):
    prefix = 'load_small_50_'
    scope = 'load_transfer_small_50_clean_only'
    protocol_file = 'LOAD_SMALL_50_PROTOCOL.md'
    budgets = (.5,)
    cases = tuple(c for c in transfer_cases(False) if c['ratio'] == .5)


def check(weight, budgets):
    device = configure()
    _, original, x = read_dataset()
    allocations = [(f'load_source_{source}', ratio, subset)
                   for ratio in budgets for source in range(4) for subset in SUBSETS]
    root = (RESULTS / f'_v4_small_check_{uuid4().hex}').resolve()
    if root.parent != RESULTS.resolve():
        raise ValueError('功能检查目录超出 V4 范围')
    root.mkdir(exist_ok=False)
    try:
        tables = fixed_splits(original, root, allocations, SELECTION / 'splits')
        usage = pd.read_csv(root / 'dataset_support.csv')
        if len(tables) != len(allocations) or len(usage) != 5 * len(allocations):
            raise ValueError('划分或配额表数量异常')
        for source in range(4):
            domain = f'load_source_{source}'
            for subset in SUBSETS:
                selected = usage[usage.domain.eq(domain) & usage.subset.eq(subset)]
                if (tuple(selected.loc[selected.role.eq('train'), 'windows'])
                        != tuple(TRAIN_COUNTS[ratio][source] for ratio in budgets)
                        or tuple(selected.loc[selected.role.eq('validation'), 'windows'])
                        != (VALIDATION_COUNTS[source],) * len(budgets)):
                    raise ValueError(f'{domain}/{subset} 训练或验证预算异常')
        for ratio in budgets:
            name = split_name('load_source_0', ratio, SUBSETS[0])
            check_one_epoch(root / name, device, x, tables[name], weight, SUBSETS[0])
    finally:
        shutil.rmtree(root)
    print(f'CHECK PASSED: locked V4 selection, {len(allocations)} fixed splits, both real-data variants.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'run'], nargs='?', default='plan')
    parser.add_argument('--budget', choices=['10-20', '50'], default='10-20')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    experiment = HalfExperiment if args.budget == '50' else SmallExperiment
    if args.command == 'plan':
        count = len(experiment.cases)
        print(f'{count} new clean models for {args.budget}%; {3*count} target evaluations after model selection.')
        return
    weight = approved_selection()
    if args.command == 'check':
        check(weight, experiment.budgets)
        return
    if args.output is None:
        parser.error('run 须指定全新 --output 目录')
    from v4_report import finish_small, finish_small_50
    run_locked(experiment, args.output, weight, finish_small_50 if args.budget == '50' else finish_small, replay=args.replay)


if __name__ == '__main__':
    main()
