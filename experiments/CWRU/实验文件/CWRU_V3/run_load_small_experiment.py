"""完成V3剩余跨负载小样本。默认仅计划，720模型全部选模后统一测试。"""
import sys
sys.dont_write_bytecode = True
import argparse
import os
import shutil
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Windows环境先加载Torch，与现有V3入口保持相同DLL初始化顺序。
import torch
import numpy as np
import pandas as pd

from context import R, HERE, PROJECT, BASE
from splits import read_fixed
from run_experiment import (Experiment, split_name, nested_subset, now, SEEDS,
                            SUBSETS, RATIOS, PROTOCOLS, VARIANTS)
from run_window_experiment import source_state, SOURCE
from metrics import scores
import engine

from experiment_paths import resolve_result_path

CASE_KEYS = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed']
TRAIN_COUNTS = {0: [9, 16, 37], 1: [17, 33, 78], 2: [17, 34, 80], 3: [17, 34, 80]}
VALIDATION_COUNTS = [19, 39, 40, 40]
TARGET_COUNTS = [92, 195, 200, 200]


def small_cases():
    for source in range(4):
        for ratio in RATIOS:
            for subset in SUBSETS:
                for protocol in PROTOCOLS:
                    for variant in VARIANTS:
                        for seed in SEEDS:
                            yield dict(domain=f'load_source_{source}', ratio=ratio, subset=subset,
                                       protocol=protocol, variant=variant, seed=seed)


def prepare_small_splits(root):
    """逐行复用原V3已保存划分，并以原固定来源划分核对嵌套与配额。"""
    tables = {}
    for source in range(4):
        domain = f'load_source_{source}'
        full = read_fixed(SOURCE/'splits'/split_name(domain, 1., 0))
        pd.testing.assert_frame_equal(full, read_fixed(BASE/'splits'/domain))
        for index, ratio in enumerate(RATIOS):
            for subset in SUBSETS:
                name = split_name(domain, ratio, subset)
                w = read_fixed(SOURCE/'splits'/name)
                pd.testing.assert_frame_equal(w, nested_subset(full, ratio, subset))
                if len(w) != 687 or w.record_id.nunique() != 40:
                    raise ValueError('必须保留全部40记录687完整窗')
                if int(w.role.eq('train').sum()) != TRAIN_COUNTS[source][index]:
                    raise ValueError('小样本训练配额不符')
                if int(w.role.eq('validation').sum()) != VALIDATION_COUNTS[source]:
                    raise ValueError('来源验证配额不符')
                if set(w.loc[w.role.isin(['train', 'validation']), 'load_hp']) != {source}:
                    raise ValueError('目标负载不能进入来源训练或验证')
                for target in range(4):
                    if int(w.load_hp.eq(target).sum()) != TARGET_COUNTS[target]:
                        raise ValueError('目标完整窗口数不符')
                shutil.copytree(SOURCE/'splits'/name, root/'splits'/name)
                tables[name] = w
    for filename in ['dataset_support.csv', 'dataset_file_usage.csv']:
        frame = pd.read_csv(SOURCE/filename)
        frame = frame[frame.domain.str.startswith('load_source_') &
                      frame.ratio.isin(RATIOS) & frame.subset.isin(SUBSETS)].copy()
        if filename == 'dataset_file_usage.csv':
            frame = frame[['domain', 'ratio', 'subset', 'role', 'record_id',
                           'used_windows', 'complete_record_windows']]
        if frame.empty:
            raise ValueError('原划分用量表缺失')
        frame.to_csv(root/filename, index=False, mode='x')
    support = pd.read_csv(root/'dataset_support.csv')
    if len(support) != 180:
        raise ValueError('36个划分各须有训练、验证和三个目标支持数')
    for _, row in support.iterrows():
        w = tables[split_name(row.domain, row.ratio, row.subset)]
        mask = w.load_hp.eq(int(row.role[-1])) if row.role.startswith('test_hp') else w.role.eq(row.role)
        counts = np.bincount(w.loc[mask, 'label'], minlength=4)
        np.testing.assert_array_equal(counts, row[['N', 'IR', 'OR', 'B']].to_numpy(dtype=int))
        if int(mask.sum()) != int(row.windows) or not (counts > 0).all():
            raise ValueError('支持数表与固定窗口不一致')
    return tables


class LoadSmallExperiment(Experiment):
    def __init__(self, root):
        self.root = Path(root).resolve()
        if self.root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not self.root.name.startswith('load_small_'):
            raise ValueError('须使用experiments/CWRU/实验结果/CWRU_V3下全新load_small_*直接子目录')
        if self.root.exists():
            raise FileExistsError('禁止覆盖、恢复或自动重启已有实验')
        _, weight = source_state(SOURCE)
        if weight != .3:
            raise ValueError('本轮固定原验证选中的lambda=0.3')
        self.jobs = []
        self.state = dict(version='CWRU_V3_load_small', stage='preparing', started_at=now(), pid=os.getpid(),
            trained_models=0, planned_models=720, planned_new_models=720, planned_test_pairs=2160,
            planned_metric_rows=29160, planned_confusion_cells=466560,
            target_test_accessed=False, historical_test_exposure=True, selection_test_already_accessed=True,
            decisions_locked=False, decision_scope='load_transfer_small_only', automatic_resume=False,
            selected_aux_weight=weight, lambda_locked=True,
            lambda_selection_source=str(SOURCE/'selection_evaluation/selection.json'),
            fixed_split_source=str(SOURCE/'splits'), seeds=SEEDS, ratios=RATIOS, subsets=SUBSETS,
            training_spec=engine.normalize_spec(dict(aux_weight=weight)), jobs=self.jobs)
        self.root.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(HERE/'LOAD_SMALL_PROTOCOL.md', self.root/'PROTOCOL.md')
        self.persist()

    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE = self.device
        R.U.configure()
        self.tables = prepare_small_splits(self.root)
        records, original, raw = R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        if len(original) != 687 or original.record_id.nunique() != 40:
            raise ValueError('原始数据记录或窗口数不符')
        original['load_hp'] = original.record_id.str.rsplit('_', n=1).str[-1].astype(int)
        for w in self.tables.values():
            for key in ['window_index', 'record_id', 'label', 'load_hp', 'rpm', 'start_sample', 'end_sample']:
                np.testing.assert_array_equal(w[key], original[key])
        records.to_csv(self.root/'records.csv', index=False, mode='x')
        usage = pd.read_csv(self.root/'dataset_file_usage.csv')
        metadata = records[['record_id', 'path', 'variable', 'samples', 'label']].rename(
            columns={'path': 'raw_file', 'variable': 'DE_variable', 'samples': 'raw_samples'})
        usage = usage.merge(metadata, on='record_id', how='left', validate='many_to_one')
        usage['unused_tail_samples'] = usage.raw_samples % 24000
        usage.to_csv(self.root/'dataset_file_usage.csv', index=False)
        usage[usage.role.eq('train')].to_csv(self.root/'training_subset_file_usage.csv', index=False, mode='x')
        (self.root/'DATA_USAGE.md').write_text(
            '# V3跨负载小样本数据使用\n\n全部40条原始记录687完整窗，只读使用data。'
            '逐行复用原V3保存的36个来源小样本划分，不重抽、不排除记录。'
            '10%/20%/50%训练窗分别为来源0的9/16/37、来源1的17/33/78、'
            '来源2和3的17/34/80。未选训练窗不转入验证。'
            '每个预算仍另用来源验证19/39/40/40窗，目标0/1/2/3分别为92/195/200/200窗。\n\n'
            '来源0的10%预算正常类仅1个训练窗。验证标签及全样本开发信息不计入训练窗数，'
            '不能将此实验称为同等总标注预算的小样本优势。'
            '来源训练验证共享记录，目标负载参与过历史全负载选参。'
            '方向、子集、噪声实例和训练种子不是独立数据集重复。\n', encoding='utf-8')
        self.x = R.U.normalize(raw)
        self.state.update(hardware=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
                          precision='CUDA encoder BF16; remaining FP32' if self.device.type == 'cuda' else 'FP32')
        self.persist()

    def verify_cohort(self):
        expected = {tuple(c[k] for k in CASE_KEYS) for c in small_cases()}
        actual = {tuple(j[k] for k in CASE_KEYS) for j in self.jobs}
        if len(self.jobs) != 720 or actual != expected or not all(j.get('completed') for j in self.jobs):
            raise ValueError('必须完整完成720个预定小样本模型')
        for j in self.jobs:
            expected_weight = .3 if j['variant'] == 'data_aux_ce' else 0.
            if j['aux_weight'] != expected_weight:
                raise ValueError('已完成模型的系数不符合锁定设置')
            if resolve_result_path(j['folder']).resolve().parent != self.root/'models':
                raise ValueError('本轮只能评价新输出内的模型')

    def run(self):
        self.setup()
        for case in small_cases():
            self.ensure(case, self.state['selected_aux_weight'], 'load_transfer_small')
        self.verify_cohort()
        self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def evaluate(self):
        if not self.state['decisions_locked'] or self.state['target_test_accessed']:
            raise ValueError('测试须全部训练选模锁定，且只执行一次')
        self.verify_cohort()
        dest = self.root/'evaluation'
        dest.mkdir(exist_ok=False)
        (dest/'predictions').mkdir()
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        total_pairs = total_metrics = 0
        for j in self.jobs:
            metric_rows, confusion, evaluations = [], [], []
            w = self.tables[split_name(j['domain'], j['ratio'], j['subset'])]
            model = engine.load(j['folder'], self.device)
            for target in [t for t in range(4) if t != j['source_load_hp']]:
                ix = np.flatnonzero(w.load_hp.eq(target))
                labels = w.iloc[ix].label.to_numpy()
                arrays = dict(labels=labels, window_indices=w.iloc[ix].window_index.to_numpy())
                meta = {k: j[k] for k in ['job_id', *CASE_KEYS, 'aux_weight', 'source_load_hp']}
                meta['target_load_hp'] = target
                for condition, noise_seed, observed in R.conditions(self.x[ix], j['protocol'], True):
                    predictions = R.predict(model, observed, w.iloc[ix].rpm.to_numpy(np.float32), self.device)
                    arrays[f'{condition}_n{noise_seed}'] = predictions
                    values, counts = scores(labels, predictions)
                    m = dict(meta, condition=condition, noise_seed=noise_seed)
                    metric_rows.append(dict(m, **values))
                    for a in range(4):
                        for b in range(4):
                            confusion.append(dict(m, true_label=a, predicted_label=b, count=int(counts[a, b]),
                                support=int(counts[a].sum()), row_percent=float(100*counts[a, b]/counts[a].sum())))
                path = dest/'predictions'/f'{j["job_id"]}_target{target}.npz'
                with path.open('xb') as handle:
                    np.savez_compressed(handle, **arrays)
                evaluations.append(dict(meta, prediction_path=str(path), test_windows=len(ix)))
            del model
            for filename, rows in [('metrics.csv', metric_rows), ('confusion_matrices.csv', confusion),
                                   ('evaluations.csv', evaluations)]:
                path = dest/filename
                pd.DataFrame(rows).to_csv(path, index=False, mode='a', header=not path.exists())
            total_pairs += len(evaluations)
            total_metrics += len(metric_rows)
            self.state.update(evaluated_pairs=total_pairs, metric_rows=total_metrics)
            self.persist()
            print('TEST COMPLETE', j['job_id'], flush=True)
        if (total_pairs, total_metrics) != (2160, 29160):
            raise ValueError('正式测试输出数量不符')
        self.state.update(stage='reporting', testing_completed_at=now())
        self.persist()
        import load_small_report
        load_small_report.finish(self.root)
        self.state.update(stage='complete', completed_at=now())
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('720 new small-source models; fixed lambda 0.3; 10/20/50% x 3 subsets; '
              '12 directions; 2160 test pairs; 29160 metric rows. Train all before testing; stop after report.')
        return
    if args.output is None:
        parser.error('run需要--output全新目录')
    experiment = LoadSmallExperiment(args.output)
    with (experiment.root/'training_console.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                experiment.run()
            except BaseException as error:
                experiment.state.update(stage='failed', error=str(error), error_type=type(error).__name__)
                experiment.persist()
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
