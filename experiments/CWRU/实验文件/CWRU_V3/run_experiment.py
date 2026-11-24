"""V3独立入口。默认只显示计划，run只接受全新结果目录。"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import redirect_stdout, redirect_stderr
import torch
import numpy as np
import pandas as pd
from context import R, HERE, PROJECT, BASE, HISTORY
from splits import read_fixed
import engine
from metrics import scores, METRICS
import report

SEEDS = list(range(1706, 1711))
SUBSETS = [4101, 4102, 4103]
RATIOS = [.1, .2, .5]
GRID = [.01, .03, .1, .3, 1.]
PROTOCOLS = ['clean_only', 'awgn_mixed']
VARIANTS = ['baseline_data', 'data_aux_ce']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    # 原子替换仅用于本轮活动配置，避免读取半写入状态；
    # 杀毒或编辑器短暂占用目标文件时重试，避免共享冲突打断训练/测试。
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def cases(domain):
    for ratio in [1., *RATIOS]:
        for subset in ([0] if ratio == 1 else SUBSETS):
            for protocol in PROTOCOLS:
                for variant in VARIANTS:
                    for seed in SEEDS:
                        yield dict(domain=domain, ratio=ratio, subset=subset,
                                   protocol=protocol, variant=variant, seed=seed)


def split_name(domain, ratio, subset):
    return f'{domain}_r{int(ratio*100):03d}_s{subset}'


def nested_subset(full, ratio, subset):
    chosen = []
    for label, g in full[full.role.eq('train')].groupby('label'):
        rng = np.random.default_rng(np.random.SeedSequence([subset, int(label), 20261020]))
        records = rng.permutation(g.record_index.unique())
        queues = {rid: list(rng.permutation(g.index[g.record_index.eq(rid)].to_numpy())) for rid in records}
        order = []
        while any(queues.values()):
            for rid in records:
                if queues[rid]:
                    order.append(queues[rid].pop())
        chosen.extend(order[:math.ceil(ratio*len(order))])
    w = full.copy()
    w.loc[w.role.eq('train'), 'role'] = 'unused_train'
    w.loc[sorted(chosen), 'role'] = 'train'
    return w


def prepare_splits(root):
    tables, supports, usage = {}, [], []
    for domain in ['window_random', *[f'load_source_{s}' for s in range(4)]]:
        full = read_fixed(BASE/'splits'/domain)
        for ratio in [1., *RATIOS]:
            for subset in ([0] if ratio == 1 else SUBSETS):
                w = full.copy() if ratio == 1 else nested_subset(full, ratio, subset)
                if domain == 'window_random' and ratio < 1:
                    old = read_fixed(HISTORY/'subsets'/f'ratio_{int(ratio*100):03d}_subset_{subset}')
                    pd.testing.assert_frame_equal(w, old)
                name = split_name(domain, ratio, subset)
                dest = root/'splits'/name
                dest.mkdir(parents=True, exist_ok=False)
                w.to_csv(dest/'windows.csv', index=False)
                roles = {role: w.role.eq(role) for role in ['train', 'validation']}
                if domain == 'window_random':
                    roles['test'] = w.role.eq('test')
                else:
                    source = int(domain[-1])
                    assert set(w.loc[w.role.isin(['train','validation']), 'load_hp']) == {source}
                    roles.update({f'test_hp{t}': w.load_hp.eq(t) for t in range(4) if t != source})
                for role, mask in roles.items():
                    a = w[mask]
                    (dest/f'{role}.txt').write_text(''.join(f'{i}\n' for i in a.window_index), encoding='utf-8')
                    c = np.bincount(a.label, minlength=4)
                    assert (c > 0).all()
                    supports.append(dict(domain=domain, ratio=ratio, subset=subset, role=role,
                        windows=len(a), **dict(zip(['N','IR','OR','B'], map(int,c)))))
                    for rid, g in w.groupby('record_id'):
                        usage.append(dict(domain=domain, ratio=ratio, subset=subset, role=role,
                            record_id=rid, used_windows=int(a.record_id.eq(rid).sum()),
                            complete_record_windows=len(g)))
                tables[name] = w
    pd.DataFrame(supports).to_csv(root/'dataset_support.csv', index=False)
    pd.DataFrame(usage).to_csv(root/'dataset_file_usage.csv', index=False)
    return tables


def select_lambda(jobs):
    selected = [j for j in jobs if j['stage'] == 'lambda_selection' and j['variant'] == 'data_aux_ce']
    rows = []
    for weight in GRID:
        a = [j for j in selected if j['aux_weight'] == weight]
        if len(a) != 5 or sorted(j['seed'] for j in a) != SEEDS:
            raise ValueError('选参必须包含每档完整5种子')
        values = [j['validation_nll'] for j in a]
        if not np.isfinite(values).all():
            raise ValueError('选参CE非有限')
        rows.append(dict(aux_weight=weight, mean_best_validation_ce=float(np.mean(values)),
                         sample_sd=float(np.std(values, ddof=1)), seeds=5))
    # 仅数值完全相同才按较小系数打破平局，不按测试或ACC调整。
    winner = min(rows, key=lambda r: (r['mean_best_validation_ce'], r['aux_weight']))
    return winner['aux_weight'], rows


class Experiment:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if self.root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or self.root.exists():
            raise ValueError('必须使用experiments/CWRU/实验结果/CWRU_V3下全新直接子目录，禁止恢复或覆盖')
        self.root.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(HERE/'PROTOCOL.md', self.root/'PROTOCOL.md')
        self.state = dict(version='CWRU_V3', stage='preparing', started_at=now(), pid=os.getpid(),
            trained_models=0, planned_models=1020, planned_main_models=1000,
            planned_test_pairs=2600, planned_test_metric_rows=35100,
            target_test_accessed=False, historical_test_exposure=True, decisions_locked=False,
            selected_aux_weight=None, seeds=SEEDS, subsets=SUBSETS, lambda_grid=GRID,
            selection_rule='mean_of_five_best_validation_window_mean_ce; exact_tie_lower_lambda',
            training_spec=engine.normalize_spec(), jobs=[])
        self.jobs = self.state['jobs']
        self.persist()

    def persist(self):
        self.state.update(updated_at=now(), trained_models=len(self.jobs))
        write_json(self.root/'config.json', self.state)
        if self.jobs:
            pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv', index=False)

    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE = self.device
        R.U.configure()
        self.tables = prepare_splits(self.root)
        records, original, raw = R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        assert len(original) == 687 and original.record_id.nunique() == 40
        for w in self.tables.values():
            for key in ['window_index','record_id','label','start_sample','end_sample']:
                np.testing.assert_array_equal(w[key], original[key])
        records.to_csv(self.root/'records.csv', index=False)
        usage = pd.read_csv(self.root/'dataset_file_usage.csv')
        metadata = records[['record_id','path','variable','samples','label']].rename(
            columns={'path':'raw_file','variable':'DE_variable','samples':'raw_samples'})
        usage = usage.merge(metadata, on='record_id', how='left', validate='many_to_one')
        usage['unused_tail_samples'] = usage.raw_samples % 24000
        usage.to_csv(self.root/'dataset_file_usage.csv', index=False)
        usage[usage.role.eq('train')].to_csv(self.root/'training_subset_file_usage.csv', index=False)
        (self.root/'DATA_USAGE.md').write_text(
            '# V3数据使用\n\n全部40记录687完整窗，原始data只读。逐MAT路径、通道、原始采样点数、'
            '丢弃尾部和每个划分及预算的用量见dataset_file_usage.csv，逐类支持数见dataset_support.csv。'
            'training_subset_file_usage.csv列出每预算每子集的训练用量，模型选中窗口另存各模型目录。\n\n'
            '随机窗口固定481/103/103；四来源完整训练73/156/160/160、验证19/39/40/40、目标92/195/200/200。'
            '小样本仍使用完整固定验证标签。子集、训练种子、噪声重复及方向均不能作为独立数据集重复。'
            '历史测试曝光、训练测试共享记录及全负载选参对迁移结论的限制见PROTOCOL.md。', encoding='utf-8')
        self.x = R.U.normalize(raw)
        self.state.update(hardware=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',
                          precision='CUDA encoder BF16; remaining FP32' if self.device.type=='cuda' else 'FP32')
        self.persist()

    def ensure(self, case, weight, stage):
        actual_weight = weight if case['variant'] == 'data_aux_ce' else 0.
        old = [j for j in self.jobs if all(j[k] == v for k,v in case.items()) and j['aux_weight'] == actual_weight]
        if old:
            assert len(old) == 1
            return old[0]
        jid = f'm{len(self.jobs)+1:04d}'
        folder = self.root/'models'/jid
        spec = engine.normalize_spec(dict(aux_weight=weight,
            physics_fit='matched_noise' if case['protocol']=='awgn_mixed' else 'clean'))
        w = self.tables[split_name(case['domain'], case['ratio'], case['subset'])]
        # 训练器只接收训练和验证观测，测试信号不进入拟合或选模。
        mask = w.role.isin(['train','validation']).to_numpy()
        self.state.update(stage=stage, active_model=dict(job_id=jid, **case, aux_weight=actual_weight))
        self.persist()
        info = engine.train(self.x[mask], w.loc[mask].reset_index(drop=True), case['protocol'],
            case['variant'], case['seed'], case['subset'], folder, self.device, spec)
        job = dict(info, **{k:v for k,v in case.items() if k not in info}, job_id=jid,
                   stage=stage, aux_weight=actual_weight, folder=str(folder),
                   source_load_hp=-1 if case['domain']=='window_random' else int(case['domain'][-1]))
        self.jobs.append(job)
        self.state.pop('active_model', None)
        self.persist()
        print('MODEL COMPLETE', jid, case, flush=True)
        return job

    def run(self):
        self.setup()
        for variant in VARIANTS:
            for weight in (GRID if variant == 'data_aux_ce' else [.1]):
                for seed in SEEDS:
                    self.ensure(dict(domain='window_random',ratio=1.,subset=0,protocol='clean_only',
                                     variant=variant,seed=seed), weight, 'lambda_selection')
        weight, rows = select_lambda(self.jobs)
        self.state.update(selected_aux_weight=weight, lambda_locked=True, lambda_locked_at=now())
        write_json(self.root/'selection.json', dict(selected_aux_weight=weight, candidates=rows,
                   rule=self.state['selection_rule'], test_accessed=False, max_epochs=200))
        report.selection_report(self.root, self.jobs, rows, weight)
        self.persist()
        for domain in ['window_random', *[f'load_source_{s}' for s in range(4)]]:
            for case in cases(domain):
                stage = 'window_random' if domain=='window_random' else 'load_transfer'
                self.ensure(case, weight, stage)
        assert len(self.jobs) == 1020
        self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def evaluate(self):
        if not self.state['decisions_locked'] or self.state['target_test_accessed']:
            raise ValueError('测试须全部训练和决策锁定，且只执行一次')
        main = [j for j in self.jobs if j['variant']=='baseline_data' or j['aux_weight']==self.state['selected_aux_weight']]
        assert len(main) == 1000
        dest = self.root/'evaluation'
        dest.mkdir(exist_ok=False)
        (dest/'predictions').mkdir()
        self.state.update(stage='testing', target_test_accessed=True)
        self.persist()
        total_pairs = total_metrics = 0
        for j in main:
            metrics, confusion, evaluations = [], [], []
            w = self.tables[split_name(j['domain'], j['ratio'], j['subset'])]
            model = engine.load(j['folder'], self.device)
            source = j['source_load_hp']
            for target in ([-1] if source == -1 else [t for t in range(4) if t != source]):
                ix = np.flatnonzero(w.role.eq('test') if target==-1 else w.load_hp.eq(target))
                labels = w.iloc[ix].label.to_numpy()
                arrays = dict(labels=labels, window_indices=w.iloc[ix].window_index.to_numpy())
                meta = {k:j[k] for k in ['job_id','domain','ratio','subset','protocol','variant','seed','aux_weight','source_load_hp']}
                meta['target_load_hp'] = target
                for condition, noise_seed, observed in R.conditions(self.x[ix], j['protocol'], True):
                    p = R.predict(model, observed, w.iloc[ix].rpm.to_numpy(np.float32), self.device)
                    arrays[f'{condition}_n{noise_seed}'] = p
                    values, counts = scores(labels, p)
                    m = dict(meta, condition=condition, noise_seed=noise_seed)
                    metrics.append(dict(m, **values))
                    for a in range(4):
                        for b in range(4):
                            confusion.append(dict(m, true_label=a, predicted_label=b, count=int(counts[a,b]),
                                support=int(counts[a].sum()), row_percent=float(100*counts[a,b]/counts[a].sum())))
                path = dest/'predictions'/f'{j["job_id"]}_target{target}.npz'
                np.savez_compressed(path, **arrays)
                evaluations.append(dict(meta, prediction_path=str(path), test_windows=len(ix)))
            del model
            # 每模型结果立即落盘，中断时保留已完成预测及指标，不自动重测。
            for filename, rows in [('metrics.csv',metrics), ('confusion_matrices.csv',confusion),
                                   ('evaluations.csv',evaluations)]:
                path = dest/filename
                pd.DataFrame(rows).to_csv(path, index=False, mode='a', header=not path.exists())
            total_pairs += len(evaluations)
            total_metrics += len(metrics)
            self.state.update(evaluated_pairs=total_pairs, metric_rows=total_metrics)
            self.persist()
            print('TEST COMPLETE', j['job_id'], flush=True)
        assert total_pairs==2600 and total_metrics==35100
        report.finish(self.root)
        self.state.update(stage='complete', completed_at=now(), evaluated_pairs=total_pairs, metric_rows=total_metrics)
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('30 selection + 190 remaining random-window + 800 transfer = 1020 new models; 1000 main models; 2600 test pairs; 35100 metric rows. All decisions before testing.')
        return
    if args.output is None:
        parser.error('run需要--output全新目录')
    experiment = Experiment(args.output)
    with (experiment.root/'training_console.log').open('x',encoding='utf-8',buffering=1) as log:
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
