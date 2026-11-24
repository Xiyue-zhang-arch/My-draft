"""V3全样本单来源负载迁移独立入口，不运行小样本。"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, shutil, traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import torch
import numpy as np
import pandas as pd
from context import R, HERE, PROJECT, BASE
from splits import read_fixed
from run_experiment import Experiment, split_name, now, write_json, SEEDS, PROTOCOLS, VARIANTS
from run_window_experiment import source_state, SOURCE
from metrics import scores
import engine
import load_report as report


def full_cases():
    for source in range(4):
        for protocol in PROTOCOLS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    yield dict(domain=f'load_source_{source}',ratio=1.,subset=0,protocol=protocol,variant=variant,seed=seed)


def prepare_load_splits(root):
    tables={}
    for source in range(4):
        domain=f'load_source_{source}'
        name=split_name(domain,1.,0)
        w=read_fixed(SOURCE/'splits'/name)
        pd.testing.assert_frame_equal(w,read_fixed(BASE/'splits'/domain))
        assert len(w)==687 and w.record_id.nunique()==40
        assert int(w.role.eq('train').sum())==[73,156,160,160][source]
        assert int(w.role.eq('validation').sum())==[19,39,40,40][source]
        assert set(w.loc[w.role.isin(['train','validation']),'load_hp'])=={source}
        for target in range(4):
            assert int(w.load_hp.eq(target).sum())==[92,195,200,200][target]
        shutil.copytree(SOURCE/'splits'/name,root/'splits'/name)
        tables[name]=w
    for filename in ['dataset_support.csv','dataset_file_usage.csv']:
        a=pd.read_csv(SOURCE/filename)
        a=a[a.domain.str.startswith('load_source_') & a.ratio.eq(1.) & a.subset.eq(0)]
        if filename=='dataset_file_usage.csv':
            a=a[['domain','ratio','subset','role','record_id','used_windows','complete_record_windows']]
        assert len(a)>0
        a.to_csv(root/filename,index=False)
    return tables


class LoadExperiment(Experiment):
    def __init__(self,root):
        self.root=Path(root).resolve()
        if self.root.parent!=(PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not self.root.name.startswith('load_full_'):
            raise ValueError('须使用experiments/CWRU/实验结果/CWRU_V3下全新load_full_*直接子目录')
        if self.root.exists():
            raise FileExistsError('禁止覆盖或自动恢复')
        _,weight=source_state(SOURCE)
        assert weight==.3
        self.jobs=[]
        self.state=dict(version='CWRU_V3_load_full',stage='preparing',started_at=now(),pid=os.getpid(),
            trained_models=0,planned_models=80,planned_new_models=80,planned_test_pairs=240,planned_metric_rows=3240,
            target_test_accessed=False,historical_test_exposure=True,selection_test_already_accessed=True,
            decisions_locked=False,decision_scope='load_transfer_full_only',automatic_resume=False,
            selected_aux_weight=weight,lambda_locked=True,lambda_selection_source=str(SOURCE/'selection_evaluation/selection.json'),
            seeds=SEEDS,ratios=[1.],subsets=[0],training_spec=engine.normalize_spec(dict(aux_weight=weight)),jobs=self.jobs)
        self.root.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(HERE/'LOAD_PROTOCOL.md',self.root/'PROTOCOL.md')
        self.persist()

    def run(self):
        self.setup()
        for case in full_cases():
            self.ensure(case,self.state['selected_aux_weight'],'load_transfer_full')
        assert len(self.jobs)==80 and all(j['completed'] for j in self.jobs)
        assert {tuple(j[k] for k in ['domain','protocol','variant','seed']) for j in self.jobs}=={tuple(c[k] for k in ['domain','protocol','variant','seed']) for c in full_cases()}
        self.state.update(stage='decisions_locked',decisions_locked=True,decisions_locked_at=now())
        self.persist()
        self.evaluate()

    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE = self.device
        R.U.configure()
        self.tables = prepare_load_splits(self.root)
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
            '本轮四来源完整训练73/156/160/160、验证19/39/40/40、目标92/195/200/200。'
            '小样本仍使用完整固定验证标签。子集、训练种子、噪声重复及方向均不能作为独立数据集重复。'
            '历史测试曝光、训练测试共享记录及全负载选参对迁移结论的限制见PROTOCOL.md。', encoding='utf-8')
        self.x = R.U.normalize(raw)
        self.state.update(hardware=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',
                          precision='CUDA encoder BF16; remaining FP32' if self.device.type=='cuda' else 'FP32')
        self.persist()

    def evaluate(self):
        if not self.state['decisions_locked'] or self.state['target_test_accessed']:
            raise ValueError('测试须全部训练和决策锁定，且只执行一次')
        main = [j for j in self.jobs if j['variant']=='baseline_data' or j['aux_weight']==self.state['selected_aux_weight']]
        assert len(main) == 80
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
        assert total_pairs==240 and total_metrics==3240
        report.finish(self.root)
        self.state.update(stage='complete', completed_at=now(), evaluated_pairs=total_pairs, metric_rows=total_metrics)
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('80 new full-source models; fixed lambda 0.3; 12 directions; 240 test pairs; 3240 metric rows. Train all before testing; stop after report.')
        return
    if args.output is None:
        parser.error('run需要--output全新目录')
    experiment = LoadExperiment(args.output)
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
