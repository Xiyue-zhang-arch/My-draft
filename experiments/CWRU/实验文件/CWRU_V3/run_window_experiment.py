"""固定已选lambda，只完成V3记录内随机窗口的全样本、小样本和AWGN实验。"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import json
import os
import shutil
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from context import R, PROJECT, HERE, BASE
from splits import read_fixed
from run_experiment import Experiment, cases, split_name, select_lambda, SEEDS, SUBSETS, RATIOS, now, write_json
from metrics import scores
import engine

from experiment_paths import resolve_result_path

SOURCE = PROJECT/'experiments/CWRU/实验结果/CWRU_V3/run_20261022_2105'
CASE_KEYS = ['domain','ratio','subset','protocol','variant','seed']


def source_state(source):
    source = Path(source).resolve()
    state = json.loads((source/'config.json').read_text(encoding='utf-8'))
    decision = json.loads((source/'selection_evaluation/selection.json').read_text(encoding='utf-8'))
    if not (state['version']=='CWRU_V3' and state['stage']=='paused' and state['trained_models']==30
            and state['lambda_locked'] and state['selection_test_status']=='complete'
            and state['selection_test_models']==30 and state['pause_boundary_verified']):
        raise ValueError('选参阶段必须完整30模型、测试完成并在核验边界暂停')
    weight, rows = select_lambda(state['jobs'])
    if weight!=state['selected_aux_weight'] or weight!=decision['selected_aux_weight'] or rows!=decision['candidates']:
        raise ValueError('已锁定系数与原验证选择不一致，禁止修改系数')
    return state, weight


def reference_jobs(source, state, weight):
    full = read_fixed(source/'splits/window_random_r100_s0')
    jobs=[]
    for saved in state['jobs']:
        if saved['variant']!='baseline_data' and saved['aux_weight']!=weight:
            continue
        j = copy.deepcopy(saved)
        folder = resolve_result_path(j['folder']).resolve()
        if folder.parent!=(source/'models').resolve():
            raise ValueError('引用模型必须来自已锁定V3选参目录')
        info = json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
        if not (info['completed'] and j['domain']=='window_random' and j['ratio']==1.
                and j['subset']==0 and j['protocol']=='clean_only' and info['max_epochs']==200
                and info['spec']['schedule_epochs']==200 and info['spec']['aux_head']=='shared'):
            raise ValueError('引用设置与全样本纯干净V3不匹配')
        if j['variant']=='data_aux_ce' and info['spec']['aux_weight']!=weight:
            raise ValueError('引用融合系数不一致')
        for name in ['model.pt','training_spec.json','training.csv','validation_curve.csv','validation_predictions.npz']:
            if not (folder/name).is_file():
                raise ValueError(f'引用模型文件缺失: {folder/name}')
        selected=pd.read_csv(folder/'selected_windows.csv')
        np.testing.assert_array_equal(selected.window_index,full.loc[full.role.eq('train'),'window_index'])
        val=pd.read_csv(folder/'validation_metrics.csv')
        if len(val)!=1 or val.iloc[0]['condition']!='clean' or not np.isclose(
                val.iloc[0].validation_nll,j['validation_nll'],rtol=0,atol=2e-6):
            raise ValueError('引用验证指标与保存的最佳CE不一致')
        prediction=source/'selection_evaluation/predictions'/f'{j["job_id"]}.npz'
        if not prediction.is_file():
            raise FileNotFoundError('已测试模型的预测缺失，禁止用重测代替')
        j.update(source_job_id=j['job_id'],job_id=f'r{len(jobs)+1:04d}',origin='selection_reference',
                 reference_prediction=str(prediction),reference_source=str(source))
        jobs.append(j)
    expected={(v,seed) for v in ['baseline_data','data_aux_ce'] for seed in SEEDS}
    if len(jobs)!=10 or {(j['variant'],j['seed']) for j in jobs}!=expected:
        raise ValueError('需完整两模型各5种子引用')
    return jobs


class WindowExperiment(Experiment):
    def __init__(self, root, source=SOURCE):
        self.root=Path(root).resolve()
        self.source=Path(source).resolve()
        if self.root.parent!=(PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not self.root.name.startswith('window_random_'):
            raise ValueError('结果须为experiments/CWRU/实验结果/CWRU_V3下新的window_random_*直接子目录')
        if self.root.exists():
            raise FileExistsError('不能覆盖或自动恢复已有输出')
        source_config,weight=source_state(self.source)
        self.jobs=reference_jobs(self.source,source_config,weight)
        self.root.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(HERE/'WINDOW_PROTOCOL.md',self.root/'PROTOCOL.md')
        self.state=dict(version='CWRU_V3_window_random',stage='preparing',started_at=now(),pid=os.getpid(),
            reference_source=str(self.source),selected_aux_weight=weight,lambda_locked=True,
            lambda_selection_source=str(self.source/'selection_evaluation/selection.json'),
            planned_new_models=190,planned_referenced_models=10,planned_models=200,
            planned_test_pairs=200,planned_metric_rows=2700,planned_new_metric_rows=2690,
            planned_reused_predictions=10,trained_models=0,referenced_models=10,
            target_test_accessed=False,historical_test_exposure=True,selection_test_already_accessed=True,
            decisions_locked=False,decision_scope='window_random_only',cross_load_authorized=False,
            automatic_resume=False,seeds=SEEDS,subsets=SUBSETS,ratios=[1.,*RATIOS],
            training_spec=engine.normalize_spec(dict(aux_weight=weight,physics_fit='matched_noise')),
            jobs=self.jobs)
        self.persist()

    def persist(self):
        self.state.update(updated_at=now(),jobs=self.jobs,
            trained_models=sum(j.get('origin')!='selection_reference' for j in self.jobs),
            referenced_models=sum(j.get('origin')=='selection_reference' for j in self.jobs),
            completed_model_cases=len(self.jobs))
        write_json(self.root/'config.json',self.state)
        pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv',index=False)

    def setup(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE=self.device;R.U.configure()
        self.tables={}
        full=read_fixed(self.source/'splits/window_random_r100_s0')
        pd.testing.assert_frame_equal(full,read_fixed(BASE/'splits/window_random'))
        for ratio in [1.,*RATIOS]:
            for subset in ([0] if ratio==1 else SUBSETS):
                name=split_name('window_random',ratio,subset)
                source=self.source/'splits'/name
                w=read_fixed(source)
                for role in ['validation','test']:
                    np.testing.assert_array_equal(w.loc[w.role.eq(role),'window_index'],full.loc[full.role.eq(role),'window_index'])
                if int(w.role.eq('train').sum())!={1.:481,.1:49,.2:98,.5:242}[ratio]:
                    raise ValueError('固定小样本训练预算不一致')
                if not set(w.loc[w.role.eq('train'),'window_index'])<=set(full.loc[full.role.eq('train'),'window_index']):
                    raise ValueError('训练子集越界')
                shutil.copytree(source,self.root/'splits'/name)
                self.tables[name]=w
        for subset in SUBSETS:
            previous=set()
            for ratio in RATIOS:
                w=self.tables[split_name('window_random',ratio,subset)]
                ids=set(w.loc[w.role.eq('train'),'window_index'])
                if not previous<=ids:raise ValueError('预算未严格嵌套')
                previous=ids
        records,original,raw=R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        if len(original)!=687 or original.record_id.nunique()!=40:raise ValueError('原始数据范围不符')
        for w in self.tables.values():
            for key in ['window_index','record_id','label','start_sample','end_sample']:
                np.testing.assert_array_equal(w[key],original[key])
        self.x=R.U.normalize(raw)
        records.to_csv(self.root/'records.csv',index=False)
        for filename in ['dataset_support.csv','dataset_file_usage.csv','training_subset_file_usage.csv']:
            frame=pd.read_csv(self.source/filename)
            frame[frame.domain.eq('window_random')].to_csv(self.root/filename,index=False)
        (self.root/'DATA_USAGE.md').write_text(
            '# V3随机窗口续跑数据用量\n\n全部40记录687完整窗。原始data只读。严格复用已固定的记录内划分，'
            '全样本481训练、103验证、103测试。10%/20%/50%为49/98/242训练窗，各3固定嵌套子集、5训练种子。'
            '验证和测试不随训练预算改变。逐类配额及逐MAT路径、通道和尾部说明见dataset_support.csv、dataset_file_usage.csv。'
            '每模型另存selected_windows，引用模型保留原路径。\n\n'
            '总标注预算包括103验证窗；训练与测试共享原始记录。已有全样本干净测试曝光继续披露。'
            '本轮仅窗口级识别，不是跨负载、跨记录或独立轴承验证。',encoding='utf-8')
        self.state.update(hardware=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',
                          precision='CUDA encoder BF16; remaining FP32' if self.device.type=='cuda' else 'FP32')
        self.persist()

    def run(self):
        self.setup()
        weight=self.state['selected_aux_weight']
        for case in cases('window_random'):
            self.ensure(case,weight,'window_random_training')
        self.check_complete()
        self.state.update(stage='decisions_locked',decisions_locked=True,decisions_locked_at=now())
        write_json(self.root/'selection.json',dict(selected_aux_weight=weight,lambda_selection_source=self.state['lambda_selection_source'],
            decisions_locked=True,decision_scope='window_random_only',historical_test_exposure=True,
            selection_test_already_accessed=True,model_cases=200,new_models=190,referenced_models=10))
        self.persist()
        self.evaluate()

    def check_complete(self):
        actual={tuple(j[k] for k in CASE_KEYS) for j in self.jobs}
        expected={tuple(c[k] for k in CASE_KEYS) for c in cases('window_random')}
        if len(self.jobs)!=200 or actual!=expected or len(actual)!=len(self.jobs):
            raise ValueError('随机窗口两方法、预算、子集和种子组合不完整')
        if sum(j.get('origin')=='selection_reference' for j in self.jobs)!=10:
            raise ValueError('引用数量应为10')
        for j in self.jobs:
            if j['variant']=='data_aux_ce' and j['aux_weight']!=self.state['selected_aux_weight']:
                raise ValueError('不能改变统一lambda')

    def evaluate(self):
        if not self.state['decisions_locked'] or self.state['target_test_accessed']:
            raise ValueError('本轮全部模型选模锁定后才能测试，且只测试一次')
        self.check_complete()
        dest=self.root/'evaluation';dest.mkdir(exist_ok=False);(dest/'predictions').mkdir()
        self.state.update(stage='testing',target_test_accessed=True);self.persist()
        total_metrics=reused=0
        for index,j in enumerate(self.jobs,1):
            w=self.tables[split_name('window_random',j['ratio'],j['subset'])]
            ix=np.flatnonzero(w.role.eq('test'));labels=w.iloc[ix].label.to_numpy()
            if j.get('origin')=='selection_reference':
                path=resolve_result_path(j['reference_prediction'])
                with np.load(path) as archive:arrays={key:archive[key] for key in archive.files}
                reused+=1
            else:
                model=engine.load(j['folder'],self.device)
                arrays=dict(labels=labels,window_indices=w.iloc[ix].window_index.to_numpy())
                for condition,seed,observed in R.conditions(self.x[ix],j['protocol'],True):
                    arrays[f'{condition}_n{seed}']=R.predict(model,observed,w.iloc[ix].rpm.to_numpy(np.float32),self.device)
                del model
                path=dest/'predictions'/f'{j["job_id"]}.npz'
                np.savez_compressed(path,**arrays)
            np.testing.assert_array_equal(arrays['labels'],labels)
            np.testing.assert_array_equal(arrays['window_indices'],w.iloc[ix].window_index)
            expected={'clean_n0'}
            if j['protocol']=='awgn_mixed':expected|={f'white_{snr:+d}_n{seed}' for snr in [-4,-2,0,2,4] for seed in range(7201,7206)}
            if set(arrays)-{'labels','window_indices'}!=expected:raise ValueError('预测条件与固定协议不一致')
            meta={k:j[k] for k in ['job_id','domain','ratio','subset','protocol','variant','seed','aux_weight','source_load_hp']}
            meta.update(target_load_hp=-1,origin=j.get('origin','new'))
            metrics=[];confusion=[]
            for name in sorted(expected):
                condition,noise_seed=name.rsplit('_n',1)
                p=arrays[name]
                if p.shape!=(103,4) or not np.isfinite(p).all():raise ValueError('预测形状或数值无效')
                np.testing.assert_allclose(p.sum(1),1.,rtol=0,atol=1e-5)
                values,counts=scores(labels,p)
                row=dict(meta,condition=condition,noise_seed=int(noise_seed))
                metrics.append(dict(row,**values))
                for a in range(4):
                    for b in range(4):
                        confusion.append(dict(row,true_label=a,predicted_label=b,count=int(counts[a,b]),
                            support=int(counts[a].sum()),row_percent=float(100*counts[a,b]/counts[a].sum())))
            evaluation=dict(meta,test_windows=len(ix),prediction_path=str(path),latency_ms=j['latency_ms'])
            for filename,rows in [('metrics.csv',metrics),('confusion_matrices.csv',confusion),('evaluations.csv',[evaluation])]:
                output=dest/filename
                pd.DataFrame(rows).to_csv(output,index=False,mode='a',header=not output.exists())
            total_metrics+=len(metrics)
            self.state.update(evaluated_pairs=index,metric_rows=total_metrics,reused_test_predictions=reused)
            self.persist();print('TEST COMPLETE',j['job_id'],meta['origin'],flush=True)
        if total_metrics!=2700 or reused!=10:raise ValueError('测试及只读复用数量不符合预定计划')
        import window_report
        window_report.finish(self.root)
        self.state.update(stage='complete',completed_at=now(),stop_reason='window_random_scope_complete',
                          cross_load_started=False)
        self.persist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'],nargs='?',default='plan')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='plan':
        _,weight=source_state(SOURCE)
        print(f'Fixed lambda={weight}; random-window only; 190 new + 10 referenced models; 200 evaluations; 2700 metric rows; 10 predictions reused. Stop after report.')
        return
    if args.output is None:parser.error('run须指定全新--output')
    experiment=WindowExperiment(args.output)
    with (experiment.root/'training_console.log').open('x',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(log),redirect_stderr(log):
            try:experiment.run()
            except BaseException as error:
                experiment.state.update(stage='failed',error=str(error),error_type=type(error).__name__)
                experiment.persist();traceback.print_exc();raise


if __name__=='__main__':
    main()
