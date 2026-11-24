"""固定一个已有子集，三种子，先标准化后系数，不访问测试预测。"""
from pathlib import Path
import argparse
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
import traceback
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose import R, BASE, OUT as HISTORY, HERE
from splits import read_fixed
from models import baseline_equivalence
from . import engine, design
from .runner import now, write_json

from experiment_paths import resolve_result_path

PROJECT=HERE.parents[3]
SEEDS=[1706,1707,1708]
RATIOS=[.1,.2]
SUBSET=4101
SPEC={**design.DEFAULT_SPEC,'max_epochs':30}


def cases(protocols):
    return [dict(ratio=r,subset=SUBSET,protocol=p,seed=s) for r in RATIOS for p in protocols for s in SEEDS]


def assess_physics(frame):
    keys=['ratio','subset','protocol','seed','condition']
    a=frame[frame.physics_fit.eq('matched_noise')].set_index(keys)
    b=frame[frame.physics_fit.eq('clean')].set_index(keys)
    if not a.index.is_unique or not b.index.is_unique or set(a.index)!=set(b.index):
        raise ValueError('标准化对照不完整或重复')
    expected={(c['ratio'],SUBSET,'awgn_mixed',c['seed'],cond) for c in cases(['awgn_mixed'])
              for cond in ['clean',*design.NOISE_CONDITIONS]}
    if set(a.index)!=expected or not frame.noise_seed.isin([0,6201]).all():
        raise ValueError('只能用完整预设验证条件选择标准化')
    paired=a.reset_index()[keys].copy()
    for metric in ['balanced_acc','recall_OR','validation_nll']:
        paired[metric+'_gain']=a[metric].to_numpy()-b.loc[a.index,metric].to_numpy()
    strong=paired[paired.condition.isin(['white_-4','white_-2'])]
    model_gain=strong.groupby(['ratio','subset','seed']).balanced_acc_gain.mean()
    gain=float(model_gain.mean())
    clean=float(paired[paired.condition.eq('clean')].balanced_acc_gain.mean())
    or_gain=float(strong.recall_OR_gain.mean())
    checks=dict(mean_gain=gain>=.005-1e-12,positive_pairs=float((model_gain>1e-12).mean())>=2/3-1e-12,
                clean_preservation=clean>=-.005-1e-12,or_preservation=or_gain>=-.01-1e-12,
                each_budget_preservation=bool(strong.groupby('ratio').balanced_acc_gain.mean().ge(-.005-1e-12).all()))
    adopted=all(checks.values())
    return dict(adopted=adopted,selected_physics_fit='matched_noise' if adopted else 'clean',checks=checks,
                noisy_ba_gain_pp=100*gain,clean_ba_gain_pp=100*clean,or_gain_pp=100*or_gain,
                positive_pair_fraction=float((model_gain>1e-12).mean()),sd_is_hard_gate=False,
                decision_data='fixed validation only',scope='one fixed subset, three seeds, 30 epochs; provisional screening'),paired


class Quick:
    def __init__(self,root):
        self.root=Path(root).resolve()
        if self.root.parent!=(PROJECT/'experiments/CWRU/实验结果/CWRU_V2').resolve() or not self.root.name.startswith('quick_'):
            raise ValueError('只允许experiments/CWRU/实验结果/CWRU_V2下全新quick_*目录')
        if self.root.exists():
            raise FileExistsError('拒绝覆盖或自动恢复已有短程实验')
        for reference in [BASE,HISTORY]:
            if json.loads((reference/'config.json').read_text(encoding='utf-8'))['stage']!='complete':
                raise ValueError('历史基线未完成')
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE=self.device;R.U.configure();baseline_equivalence(self.device)
        self.records,self.original,raw=R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        self.x=R.U.normalize(raw)
        if len(self.original)!=687 or len(self.records)!=40:
            raise ValueError('固定数据范围不符')
        self.windows={}
        for ratio in RATIOS:
            w=read_fixed(HISTORY/'subsets'/f'ratio_{int(ratio*100):03d}_subset_{SUBSET}')
            for column in ['window_index','record_id','label','start_sample','end_sample']:
                np.testing.assert_array_equal(w[column],self.original[column])
            fixed=read_fixed(BASE/'splits/window_random')
            for role in ['validation','test']:
                np.testing.assert_array_equal(w.loc[w.role.eq(role),'window_index'],fixed.loc[fixed.role.eq(role),'window_index'])
            self.windows[ratio]=w
        self.root.mkdir(parents=True,exist_ok=False)
        self.jobs=[]
        self.state=dict(stage='prepared',started_at=now(),jobs=self.jobs,trained_models=0,planned_models=42,
            target_test_accessed=False,historical_test_exposure=True,subset=SUBSET,seeds=SEEDS,ratios=RATIOS,
            spec=SPEC,stage_order=['physics_fit','remaining_lambda'],selected_physics_fit=None,
            short_run_only=True,automatic_extension=False,hardware=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU')
        (self.root/'PROTOCOL.md').write_text((HERE/'QUICK_PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
        self.records.to_csv(self.root/'records.csv',index=False)
        usage=[]
        for ratio,w in self.windows.items():
            out=self.root/'splits'/f'ratio_{int(ratio*100):03d}'
            out.mkdir(parents=True)
            w.to_csv(out/'windows.csv',index=False)
            for role in ['train','validation','test']:
                (out/f'{role}.txt').write_text(''.join(f'{i}\n' for i in w.loc[w.role.eq(role),'window_index']),encoding='utf-8')
                for record in self.records.to_dict('records'):
                    usage.append(dict(ratio=ratio,subset=SUBSET,role=role,record_id=record['record_id'],
                                      raw_file=record['path'],windows=int((w.role.eq(role)&w.record_id.eq(record['record_id'])).sum())))
        pd.DataFrame(usage).to_csv(self.root/'dataset_file_usage.csv',index=False)
        (self.root/'DATA_USAGE.md').write_text('固定40条记录、687窗；10%训练49窗、20%训练98窗，各使用103验证窗。'
            '测试103窗仅保存划分，不读取测试预测或开展测试。逐MAT与集合数量见dataset_file_usage.csv。'
            '单一固定子集4101，三训练种子不是三个独立数据集。',encoding='utf-8')
        self.persist()

    def persist(self):
        self.state.update(jobs=self.jobs,trained_models=len(self.jobs),updated_at=now())
        write_json(self.root/'config.json',self.state)
        if self.jobs:pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv',index=False)

    def ensure(self,spec,case,stage):
        effective={**spec,'physics_fit':'clean'} if case['protocol']=='clean_only' else dict(spec)
        cid=design.config_id('data_aux_ce',effective)
        existing=next((j for j in self.jobs if j['config_id']==cid and all(j[k]==v for k,v in case.items())),None)
        if existing:return existing
        jid=f'm{len(self.jobs)+1:03d}'
        folder=self.root/'models'/jid
        info=engine.train(self.x,self.windows[case['ratio']],case['protocol'],'data_aux_ce',case['seed'],SUBSET,folder,self.device,effective)
        job=dict(job_id=jid,config_id=cid,folder=str(folder),stage=stage,**case,
                 physics_fit=effective['physics_fit'],aux_weight=effective['aux_weight'],spec=effective,**{
                 k:info[k] for k in ['epochs_trained','best_epoch','parameters','conv_linear_flops','latency_ms',
                                     'train_seconds_per_epoch','peak_gpu_mib','gradient_diagnostic_seconds']})
        self.jobs.append(job);self.persist()
        print(f'COMPLETE {jid} stage={stage} lambda={effective["aux_weight"]} fit={effective["physics_fit"]} {case}',flush=True)
        return job

    def metrics(self,jobs):
        frames=[]
        for job in jobs:
            frame=pd.read_csv(resolve_result_path(job['folder'])/'validation_metrics.csv')
            for k in ['job_id','config_id','ratio','subset','protocol','seed','physics_fit','aux_weight']:
                frame[k]=job[k]
            frames.append(frame)
        return pd.concat(frames,ignore_index=True)

    def report(self,name,jobs,decision):
        out=self.root/'stages'/name;out.mkdir(parents=True,exist_ok=False)
        metrics=self.metrics(jobs);metrics.to_csv(out/'validation_metrics.csv',index=False)
        keys=['ratio','protocol','physics_fit','aux_weight','condition']
        summary=metrics.groupby(keys)[['acc','balanced_acc','recall_OR','validation_nll']].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
        summary.to_csv(out/'mean_std.csv',index=False)
        write_json(out/'decision.json',decision)
        curves=[];trends=[]
        plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
        for job in jobs:
            h=pd.read_csv(resolve_result_path(job['folder'])/'training.csv')
            h['job_id']=job['job_id'];curves.append(h)
            last=h.tail(10)
            trend=dict(job_id=job['job_id'],ratio=job['ratio'],seed=job['seed'],protocol=job['protocol'],
                       physics_fit=job['physics_fit'],aux_weight=job['aux_weight'],
                       last5_nll=float(h.tail(5).validation_nll.mean()),
                       preceding5_nll=float(h.iloc[-10:-5].validation_nll.mean()),
                       last10_nll_step_sd=float(last.validation_nll.diff().dropna().std()),
                       last10_clean_acc_range_pp=float(100*(last.validation_clean_acc.max()-last.validation_clean_acc.min())))
            trend['last5_nll_relative_improvement']=(trend['preceding5_nll']-trend['last5_nll'])/max(trend['preceding5_nll'],1e-8)
            if 'validation_white_-4_balanced_acc' in h:
                v=last['validation_white_-4_balanced_acc']
                trend.update(last10_minus4_ba_range_pp=float(100*(v.max()-v.min())),last10_minus4_ba_step_sd_pp=float(100*v.diff().dropna().std()))
            trends.append(trend)
        pd.DataFrame(trends).to_csv(out/'curve_diagnosis.csv',index=False)
        curve=pd.concat(curves,ignore_index=True);curve.to_csv(out/'training_curves.csv',index=False)
        plots=[]
        for ratio in RATIOS:
            for protocol in sorted({j['protocol'] for j in jobs}):
                fig,axes=plt.subplots(2,2,figsize=(12,8))
                relevant=[j for j in jobs if j['ratio']==ratio and j['protocol']==protocol]
                if not relevant:plt.close(fig);continue
                groups=sorted({(j['physics_fit'],j['aux_weight']) for j in relevant})
                for color,(fit,weight) in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'],groups):
                    ids=[j['job_id'] for j in relevant if j['physics_fit']==fit and j['aux_weight']==weight]
                    h=curve[curve.job_id.isin(ids)]
                    label=f'{fit}, λ={weight:g}'
                    fields=['main_ce','validation_nll','validation_clean_acc',
                            'validation_white_-4_balanced_acc' if protocol=='awgn_mixed' else 'train_acc_sampled_augmented']
                    for ax,field in zip(axes.flat,fields):
                        for _,single in h.groupby('job_id'):ax.plot(single.epoch,single[field],color=color,alpha=.2,lw=.8)
                        average=h.groupby('epoch')[field].mean();ax.plot(average.index,average.values,color=color,label=label)
                        ax.set(xlabel='epoch',ylabel=field);ax.grid(alpha=.2)
                axes[0,0].legend(fontsize=8);fig.suptitle(f'{name} | {ratio:.0%} | {protocol} | 细线各种子，粗线均值')
                fig.tight_layout();stem=f'curves_{int(ratio*100)}_{protocol}'
                for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=140)
                plt.close(fig);plots.append(f'{stem}.png')
        text=['# 30轮探索验证报告','',json.dumps(decision,ensure_ascii=False,indent=2),'',
              '仅固定验证集，未测试。每配置三训练种子、一个固定子集。表内准确率为0至1比例。',
              summary.round(6).to_markdown(index=False),'',
              'LOSS下降趋势、最后10轮变化幅度和种子曲线见curve_diagnosis.csv及图。最后10轮仍下降不能判定收敛；'
              '波动幅度仅描述，不以0.5点SD否决。训练ACC来自采样及增强观测，不能视为全部干净训练集ACC。'
              '本轮没有同预算数据单分支，不据此声称优于数据单分支；历史100轮不作30轮性能对照。']
        text.extend(f'\n![曲线]({p})' for p in plots)
        (out/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
        return out

    def run(self):
        try:
            self.state['stage']='physics_fit';self.persist()
            jobs=[]
            for fit in ['clean','matched_noise']:
                for case in cases(['awgn_mixed']):jobs.append(self.ensure({**SPEC,'physics_fit':fit},case,'physics_fit'))
            decision,paired=assess_physics(self.metrics(jobs))
            self.state.update(physics_decision=decision,selected_physics_fit=decision['selected_physics_fit']);self.persist()
            out=self.report('physics_fit',jobs,decision);paired.to_csv(out/'paired_validation.csv',index=False)
            diagnostics=[]
            for job in jobs:
                if job['seed']!=SEEDS[0]:continue
                model=engine.load(resolve_result_path(job['folder']),self.device)
                table=engine.physical_standardization_diagnostic(model,self.x,self.windows[job['ratio']],self.device,job['spec'])
                for key in ['job_id','ratio','physics_fit']:table[key]=job[key]
                diagnostics.append(table);del model
            pd.concat(diagnostics,ignore_index=True).to_csv(out/'physical_feature_diagnosis.csv',index=False)
            self.state.update(stage='remaining_lambda',physics_completed_at=now());self.persist()
            spec={**SPEC,'physics_fit':decision['selected_physics_fit']}
            jobs=[]
            for weight in [.1,.3,1.]:
                for case in cases(['clean_only','awgn_mixed']):jobs.append(self.ensure({**spec,'aux_weight':weight},case,'remaining_lambda'))
            self.report('remaining_lambda',jobs,dict(selected_physics_fit=spec['physics_fit'],
                        coefficients=[.1,.3,1.],best_lambda_not_selected=True,scope='30轮探索，不以短程测试决定最终方案'))
            self.state.update(stage='complete',completed_at=now(),target_test_accessed=False);self.persist()
            (self.root/'REPORT.md').write_text('# 短程探索完成\n\n'
                '共42个从头训练的30轮上限模型。先用固定lambda=0.1比较物理标准化，再固定暂定标准化探索0.3和1.0。\n\n'
                '[标准化对照](stages/physics_fit/REPORT.md) · [系数曲线](stages/remaining_lambda/REPORT.md)\n\n'
                '只用训练与验证。未访问测试预测，未新增测试。单个固定子集，结果仅作探索；旧结果保留且已有测试曝光须披露。',encoding='utf-8')
            print('QUICK EXPLORATION COMPLETE; NO TEST; NO FURTHER STAGES',flush=True)
        except BaseException as error:
            self.state.update(stage='failed',error_type=type(error).__name__,error=str(error));self.persist();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    quick=Quick(args.output)
    with (quick.root/'training_console.log').open('x',encoding='utf-8',buffering=1) as stdout, \
         (quick.root/'training_errors.log').open('x',encoding='utf-8',buffering=1) as stderr:
        with redirect_stdout(stdout),redirect_stderr(stderr):
            try:
                quick.run()
            except BaseException:
                traceback.print_exc()
                raise
