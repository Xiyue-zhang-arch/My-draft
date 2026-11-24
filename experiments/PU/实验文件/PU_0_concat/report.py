"""Recompute metrics for the six four-harmonic experiment groups."""
import torch
import csv
from pathlib import Path
import numpy as np
from common import *

REPORT_METHODS=METHODS
COLS=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES])
SHORT=('V','V+P','C','C+P','Concat','Concat+P')

def entries(out):
    for spec in models():
        for case in cases():yield dict(**spec,source='new'),case,out/'predictions'/spec['model_id']/(case['case_id']+'.npz')

def collect(out,synthetic=False):
    state=read(out/'config.json')
    if bool(state.get('synthetic',False))!=synthetic or not state['decisions_locked'] or not state['weights_verified'] or len(state['completed_models'])!=30 or state['prediction_cases']!=1890:raise ValueError('Complete 30-model/1890-case barrier required')
    if [{k:r[k] for k in ('model_id','method','seed')} for r in state['completed_models']]!=models():raise ValueError('Wrong model plan')
    index=np.load(FROZEN_SPLIT/'split_indices.npz');windows=read(DATA/'windows.json')
    desc={r['case_id']:r for r in read(out/'test_cases.json')}
    if len(desc)!=63:raise ValueError('Incomplete case descriptions')
    rows=[];cms=[];newhash={};clean={};invariant=0;error=0.
    for spec,case,path in entries(out):
        ids=index[case['target']];y=np.array([windows[int(i)]['label'] for i in ids])
        with np.load(path) as z:
            if not np.array_equal(z['y'],y) or not np.array_equal(z['window_index'],ids):raise ValueError('Sample identity mismatch')
            if str(z['method'])!=spec['method'] or str(z['model_id'])!=spec['model_id'] or int(z['seed'])!=spec['seed'] or str(z['case_id'])!=case['case_id']:raise ValueError('Model/case identity mismatch')
            if str(z['waveform_input_sha256'])!=desc[case['case_id']]['waveform_input_sha256'] or str(z['descriptor_input_sha256'])!=desc[case['case_id']]['descriptor_input_sha256']:raise ValueError('Prediction input hash mismatch')
            m,p=metrics(z['logits'],y);e=float(np.max(np.abs(p-z['probability'])));error=max(error,e)
            if e>1e-12:raise ValueError('Probability mismatch')
            s=z['scale_mean'];branches=1 if spec['method'].startswith(('vibration_','current_')) else 2
            if s.shape!=(4000,branches) or not np.isfinite(s).all() or s.min()<.75-1e-6 or s.max()>1.25+1e-6:raise ValueError('Invalid channel-scale summaries')
            if spec['method'] in ('vibration_plain','current_plain','concat_plain') and not np.array_equal(s,np.ones_like(s)):raise ValueError('Unconditioned model altered channels')
            if spec['method'] in ('current_plain','current_physical'):
                key=(spec['model_id'],case['target'])
                if case['snr_db'] is None:clean[key]=z['logits'].copy()
                else:
                    if not np.array_equal(clean[key],z['logits']):raise ValueError('Unused noisy modality changed current-only output')
                    invariant+=1
            rows.append(dict(**spec,case_id=case['case_id'],target=case['target'],endpoint='clean' if case['snr_db'] is None else 'noisy',snr_db=case['snr_db'],noise_seed=case['noise_seed'],
                             **{k:m[k] for k in ('ACC','BA','Macro_F1','CE')},**{'recall_'+c:float(m['recall'][i]) for i,c in enumerate(CLASS_NAMES)}))
            cms.append(m['confusion'])
        newhash[str(path.relative_to(out))] = sha(path)
    if len(rows)!=1890 or len(newhash)!=1890 or invariant!=600 or len(list((out/'predictions').glob('*/*.npz')))!=1890:
        raise ValueError('Incorrect prediction scope')
    return rows,np.stack(cms),dict(new_predictions=1890,reference_predictions=0,new_confusion_cells=30240,
                                   probability_max_error=error,current_only_unused_channel_invariance=600,new_prediction_sha256=newhash)

def summarize(rows):
    perseed=[];summary=[];targets=[];pairs=[]
    for method in REPORT_METHODS:
        for endpoint in ('clean','noisy'):
            group=[]
            for s in SEEDS:
                r=[v for v in rows if v['method']==method and v['seed']==s and v['endpoint']==endpoint]
                if len(r)!=(3 if endpoint=='clean' else 60):raise ValueError('Unbalanced endpoint')
                a=dict(method=method,seed=s,endpoint=endpoint,**{k:float(np.mean([x[k] for x in r])) for k in COLS});group.append(a);perseed.append(a)
            summary.append(dict(method=method,endpoint=endpoint,**{k+suffix:float(np.mean([x[k] for x in group])) if suffix=='_mean' else float(np.std([x[k] for x in group],ddof=1)) for k in COLS for suffix in ('_mean','_SD')}))
        for target in TARGETS:
            for snr in (None,*SNRS):
                group=[]
                for s in SEEDS:
                    r=[v for v in rows if v['method']==method and v['seed']==s and v['target']==target and v['snr_db']==snr]
                    group.append({k:float(np.mean([x[k] for x in r])) for k in COLS})
                targets.append(dict(method=method,target=target,snr_db=snr,**{k+suffix:float(np.mean([x[k] for x in group])) if suffix=='_mean' else float(np.std([x[k] for x in group],ddof=1)) for k in COLS for suffix in ('_mean','_SD')}))
    comparisons=[('concat_physical',m) for m in REPORT_METHODS if m!='concat_physical']+[
        ('concat_plain','vibration_plain'),('concat_plain','current_plain'),
        ('vibration_physical','vibration_plain'),('current_physical','current_plain')]
    for endpoint in ('clean','noisy'):
        def v(m):return np.array([next(r['BA'] for r in perseed if r['method']==m and r['seed']==s and r['endpoint']==endpoint) for s in SEEDS])
        for a,b in comparisons:
            d=v(a)-v(b)
            pairs.append(dict(endpoint=endpoint,method=a,comparator=b,BA_difference_mean=float(d.mean()),BA_difference_SD=float(d.std(ddof=1)),positive_seeds=int((d>1e-12).sum()),per_seed=d.tolist()))
    return perseed,summary,targets,pairs

def draw(out,summary,targets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=out/'report';fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,endpoint in zip(axes,('clean','noisy')):
        r=[next(x for x in summary if x['method']==m and x['endpoint']==endpoint) for m in REPORT_METHODS]
        ax.barh(np.arange(len(REPORT_METHODS)),[x['BA_mean']*100 for x in r],xerr=[x['BA_SD']*100 for x in r],capsize=3)
        ax.set_yticks(np.arange(len(REPORT_METHODS)),SHORT);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_title(endpoint);ax.set_xlabel('Balanced accuracy (%)')
    fig.savefig(report/'primary_BA.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,target in zip(axes,TARGETS):
        for method in ('vibration_physical','current_physical','concat_plain','concat_physical'):
            r=[next(x for x in targets if x['method']==method and x['target']==target and x['snr_db']==s) for s in SNRS]
            ax.errorbar(SNRS,[x['BA_mean']*100 for x in r],yerr=[x['BA_SD']*100 for x in r],marker='o',capsize=2,label=SHORT[REPORT_METHODS.index(method)])
        ax.set_title(target);ax.set_xlabel('Raw record vibration SNR (dB)');ax.set_ylim(0,105)
    axes[0].set_ylabel('Balanced accuracy (%)');axes[-1].legend(fontsize=8)
    fig.savefig(report/'noise_BA.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,method in zip(axes.flat,METHODS):
        for spec in [m for m in models() if m['method']==method]:
            with (out/'models'/spec['model_id']/'history.csv').open(encoding='utf-8',newline='') as f:r=list(csv.DictReader(f))
            ax.plot([int(x['epoch']) for x in r],[float(x['val_CE']) for x in r])
        ax.set_title(method);ax.set_xlabel('Epoch');ax.set_ylabel('Source validation CE');ax.set_yscale('log')
    fig.savefig(report/'validation_CE.png',dpi=160);plt.close(fig)

def generate(out,synthetic=False,figures=True):
    out=Path(out);rows,cm,proof=collect(out,synthetic);perseed,summary,targets,pairs=summarize(rows)
    report=out/'report';report.mkdir(exist_ok=False)
    for name,data in [('metrics_recomputed.csv',rows),('per_seed_endpoints.csv',perseed),('endpoints_summary.csv',summary),('target_SNR_summary.csv',targets),('paired_differences.csv',pairs)]:csvwrite(report/name,data)
    npz(report/'confusions.npz',confusion=cm)
    lines=['# PU_0 直接拼接与单模态补充实验','',
           '本批次含30个模型和1890份预测。每个模型训练50轮，共2500次更新。使用物理特征的组采用四谐波描述符 PU_harmonic_mean_20261122_a。',
           '','|方法|干净BA，%|仅振动受扰BA，%|','|---|---:|---:|']
    for method in REPORT_METHODS:
        a,b=[next(r for r in summary if r['method']==method and r['endpoint']==e) for e in ('clean','noisy')]
        lines.append(f"|{method}|{a['BA_mean']*100:.2f} ± {a['BA_SD']*100:.2f}|{b['BA_mean']*100:.2f} ± {b['BA_SD']*100:.2f}|")
    lines+=['','直接拼接两个64维特征为128维，不做平均、缩放或融合MLP。分类头从Linear64至4改成Linear128至4，多256参数。初始化两半为原64维权重的一半，训练后完全独立，不保留等权约束；只在eval初始化时与原均值函数等价。Dropout维数变化，因此旧均值与新拼接的差不能作为完全排除容量和随机掩码差异的纯操作效应。',
            '','单模态分别包含普通CNN和仅自身描述符调制版本，严禁读取另一模态信息。自己的4维描述符重复为8维只是匹配调制器结构。无预训练、无辅助头、单一最终交叉熵。',
            '','六组共享3200个来源训练窗、800个来源验证窗、波形定标和目标窗口。物理描述符定标只使用来源训练窗。',
            '','本批次按种子配对比较六组模型。旧等权结果见历史正式报告。',
            '','历史目标已曝光，研究为探索性，固定已知轴承和工况不支持新轴承总体泛化。50轮最后模型不等于全部充分收敛。原PU_0和PU_1全部保留。']
    if synthetic:lines.insert(2,'这是人工预测的程序测试，不是正式性能结果。')
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    review=['# 结果评审','']
    for endpoint in ('clean','noisy'):
        review+=['## '+endpoint]
        for r in [r for r in pairs if r['endpoint']==endpoint]:review.append(f"- {r['method']}减{r['comparator']}，BA差{r['BA_difference_mean']*100:+.4f}个百分点，种子SD {r['BA_difference_SD']*100:.4f}，正差{r['positive_seeds']}/5。")
    for target in TARGETS:
        r=next(r for r in targets if r['method']=='concat_physical' and r['target']==target and r['snr_db'] is None)
        review.append(f"\n{target}主方案干净BA {r['BA_mean']*100:.2f} ± {r['BA_SD']*100:.2f}%，外圈/内圈/复合召回{r['recall_outer_mean']*100:.2f}/{r['recall_inner_mean']*100:.2f}/{r['recall_compound_mean']*100:.2f}%。")
    review+=['','判断优于单模态时必须同时报告两个物理单模态，不只挑较弱的普通基线。若提升只见于特定类别、工况或种子，不称全面或普适优势。物理调制仍是已有方案适配，不将预测增益当作模态信息量或机械因果证据。']
    (report/'REVIEW.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    if not synthetic:
        deg=[]
        for spec in models():
            with (out/'models'/spec['model_id']/'history.csv').open(encoding='utf-8',newline='') as f:h=list(csv.DictReader(f))
            modalities=(0,1) if spec['method'].startswith('concat') else (0,) if spec['method'].startswith('vibration') else (1,)
            zero=[int(r['epoch']) for r in h if any(float(r['gradient_v' if m==0 else 'gradient_c'])==0 for m in modalities)]
            with np.load(out/'models'/spec['model_id']/'validation.npz') as z:classes=len(np.unique(z['logits'].argmax(1)))
            deg.append(dict(**spec,zero_gradient_epoch_count=len(zero),zero_gradient_epochs=zero,validation_predicted_classes=classes))
        csvwrite(report/'degeneration_summary.csv',deg)
    if figures and not synthetic:draw(out,summary,targets)
    proof.update(synthetic=synthetic,report_files_sha256={p.name:sha(p) for p in report.iterdir() if p.is_file()});write(report/'verification.json',proof)
    return proof
