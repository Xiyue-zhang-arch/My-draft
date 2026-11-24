"""Within-condition summaries recomputed exclusively from saved predictions."""
import torch
import csv
import json
from pathlib import Path
import numpy as np
from common import DATA,METHODS,SEEDS,SNRS,CLASS_NAMES,CONDITIONS,models,cases,descriptor_profile,metrics,sha,write_json,write_csv,save_npz

def collect(out):
    state=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if not state['decisions_locked'] or not state['weights_verified'] or state['prediction_cases']!=2520:
        raise ValueError('Complete training barrier and all 2520 predictions required')
    if [{k:r[k] for k in ('model_id','method','seed')} for r in state['completed_models']]!=models():
        raise ValueError('Exact 30 model plan required')
    descriptions=json.loads((out/'test_cases.json').read_text(encoding='utf-8'))
    if len(descriptions)!=84 or [d['case_id'] for d in descriptions]!=[c['case_id'] for c in cases()]:
        raise ValueError('Missing or reordered cases')
    split=np.load(DATA/'split_indices.npz'); windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    rows=[]; matrices=[]; hashes={}
    for spec in models():
        profile=descriptor_profile(spec['method'])
        for case,desc in zip(cases(),descriptions):
            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64')
            path=out/'predictions'/spec['model_id']/(case['case_id']+'.npz')
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['window_index'],ids) or not np.array_equal(z['y'],y):
                    raise ValueError('Saved labels or window identities changed')
                if any(str(z[k])!=str(v) for k,v in spec.items()) or str(z['case_id'])!=case['case_id'] or str(z['target'])!=case['target'] or str(z['kind'])!='main':
                    raise ValueError('Saved model/case identity changed')
                if str(z['waveform_input_sha256'])!=desc['waveform_input_sha256'] or str(z['descriptor_input_sha256'])!=desc['descriptor_input_sha256'][profile]:
                    raise ValueError('Paired input hashes changed')
                result,p=metrics(z['logits'],y)
                if not np.allclose(p,z['probability'],rtol=0,atol=1e-12):
                    raise ValueError('Probability mismatch')
                scales=z['scale_mean']
                if scales.shape!=(400,2) or not np.isfinite(scales).all() or scales.min()<.75-1e-6 or scales.max()>1.25+1e-6:
                    raise ValueError('Invalid scale statistics')
                if spec['method'] in ('equal','late_concat') and not np.array_equal(scales,np.ones_like(scales)):
                    raise ValueError('Unmodulated branch changed scaling')
                row=dict(**spec,case_id=case['case_id'],condition=case['target'],endpoint='clean' if case['noise_seed'] is None else 'noisy',
                         snr_db=case['snr_db'],noise_seed=case['noise_seed'],**{k:result[k] for k in ('ACC','BA','Macro_F1','CE')})
                row.update({'recall_'+c:float(result['recall'][i]) for i,c in enumerate(CLASS_NAMES)})
                row.update(mean_vibration_scale=float(scales[:,0].mean()),mean_current_scale=float(scales[:,1].mean()))
                rows.append(row); matrices.append(result['confusion']); hashes[str(path.relative_to(out))]=sha(path)
    if len(list((out/'predictions').glob('*/*.npz')))!=2520:
        raise ValueError('Prediction count differs from protocol')
    return rows,np.stack(matrices),hashes

def summarize(rows):
    columns=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES],'mean_vibration_scale','mean_current_scale')
    perseed=[]; summary=[]; snrsummary=[]
    for method in METHODS:
        for endpoint in ('clean','noisy'):
            values=[]
            for seed in SEEDS:
                chosen=[r for r in rows if r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                if len(chosen)!=(4 if endpoint=='clean' else 80):
                    raise ValueError('Unbalanced endpoint aggregation')
                r=dict(method=method,seed=seed,endpoint=endpoint,**{k:float(np.mean([v[k] for v in chosen])) for k in columns})
                perseed.append(r); values.append(r)
            summary.append(dict(method=method,endpoint=endpoint,**{key:value for k in columns for key,value in
                ((k+'_mean',float(np.mean([v[k] for v in values]))),(k+'_SD',float(np.std([v[k] for v in values],ddof=1))))}))
        for snr in (None,*SNRS):
            vals=[{k:float(np.mean([r[k] for r in rows if r['method']==method and r['seed']==s and r['snr_db']==snr])) for k in columns} for s in SEEDS]
            snrsummary.append(dict(method=method,snr_db=snr,**{key:value for k in columns for key,value in
                ((k+'_mean',float(np.mean([v[k] for v in vals]))),(k+'_SD',float(np.std([v[k] for v in vals],ddof=1))))}))
    paired=[]; pairseeds=[]
    for endpoint in ('clean','noisy'):
        for comparator in [m for m in METHODS if m!='physical']:
            diffs={k:[] for k in ('BA','Macro_F1')}
            for seed in SEEDS:
                a=next(r for r in perseed if r['method']=='physical' and r['endpoint']==endpoint and r['seed']==seed)
                b=next(r for r in perseed if r['method']==comparator and r['endpoint']==endpoint and r['seed']==seed)
                delta={k:a[k]-b[k] for k in diffs}
                pairseeds.append(dict(comparison='physical_minus_'+comparator,endpoint=endpoint,seed=seed,**delta))
                for k in diffs:
                    diffs[k].append(delta[k])
            paired.append(dict(comparison='physical_minus_'+comparator,endpoint=endpoint,**{key:value for k,d in diffs.items() for key,value in
                ((k+'_difference_mean',float(np.mean(d))),(k+'_difference_SD',float(np.std(d,ddof=1))),(k+'_positive_seeds',int((np.array(d)>0).sum())))}))
    return perseed,summary,snrsummary,paired,pairseeds


def condition_summaries(rows):
    columns=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES])
    seeds=[]; summary=[]
    for condition in CONDITIONS:
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                values=[]
                for seed in SEEDS:
                    chosen=[r for r in rows if r['condition']==condition and r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                    assert len(chosen)==(1 if endpoint=='clean' else 20)
                    v=dict(condition=condition,method=method,seed=seed,endpoint=endpoint,**{k:float(np.mean([r[k] for r in chosen])) for k in columns})
                    seeds.append(v); values.append(v)
                summary.append(dict(condition=condition,method=method,endpoint=endpoint,**{key:value for k in columns for key,value in
                    ((k+'_mean',float(np.mean([v[k] for v in values]))),(k+'_SD',float(np.std([v[k] for v in values],ddof=1))))}))
    return seeds,summary


def generate(out,figures=True):
    rows,matrices,hashes=collect(out); perseed,summary,snrsummary,paired,pairseeds=summarize(rows)
    condition_seeds,condition_summary=condition_summaries(rows)
    folder=out/'report'; folder.mkdir(exist_ok=False)
    write_csv(folder/'condition_per_seed_endpoints.csv',condition_seeds)
    write_csv(folder/'condition_endpoints_summary.csv',condition_summary)
    for name,values in (('metrics_recomputed.csv',rows),('per_seed_endpoints.csv',perseed),('endpoints_summary.csv',summary),
                        ('SNR_summary.csv',snrsummary),('paired_differences.csv',paired),('per_seed_differences.csv',pairseeds)):
        write_csv(folder/name,values)
    save_npz(folder/'confusions.npz',confusion=matrices)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,endpoint in zip(axes,('clean','noisy')):
            values=[next(r for r in summary if r['method']==m and r['endpoint']==endpoint) for m in METHODS]
            ax.bar(range(6),[100*r['BA_mean'] for r in values],yerr=[100*r['BA_SD'] for r in values],capsize=3)
            ax.set_xticks(range(6),METHODS,rotation=25); ax.set_ylim(0,105); ax.set_ylabel('Balanced accuracy (%)'); ax.set_title(endpoint+' mixed conditions')
        fig.tight_layout(); fig.savefig(folder/'all_conditions_BA.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for method in METHODS:
            vals=[next(r for r in snrsummary if r['method']==method and r['snr_db']==s) for s in SNRS]
            ax.errorbar(SNRS,[100*r['BA_mean'] for r in vals],yerr=[100*r['BA_SD'] for r in vals],marker='o',capsize=3,label=method)
        ax.set_xlabel('Raw record vibration SNR (dB)'); ax.set_ylabel('Balanced accuracy (%)'); ax.set_ylim(0,105); ax.legend()
        fig.tight_layout(); fig.savefig(folder/'all_conditions_noise.png',dpi=160); plt.close(fig)
    table=['|方法|干净BA %|受扰BA %|干净Macro-F1 %|受扰Macro-F1 %|','|---|---:|---:|---:|---:|']
    for method in METHODS:
        clean,noisy=[next(r for r in summary if r['method']==method and r['endpoint']==e) for e in ('clean','noisy')]
        vals=[(clean,'BA'),(noisy,'BA'),(clean,'Macro_F1'),(noisy,'Macro_F1')]
        table.append('|'+method+'|'+'|'.join(f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_SD']:.2f}" for r,k in vals)+'|')
    text=['# PU_0四工况混合7∶2∶1实验','',
        '四个工况均进入训练、验证和测试。按工况与轴承分层，完整重复记录组不跨集合，窗口数11200/3200/1600。每工况2800/800/400窗，每类每工况700/200/100窗。',
        '沿用六组模型、五种子、Adam0.001、batch64及50轮末轮。每轮175更新、每模型8750更新，不能声称与旧2500更新计算预算相同。30个模型全部核验后统一2520条件预测。','',*table,'',
        '主结果对四工况等权，受扰再对四SNR和五噪声实例等权。误差为五个训练种子的样本标准差。逐工况、900rpm、各类召回及配对种子差在CSV中完整保存。',
        '三个集合共享全部工况和同8个轴承实体，属于已知工况混合的记录留出评价，不能证明新工况或新轴承泛化。历史数据与目标工况已曝光，继续按探索性研究解释。',
        '原始640文件去重636条记录。为同时保持记录排他及严格7∶2∶1，只在必要KA04记录调整取窗配额。未以重复窗口补数。所有定标只拟合11200训练窗。',
        'physical、late_concat 和 self_only 使用四谐波物理描述符。wrong 沿用旧替代阶次及旧计算式，和新版 physical 的差异涉及阶次及计算式，不能单独解释为正确阶次的贡献。',
        '同一记录内窗口相关、共享噪声及五种子规模限制统计外推。若接近100%，按上限效应报告，不追加调参。有限零梯度或常量预测保留，不自动重训。','']
    for r in paired:
        text.append(f"- {r['endpoint']} {r['comparison']}，BA差{100*r['BA_difference_mean']:+.4f}个百分点，SD {100*r['BA_difference_SD']:.4f}，正差{r['BA_positive_seeds']}/5；Macro-F1差{100*r['Macro_F1_difference_mean']:+.4f}个百分点。")
    (folder/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(folder/'verification.json',dict(prediction_cases=2520,confusion_cells=40320,prediction_sha256=hashes,
        report_files_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    return summary,paired
