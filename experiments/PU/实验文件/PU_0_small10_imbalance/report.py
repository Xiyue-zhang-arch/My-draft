"""Recompute all outcomes and protocol contrasts exclusively from saved predictions."""
import torch
import json
from pathlib import Path
import numpy as np
from common import DATA,PROTOCOLS,METHODS,SEEDS,CONDITIONS,CLASS_NAMES,models,cases,descriptor_profile,metrics,sha,write_json,write_csv,save_npz
from report_helpers import summarize,condition_summaries

def collect(out):
    state=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if not state['decisions_locked'] or not state['weights_verified'] or state['prediction_cases']!=7560:
        raise ValueError('Complete barrier and all 7560 predictions required')
    if [{k:r[k] for k in ('model_id','protocol','seed','method')} for r in state['completed_models']]!=models():
        raise ValueError('Exact 90 model plan required')
    descriptions=json.loads((out/'test_cases.json').read_text(encoding='utf-8'))
    if len(descriptions)!=84 or [r['case_id'] for r in descriptions]!=[c['case_id'] for c in cases()]:
        raise ValueError('Case plan changed')
    windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8')); split=np.load(DATA/'split_indices.npz')
    rows=[]; matrices=[]; hashes={}
    for spec in models():
        protocol=spec['protocol']; profile=descriptor_profile(spec['method'])
        for case,description in zip(cases(),descriptions):
            path=out/'predictions'/spec['model_id']/(case['case_id']+'.npz')
            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids])
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['y'],y) or not np.array_equal(z['window_index'],ids):
                    raise ValueError('Saved labels or window identities changed')
                if any(str(z[k])!=str(v) for k,v in spec.items()) or str(z['case_id'])!=case['case_id'] or str(z['target'])!=case['target']:
                    raise ValueError('Model/protocol/condition identity changed')
                if str(z['waveform_input_sha256'])!=description['waveform_input_sha256'][protocol] or str(z['descriptor_input_sha256'])!=description['descriptor_input_sha256'][protocol][profile]:
                    raise ValueError('Paired input hashes changed')
                result,p=metrics(z['logits'],y)
                if not np.allclose(p,z['probability'],rtol=0,atol=1e-12):
                    raise ValueError('Probability mismatch')
                scales=z['scale_mean']
                if scales.shape!=(40,2) or not np.isfinite(scales).all() or scales.min()<.75-1e-6 or scales.max()>1.25+1e-6:
                    raise ValueError('Invalid bounded scale statistics')
                if spec['method'] in ('equal','late_concat') and not np.array_equal(scales,np.ones_like(scales)):
                    raise ValueError('Unmodulated branch changed scales')
                row=dict(**spec,case_id=case['case_id'],condition=case['target'],endpoint='clean' if case['noise_seed'] is None else 'noisy',
                    snr_db=case['snr_db'],noise_seed=case['noise_seed'],**{k:result[k] for k in ('ACC','BA','Macro_F1','CE')})
                row.update({'recall_'+c:float(result['recall'][k]) for k,c in enumerate(CLASS_NAMES)})
                row.update(mean_vibration_scale=float(scales[:,0].mean()),mean_current_scale=float(scales[:,1].mean()))
                rows.append(row); matrices.append(result['confusion']); hashes[str(path.relative_to(out))]=sha(path)
    if len(list((out/'predictions').glob('*/*.npz')))!=7560:
        raise ValueError('Incomplete prediction files')
    return rows,np.stack(matrices),hashes

def generate(out,figures=True):
    rows,matrices,hashes=collect(out); allseeds=[]; summaries=[]; snrs=[]; pairs=[]; pairseeds=[]; condseeds=[]; conds=[]
    for protocol in PROTOCOLS:
        subset=[r for r in rows if r['protocol']==protocol]
        perseed,summary,snr,paired,paired_seed=summarize(subset); cs,cc=condition_summaries(subset)
        for destination,source in ((allseeds,perseed),(summaries,summary),(snrs,snr),(pairs,paired),(pairseeds,paired_seed),(condseeds,cs),(conds,cc)):
            destination.extend(dict(protocol=protocol,**r) for r in source)
    effects=[]; effectseeds=[]
    for a,b in (('imbalanced','matched_balanced'),('imbalanced','balanced10'),('matched_balanced','balanced10')):
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                diffs={k:[] for k in ('BA','Macro_F1')}
                for seed in SEEDS:
                    aa=next(r for r in allseeds if r['protocol']==a and r['method']==method and r['endpoint']==endpoint and r['seed']==seed)
                    bb=next(r for r in allseeds if r['protocol']==b and r['method']==method and r['endpoint']==endpoint and r['seed']==seed)
                    delta={k:aa[k]-bb[k] for k in diffs}
                    effectseeds.append(dict(comparison=a+'_minus_'+b,method=method,endpoint=endpoint,seed=seed,**delta))
                    for k in diffs:
                        diffs[k].append(delta[k])
                effects.append(dict(comparison=a+'_minus_'+b,method=method,endpoint=endpoint,**{key:value for k,d in diffs.items() for key,value in
                    ((k+'_difference_mean',float(np.mean(d))),(k+'_difference_SD',float(np.std(d,ddof=1))),(k+'_positive_seeds',int((np.array(d)>0).sum())))}))
    report=out/'report'; report.mkdir(exist_ok=False)
    for name,values in (('metrics_recomputed.csv',rows),('per_seed_endpoints.csv',allseeds),('endpoints_summary.csv',summaries),
        ('SNR_summary.csv',snrs),('paired_differences.csv',pairs),('per_seed_differences.csv',pairseeds),
        ('condition_per_seed_endpoints.csv',condseeds),('condition_endpoints_summary.csv',conds),
        ('protocol_effects.csv',effects),('protocol_effects_per_seed.csv',effectseeds)):
        write_csv(report/name,values)
    save_npz(report/'confusions.npz',confusion=matrices)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for endpoint in ('clean','noisy'):
            fig,axes=plt.subplots(1,3,figsize=(16,4.5))
            for ax,protocol in zip(axes,PROTOCOLS):
                values=[next(r for r in summaries if r['protocol']==protocol and r['method']==m and r['endpoint']==endpoint) for m in METHODS]
                ax.bar(range(6),[100*r['BA_mean'] for r in values],yerr=[100*r['BA_SD'] for r in values],capsize=3)
                ax.set_xticks(range(6),METHODS,rotation=30); ax.set_ylim(0,105); ax.set_title(protocol+' '+endpoint); ax.set_ylabel('Balanced accuracy (%)')
            fig.tight_layout(); fig.savefig(report/(endpoint+'_BA.png'),dpi=150); plt.close(fig)
    text=['# 10%小样本与训练类别不平衡结果','',
        'balanced10为1120训练/320验证/160测试窗；imbalanced为280/28/28/28训练窗；matched_balanced为每类91训练窗，后两项总量364。全部协议共用均衡验证和测试窗口，继承父实验记录排他划分。',
        '六方法、五种子、三数据协议共90新模型，固定50轮末轮。1120窗协议每模型900更新，两个364窗协议各300更新。全部90模型核验后统一7560条件预测。','']
    for protocol in PROTOCOLS:
        text+=['## '+protocol,'','|方法|干净BA %|受扰BA %|干净Macro-F1 %|受扰Macro-F1 %|','|---|---:|---:|---:|---:|']
        for method in METHODS:
            clean,noisy=[next(r for r in summaries if r['protocol']==protocol and r['method']==method and r['endpoint']==e) for e in ('clean','noisy')]
            values=[(clean,'BA'),(noisy,'BA'),(clean,'Macro_F1'),(noisy,'Macro_F1')]
            text.append('|'+method+'|'+'|'.join(f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_SD']:.2f}" for r,k in values)+'|')
        text.append('')
    text+=['均值对四工况等权，受扰再对四SNR和五噪声实例等权，误差为五训练种子样本SD。训练为普通无权重交叉熵，无过采样或类别权重。',
        '单工况每类仅10测试窗，召回分辨率10个百分点；同记录窗口相关，训练种子SD不代表数据抽样或轴承总体不确定性。只有一次固定数据子集抽样，不能声称对任意小样本划分稳定。',
        '等量均衡对照控制训练总量和更新次数，但记录覆盖、窗口身份和各自训练定标可能不同，不能唯一解释为类别先验的因果效应。全部工况和同8轴承在各集合出现，历史曝光，不能证明新工况或新轴承泛化。',
        '有限零梯度、常量预测或少数类召回0均保留，不重训以筛选结果。不能将描述符或通道权重等同于已验证的信息丰富度。',
        'physical、late_concat 和 self_only 使用四谐波物理描述符。wrong 沿用旧替代阶次及旧计算式。两者的差异同时涉及阶次及计算式。',
        '逐协议物理对照差、逐种子差及数据协议效应见CSV。结果完成后冻结，不按测试追加范围。','']
    (report/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    write_json(report/'verification.json',dict(prediction_cases=7560,confusion_cells=120960,prediction_sha256=hashes,
        report_files_sha256={p.name:sha(p) for p in report.iterdir() if p.is_file()}))
    return summaries,pairs,effects
