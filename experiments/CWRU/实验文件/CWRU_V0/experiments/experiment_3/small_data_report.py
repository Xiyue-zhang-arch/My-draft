"""实验3小样本配对对照，旧融合预测只读复用。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score
from window_report import CONDITIONS,KEYS,METRICS,panel,plt,validate_output_path,markdown_table


def report(directory,source):
    out=validate_output_path(directory,existing=True);source=Path(source)
    w=pd.read_csv(out/'windows.csv');test=w[w.role.eq('test')];support=np.bincount(test.label,minlength=4)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));complexity=pd.read_csv(out/'complexity.csv')
    assert len(complexity)==54 and cfg['evaluated_models']==54
    details=[];pooled={}
    for row in complexity.to_dict('records'):
        folder=out/row['protocol']/row['method']/f"ratio_{round(row['ratio']*100):03d}_subset_{row['subset']}_seed_{row['seed']}"
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['labels'],test.label);np.testing.assert_array_equal(z['window_indices'],test.window_index)
            for condition,_ in (CONDITIONS[:1] if row['protocol']=='clean_only' else CONDITIONS):
                total=np.zeros((4,4),dtype=np.int64)
                names=[k for k in z.files if k.startswith(condition+'_n')];assert len(names)==(1 if condition=='clean' else 5)
                for name in names:
                    probabilities=z[name];assert probabilities.shape==(100,4) and np.isfinite(probabilities).all()
                    np.testing.assert_allclose(probabilities.sum(1),1,atol=1e-5)
                    counts=confusion_matrix(z['labels'],probabilities.argmax(1),labels=range(4));recall=counts.diagonal()/support
                    details.append({k:row[k] for k in KEYS[:5]}|dict(condition=condition,noise_seed=int(name.rsplit('_n',1)[1]),
                        acc=float(np.trace(counts)/100),balanced_acc=float(recall.mean()),
                        f1_macro=f1_score(z['labels'],probabilities.argmax(1),labels=range(4),average='macro',zero_division=0),**dict(zip(METRICS[3:],recall))))
                    total+=counts
                np.testing.assert_array_equal(total,z[condition+'_confusion_counts'])
                key=f"{row['protocol']}_{row['method']}_{round(row['ratio']*100):03d}_{condition}"
                pooled[key]=pooled.get(key,np.zeros((4,4),dtype=np.int64))+total
    detailed=pd.DataFrame(details);saved=pd.read_csv(out/'metrics.csv')
    merged=detailed.merge(saved,on=KEYS,validate='one_to_one',suffixes=('_check','_saved'))
    assert len(detailed)==len(merged)==len(saved)==729
    for metric in ['acc','f1_macro']:np.testing.assert_allclose(merged[metric+'_check'],merged[metric+'_saved'],atol=1e-12,rtol=0)
    detailed.to_csv(out/'class_metrics.csv',index=False)
    old=pd.read_csv(source/'class_metrics.csv');combined=pd.concat([old,detailed],ignore_index=True)
    assert len(combined)==1620 and not combined.duplicated(KEYS).any()
    combined.to_csv(out/'combined_class_metrics.csv',index=False)
    per_model=combined.groupby(KEYS[:-1],as_index=False)[METRICS].mean();per_model.to_csv(out/'combined_per_model_metrics.csv',index=False)
    summary=per_model.groupby(['protocol','method','ratio','condition'])[METRICS].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(k for k in col if k) for col in summary.columns];assert len(summary)==56
    summary.to_csv(out/'combined_summary.csv',index=False)
    paired=per_model.pivot(index=['protocol','ratio','subset','seed','condition'],columns='method',values=METRICS)
    delta=pd.DataFrame({m+'_delta':paired[(m,'fusion')]-paired[(m,'data_only')] for m in METRICS}).reset_index()
    assert len(delta)==210
    delta.to_csv(out/'paired_model_differences.csv',index=False)
    columns=[m+'_delta' for m in METRICS]
    delta.groupby(['protocol','ratio','subset','condition'],as_index=False)[columns].mean().to_csv(out/'paired_subset_means.csv',index=False)
    paired_summary=delta.groupby(['protocol','ratio','condition'])[columns].agg(['mean','std','count']).reset_index()
    paired_summary.columns=['_'.join(k for k in col if k) for col in paired_summary.columns]
    paired_summary.to_csv(out/'paired_summary.csv',index=False)
    with np.load(source/'confusion_counts.npz') as z:
        for key in z.files:
            assert key not in pooled;pooled[key]=z[key]
    assert len(pooled)==56;np.savez_compressed(out/'combined_confusion_counts.npz',**pooled)
    old_complexity=pd.read_csv(source/'complexity.csv');all_complexity=pd.concat([old_complexity.assign(run='original'),complexity.assign(run='supplement')],ignore_index=True)
    all_complexity.to_csv(out/'combined_complexity.csv',index=False)
    cs=all_complexity.groupby(['protocol','method','ratio','run'],as_index=False)[['parameters','latency_ms','train_seconds_per_epoch','total_train_seconds','peak_gpu_mib','validation_forward_seconds']].mean()
    cs.to_csv(out/'complexity_summary.csv',index=False)
    figures=out/'figures';figures.mkdir();images=[]
    for protocol in ['clean_only','awgn_mixed']:
        for method in ['fusion','data_only']:
            if protocol=='clean_only':
                fig,axes=plt.subplots(2,2,figsize=(10,8))
                for ax,ratio in zip(axes.flat,[.1,.2,.5,1.]):panel(ax,pooled[f'{protocol}_{method}_{round(100*ratio):03d}_clean'],f'{method}; {ratio:.0%}',support)
                fig.suptitle(f'Experiment3 matched subsets; {protocol}; {method}');fig.tight_layout(rect=[0,.03,1,.95])
                name=f'{protocol}_{method}_all_budgets.png';fig.savefig(figures/name,dpi=160);plt.close(fig);images.append(name)
            else:
                for ratio in [.1,.2,.5,1.]:
                    fig,axes=plt.subplots(2,3,figsize=(13,8))
                    for ax,(condition,title) in zip(axes.flat,CONDITIONS):panel(ax,pooled[f'{protocol}_{method}_{round(100*ratio):03d}_{condition}'],title,support)
                    fig.suptitle(f'Experiment3 matched subsets; {method}; AWGN {ratio:.0%}');fig.tight_layout(rect=[0,.04,1,.95])
                    fig.text(.5,.015,'Row percentages; supports are100original test windows, not seed/noise repeats',ha='center',fontsize=9)
                    name=f'{protocol}_{method}_{round(100*ratio):03d}.png';fig.savefig(figures/name,dpi=160);plt.close(fig);images.append(name)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(protocol,condition,title) in zip(axes,[('clean_only','clean','Clean training / clean test'),('awgn_mixed','white_-4','AWGN training / -4 dB test')]):
        for method,g in summary[(summary.protocol==protocol)&(summary.condition==condition)].groupby('method'):
            g=g.sort_values('ratio');ax.errorbar(100*g.ratio,100*g.acc_mean,yerr=100*g.acc_std,marker='o',capsize=3,label=method)
        ax.set(xlabel='Training subset (%)',ylabel='Window ACC (%)',title=title);ax.legend()
    fig.tight_layout();fig.savefig(figures/'matched_accuracy_overview.png',dpi=170);plt.close(fig);images.insert(0,'matched_accuracy_overview.png')
    usage=[];budgets=[]
    base=pd.read_csv(source/'dataset_file_usage.csv')
    (out/'dataset_file_usage.csv').write_bytes((source/'dataset_file_usage.csv').read_bytes())
    for ratio in [.1,.2,.5]:
        for subset in [4101,4102,4103]:
            ref=out/'clean_only'/'data_only'/f'ratio_{round(100*ratio):03d}_subset_{subset}_seed_1706'/'selected_windows.csv'
            selected=pd.read_csv(ref);budgets.append((ratio,subset,len(selected)))
            counts=selected.record_id.value_counts()
            for row in base.to_dict('records'):
                usage.append(dict(ratio=ratio,subset_seed=subset,record_id=row['record_id'],
                    raw_file=row['raw_file'],raw_relative_path=row['raw_relative_path'],
                    selected_DE_variable=row['selected_DE_variable'],used_training_windows=int(counts.get(row['record_id'],0)),
                    new_models_sharing_subset=6,representative_selected_windows=ref.relative_to(out).as_posix()))
    pd.DataFrame(usage).to_csv(out/'training_subset_file_usage.csv',index=False,encoding='utf-8-sig')
    (out/'DATA_USAGE.md').write_text('# 实验3补充数据单分支的数据使用说明\n\n完全沿用原实验3的39记录667完整窗、候选训练467、验证100、测试100，排除OR014_3。测试正常10、内圈29、外圈30、滚动体31，训练测试共享38记录。原始数据只读。\n\n[集合及40个原始MAT逐文件用量](dataset_file_usage.csv) · [实际小样本逐文件用量](training_subset_file_usage.csv)。小样本48/95/235窗，各档3子集，各子集两协议3种子共6个新模型复用；验证100不随比例缩减。每个模型保存selected_windows.csv，逐行等于原融合对应子集。\n\n未复制旧权重或旧概率文件。原融合及全样本结果从原实验3只读引用，仅新增单分支54模型，不将模型、种子或噪声重复计为新增独立样本。\n',encoding='utf-8')
    lines=['# 实验3 数据单分支小样本补充与配对比较','','已完成54个随机初始化数据单分支模型，729组新增逐次指标。原66模型及891组指标只读复用，合并展示120模型、1620组逐次指标、56组汇总混淆计数。没有重新训练或重新测试原融合模型。','',
        '同预算对照匹配原始训练子集、100验证窗、100测试窗、抽样与噪声规则、优化器、最多600更新和早停选模规则。实际停止更新数可因模型不同而不同，不声称实际训练更新总量完全一致。原结果曝光后才补充对照，不属于新的独立确认性评价。','',
        '## 窗口准确率','','每档小样本3子集×3训练种子；全样本3训练种子。先平均模型内5噪声实例，再报告组合均值与样本标准差。9组合不是9个独立数据集，标准差不是独立轴承置信区间。','',
        '|协议|比例|模型|干净ACC (%)|−4 dB ACC (%)|','|---|---:|---|---:|---:|']
    for protocol in ['clean_only','awgn_mixed']:
        for ratio in [.1,.2,.5,1.]:
            for method in ['fusion','data_only']:
                g=summary[(summary.protocol==protocol)&(summary.ratio==ratio)&(summary.method==method)]
                c=g[g.condition.eq('clean')].iloc[0];n=g[g.condition.eq('white_-4')]
                noisy='未评价' if n.empty else f'{100*n.iloc[0].acc_mean:.2f} ± {100*n.iloc[0].acc_std:.2f}'
                lines.append(f'|{protocol}|{ratio:.0%}|{method}|{100*c.acc_mean:.2f} ± {100*c.acc_std:.2f}|{noisy}|')
    lines+=['','完整噪声条件、均衡准确率、宏F1与逐类召回见combined_summary.csv。配对差值为融合减去单分支，见paired_model_differences.csv，子集内种子平均见paired_subset_means.csv，不筛有利组合。','',
        '## 数据与解释边界','','[逐文件用量及实际训练子集](DATA_USAGE.md)。各档额外使用100标注验证窗，不能将48训练窗称作全部标注预算48。训练测试共享记录，不作为跨负载、跨记录或独立轴承泛化证据。原始OR014_3事后排除及历史测试曝光仍存在。','',
        '新增同预算对照使得比较两模型在当前固定划分下的样本预算表现成为可能，但不自动证明物理特征的因果机制。两模型同时改变信息、结构和参数量，不能推广到其他数据集、故障试件和现场噪声。','',
        '## 计算开销','',f"新运行硬件{cfg['hardware']}，PyTorch {cfg['torch']}，编码器BF16其余FP32。推理batch1、预热10重复50，归一化加网络，排除磁盘/H2D；数据单分支不提取物理特征。训练batch32，训练总/轮均时间、验证前向、峰值allocated显存分列。新旧运行时间不混为同次独立计时。",'',
        '\n'.join(markdown_table(cs.round(4))),'','[原实验3报告](../experiment_3/REPORT.md) · [固定补充协议](PROTOCOL.md)','','## 图表','']
    for name in images:lines += [f'![{name}](figures/{name})','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary[summary.condition.isin(['clean','white_-4'])][['protocol','method','ratio','condition','acc_mean','balanced_acc_mean']].to_string(index=False),flush=True)
