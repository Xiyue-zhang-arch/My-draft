"""汇总从头训练的排除实验，不读取原实验模型、预测或筛选主结果。"""
from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'shared'))
from experiment_paths import validate_output_path
from confusion_gallery import panel,CONDITIONS,SUPPORT


def report(out):
    out=validate_output_path(out,existing=True)
    complexity=pd.read_csv(out/'complexity.csv');original=pd.read_csv(out/'metrics.csv')
    details=[];pooled={}
    for row in complexity.to_dict('records'):
        folder=out/row['protocol']/row['method']/f"ratio_{round(row['ratio']*100):03d}_subset_{row['subset']}_seed_{row['seed']}"
        with np.load(folder/'predictions.npz',allow_pickle=False) as arrays:
            labels=arrays['labels'];np.testing.assert_array_equal(np.bincount(labels,minlength=4),SUPPORT)
            for key in arrays.files:
                if '_n' not in key:continue
                condition,ns=key.rsplit('_n',1)
                if condition not in dict(CONDITIONS):continue
                probabilities=arrays[key];predicted=probabilities.argmax(1)
                counts=confusion_matrix(labels,predicted,labels=range(4));recall=counts.diagonal()/counts.sum(1)
                details.append({k:row[k] for k in ['protocol','method','ratio','subset','seed']}|dict(
                    condition=condition,noise_seed=int(ns),acc=accuracy_score(labels,predicted),
                    f1_macro=f1_score(labels,predicted,labels=range(4),average='macro',zero_division=0),
                    balanced_acc=recall.mean(),**dict(zip(['recall_N','recall_IR','recall_OR','recall_B'],recall))))
                group=(row['protocol'],row['method'],round(row['ratio']*100),condition)
                pooled[group]=pooled.get(group,np.zeros((4,4),dtype=np.int64))+counts
    detailed=pd.DataFrame(details)
    keys=['protocol','method','ratio','subset','seed','condition','noise_seed']
    check=detailed.merge(original,on=keys,suffixes=('_check','_saved'),validate='one_to_one')
    assert len(check)==len(original)
    for metric in ['acc','f1_macro']:np.testing.assert_allclose(check[metric+'_check'],check[metric+'_saved'],atol=1e-12,rtol=0)
    detailed.to_csv(out/'class_metrics.csv',index=False)
    columns=['acc','f1_macro','balanced_acc','recall_N','recall_IR','recall_OR','recall_B']
    per_model=detailed.groupby(keys[:-1],as_index=False)[columns].mean()
    summary=per_model.groupby(['protocol','method','ratio','condition'])[columns].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(x for x in col if x) if isinstance(col,tuple) else col for col in summary.columns]
    summary.to_csv(out/'summary.csv',index=False)
    np.savez_compressed(out/'confusion_counts.npz',**{f'{p}_{m}_{r:03d}_{c}':v for (p,m,r,c),v in pooled.items()})
    gallery=out/'figures/all_conditions';gallery.mkdir(parents=True,exist_ok=False)
    images=[]
    for protocol,models in complexity.groupby('protocol',sort=False):
        groups=models[['method','ratio']].drop_duplicates()
        if protocol=='clean_only':
            fig,axes=plt.subplots(2,3,figsize=(13,8));axes=list(axes.flat)
            for ax,row in zip(axes,groups.itertuples()):
                ratio=round(row.ratio*100);panel(ax,pooled[(protocol,row.method,ratio,'clean')],f'{row.method}; train {ratio}%')
            for ax in axes[len(groups):]:ax.axis('off')
            fig.suptitle('From-scratch fixed 212-D; clean-only; OR014_3 excluded')
            fig.tight_layout(rect=[0,.04,1,.95]);fig.text(.5,.015,'Rows sum to 100%; brackets show 180-window support, not noise/model repetitions',ha='center',fontsize=9)
            filename='clean_all_budgets.png';fig.savefig(gallery/filename,dpi=180);plt.close(fig);images.append(filename)
        else:
            for row in groups.itertuples():
                ratio=round(row.ratio*100);fig,axes=plt.subplots(2,3,figsize=(13,8))
                for ax,(condition,title) in zip(axes.flat,CONDITIONS):panel(ax,pooled[(protocol,row.method,ratio,condition)],title)
                fig.suptitle(f'From-scratch fixed 212-D; {row.method}; AWGN train {ratio}%; OR014_3 excluded')
                fig.tight_layout(rect=[0,.04,1,.95]);fig.text(.5,.015,'Rows sum to 100%; noisy counts pool 5 realizations per window and model; no probability ensemble',ha='center',fontsize=9)
                filename=f'awgn_{row.method}_{ratio:03d}_all_conditions.png';fig.savefig(gallery/filename,dpi=180);plt.close(fig);images.append(filename)
    lines=['# 排除OR014_3后从头训练的固定212维实验','',
        f'本轮从头训练{len(complexity)}个模型。没有使用旧模型权重、预训练或原预测筛选来代替训练。原完整200窗口实验与其比较模型保持冻结。','',
        '训练0/1HP共287窗口，验证2HP共200窗口保持不变，仅从3HP测试集中排除OR014_3的20窗口。测试180窗口，Normal20、IR60、OR40、B60。OR014_0/1/2保留。','',
        '融合模型包含全样本及各类10%、20%、50%训练窗口，分别287、29、58、144。全样本3种子，小样本每比例3子集乘3种子。单数据分支依旧只做全样本3种子。干净与非配对AWGN两套协议各33模型。','',
        'AWGN每次窗口出现时独立50%保持干净，否则连续均匀采样U[-4,4]dB加性高斯白噪声。每批32观测、一次CE，双分支使用同一观测。纯干净组训练、验证及测试均不人工加噪。验证选模规则不变。全部模型完成验证选模后统一测试。','',
        '测试噪声先按原200窗口顺序生成，再只保留180窗口，因此其余窗口的噪声方向与原协议一致。每档5实例。不是重新挑选有利噪声，不将重复观测当作独立窗口。','',
        '## 测试结果','',
        '先在每个模型内平均噪声重复，再报告模型间均值和样本标准差。标准差不是独立轴承泛化的置信区间。小样本仍使用200个带标签验证窗口。','']
    for protocol in complexity.protocol.unique():
        conditions=['clean'] if protocol=='clean_only' else [k for k,_ in CONDITIONS]
        lines += [f'### {protocol}，窗口ACC（%）','', '|模型|训练比例|'+ '|'.join(conditions)+'|','|---|---:|'+ '|'.join(['---:']*len(conditions))+'|']
        for row in complexity[complexity.protocol==protocol][['method','ratio']].drop_duplicates().itertuples():
            cells=[]
            for condition in conditions:
                value=summary[(summary.protocol==protocol)&(summary.method==row.method)&(summary.ratio==row.ratio)&(summary.condition==condition)].iloc[0]
                cells.append(f'{100*value.acc_mean:.2f} ± {100*value.acc_std:.2f}')
            lines.append(f'|{row.method}|{row.ratio:.0%}|'+ '|'.join(cells)+'|')
        lines.append('')
    lines += ['均衡ACC、宏F1和四类召回率见summary.csv，逐次结果见metrics.csv与class_metrics.csv。比例混淆矩阵按真实类别逐行归一化，括号为原始窗口支持数。原始累计计数见confusion_counts.npz。','',
        '## 计算复杂度','', '|协议|模型|训练比例|推理ms/样本|训练s/轮|峰值MiB|参数K|训练子集拟合s|','|---|---|---:|---:|---:|---:|---:|---:|']
    cs=complexity.groupby(['protocol','method','ratio'],as_index=False)[['latency_ms','train_seconds_per_epoch','peak_gpu_mib','parameters','fit_seconds']].mean()
    cs.to_csv(out/'complexity_summary.csv',index=False)
    for row in cs.itertuples():lines.append(f'|{row.protocol}|{row.method}|{row.ratio:.0%}|{row.latency_ms:.3f}|{row.train_seconds_per_epoch:.3f}|{row.peak_gpu_mib:.1f}|{row.parameters/1000:.3f}|{row.fit_seconds:.3f}|')
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    lines += ['',f"硬件{cfg['hardware']}，精度{cfg['precision']}。推理批量1，10次预热、50次重复，包含归一化、在线物理特征及网络，排除磁盘和H2D。训练批量32，计时包含在线提取、增强和反向，排除验证。训练子集标准化拟合另列。峰值为PyTorch已分配MiB。",
        '本轮训练和推理均在线提取物理特征。旧纯干净实验缓存物理特征，旧训练时间还排除了预热轮，不能仅按秒/轮直接比较端到端训练效率。','',
        '## 混淆矩阵','']
    for filename in images:lines += [f'![全部条件](figures/all_conditions/{filename})','']
    lines += ['## 解释边界','',
        'OR014_3的排除依据来自既有测试结果，因此即使重新训练，仍属于事后测试组成敏感性实验，不能替代完整200窗口主结果，不能恢复独立确认性质。训练和验证信息没有改变，指标变化不能单独解释为模型能力改善。',
        '融合与单数据分支同时改变输入信息、参数量和融合结构，不能单独证明物理因果机理。AWGN结论限于当前合成白噪声与SNR范围。','', '[固定协议](PROTOCOL.md)']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (gallery/'README.md').write_text('# 从头训练排除实验的混淆矩阵\n\n'+ '\n\n'.join(f'![全部条件]({name})' for name in images),encoding='utf-8')
    (out/'README.md').write_text('# 排除OR014_3后从头训练\n\n[正式报告](REPORT.md) · [全部混淆矩阵](figures/all_conditions/README.md)\n\n模型和预测均由本轮从头训练生成，路径为协议/模型/训练比例与种子。当前报告仅汇总本轮模型。\n',encoding='utf-8')
    print(summary[['protocol','method','ratio','condition','acc_mean']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True,type=Path)
    report(parser.parse_args().results)
