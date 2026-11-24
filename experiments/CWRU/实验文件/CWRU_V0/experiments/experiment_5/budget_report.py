"""新窗口划分实验的指标、动态支持数混淆矩阵与证据边界报告。"""

from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from experiment_paths import validate_output_path

CONDITIONS=[('clean','Clean'),('white_-4','-4 dB'),('white_-2','-2 dB'),
            ('white_+0','0 dB'),('white_+2','+2 dB'),('white_+4','+4 dB')]
LABELS=['N','IR','OR','B']
KEYS=['protocol','method','ratio','subset','seed','condition','noise_seed']
METRICS=['acc','balanced_acc','f1_macro','recall_N','recall_IR','recall_OR','recall_B']


def panel(ax,counts,title,support):
    fraction=counts/counts.sum(1,keepdims=True)
    ax.imshow(fraction*100,cmap='Blues',vmin=0,vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{100*fraction[i,j]:.1f}%',ha='center',va='center',
                    color='white' if fraction[i,j]>.55 else 'black',fontsize=9)
    ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,
           yticklabels=[f'{label} ({n})' for label,n in zip(LABELS,support)],
           xlabel='Predicted',ylabel='True',title=f'{title}; ACC {100*np.trace(counts)/counts.sum():.2f}%')


def markdown_table(frame):
    frame=frame.reset_index()
    return ['|'+'|'.join(map(str,frame.columns))+'|',
            '|'+'|'.join(['---']*len(frame.columns))+'|']+[
            '|'+'|'.join(map(str,row))+'|' for row in frame.itertuples(index=False,name=None)]


def report(directory):
    out=validate_output_path(directory,existing=True)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    windows=pd.read_csv(out/'windows.csv')
    complexity=pd.read_csv(out/'complexity.csv');saved=pd.read_csv(out/'metrics.csv')
    assert len(complexity)==cfg['planned_models']==cfg['trained_models']==cfg['evaluated_models']
    test=windows.role.eq('test').to_numpy()
    support=np.bincount(windows.loc[test,'label'],minlength=4)
    np.testing.assert_array_equal(support,cfg['test_support'])
    assert support.sum()==cfg['test_windows']
    details=[];pooled={}
    for row in complexity.to_dict('records'):
        folder=out/row['protocol']/row['method']/f"ratio_{round(row['ratio']*100):03d}_subset_{row['subset']}_seed_{row['seed']}"
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            labels=z['labels'];np.testing.assert_array_equal(labels,windows.loc[test,'label'])
            np.testing.assert_array_equal(z['window_indices'],windows.loc[test,'window_index'])
            for condition in (['clean'] if row['protocol']=='clean_only' else [x for x,_ in CONDITIONS]):
                counted=np.zeros((4,4),dtype=np.int64)
                for key in [k for k in z.files if k.startswith(condition+'_n')]:
                    p=z[key];assert p.shape==(len(labels),4) and np.isfinite(p).all()
                    np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
                    predicted=p.argmax(1);counts=confusion_matrix(labels,predicted,labels=range(4))
                    recall=counts.diagonal()/counts.sum(1)
                    details.append({k:row[k] for k in KEYS[:5]}|dict(condition=condition,
                        noise_seed=int(key.rsplit('_n',1)[1]),acc=accuracy_score(labels,predicted),
                        balanced_acc=recall.mean(),f1_macro=f1_score(labels,predicted,labels=range(4),average='macro',zero_division=0),
                        **dict(zip(METRICS[3:],recall))))
                    counted+=counts
                np.testing.assert_array_equal(counted,z[condition+'_confusion_counts'])
                group=(row['protocol'],row['method'],round(row['ratio']*100),condition)
                pooled[group]=pooled.get(group,np.zeros((4,4),dtype=np.int64))+counted
    detailed=pd.DataFrame(details)
    checked=detailed.merge(saved,on=KEYS,suffixes=('_check','_saved'),validate='one_to_one')
    assert len(checked)==len(saved)==len(detailed)==cfg['prediction_groups']
    for metric in ['acc','f1_macro']:
        np.testing.assert_allclose(checked[metric+'_check'],checked[metric+'_saved'],atol=1e-12,rtol=0)
    detailed.to_csv(out/'class_metrics.csv',index=False)
    per_model=detailed.groupby(KEYS[:-1],as_index=False)[METRICS].mean()
    per_model.to_csv(out/'per_model_metrics.csv',index=False)
    summary=per_model.groupby(['protocol','method','ratio','condition'])[METRICS].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(x for x in col if x) for col in summary.columns]
    summary.to_csv(out/'summary.csv',index=False)
    np.savez_compressed(out/'confusion_counts.npz',**{f'{p}_{m}_{r:03d}_{c}':v for (p,m,r,c),v in pooled.items()})
    columns=['latency_ms','train_seconds_per_epoch','peak_gpu_mib','parameters','fit_seconds','total_train_seconds','validation_forward_seconds']
    cs=complexity.groupby(['protocol','method','ratio'],as_index=False)[columns].mean()
    cs.to_csv(out/'complexity_summary.csv',index=False)
    figures=out/'figures';figures.mkdir(exist_ok=False);images=[]
    for protocol in complexity.protocol.unique():
        groups=complexity[complexity.protocol.eq(protocol)][['method','ratio']].drop_duplicates()
        if protocol=='clean_only':
            fig,axes=plt.subplots(2,3,figsize=(13,8));axes=list(axes.flat)
            for ax,row in zip(axes,groups.itertuples()):
                panel(ax,pooled[(protocol,row.method,round(row.ratio*100),'clean')],f'{row.method}; budget {row.ratio:.0%}',support)
            for ax in axes[len(groups):]:ax.axis('off')
            fig.suptitle('Window-random split within recordings; clean-only; fixed 212-D')
            fig.tight_layout(rect=[0,.04,1,.95]);fig.text(.5,.015,'Rows sum to 100%; brackets show original test window support, not repeated predictions',ha='center',fontsize=9)
            name='clean_all_budgets.png';fig.savefig(figures/name,dpi=180);plt.close(fig);images.append(name)
        else:
            for row in groups.itertuples():
                fig,axes=plt.subplots(2,3,figsize=(13,8))
                for ax,(condition,title) in zip(axes.flat,CONDITIONS):panel(ax,pooled[(protocol,row.method,round(row.ratio*100),condition)],title,support)
                fig.suptitle(f'Window-random split; {row.method}; AWGN budget {row.ratio:.0%}; fixed 212-D')
                fig.tight_layout(rect=[0,.04,1,.95]);fig.text(.5,.015,'Rows sum to 100%; noisy counts pool 5 observations per window and model, not independent samples',ha='center',fontsize=9)
                name=f'awgn_{row.method}_{round(row.ratio*100):03d}_all_conditions.png'
                fig.savefig(figures/name,dpi=180);plt.close(fig);images.append(name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(protocol,condition,title) in zip(axes,[('clean_only','clean','Clean training / clean test'),('awgn_mixed','white_-4','AWGN training / -4 dB test')]):
        data=summary[(summary.protocol==protocol)&(summary.condition==condition)]
        for method,g in data.groupby('method'):
            g=g.sort_values('ratio');ax.errorbar(100*g.ratio,100*g.acc_mean,yerr=100*g.acc_std.fillna(0),marker='o',capsize=3,label=method)
        ax.set(xlabel='Common training budget (%)',ylabel='Window ACC (%)',title=title);ax.legend()
    fig.tight_layout();fig.savefig(figures/'accuracy_overview.png',dpi=180);plt.close(fig);images.insert(0,'accuracy_overview.png')
    class_support=pd.crosstab(windows.role,windows.label).reindex(['train','validation','test'])
    class_support.columns=LABELS
    load_support=pd.crosstab(windows.role,windows.load_hp).reindex(['train','validation','test'])
    history=pd.crosstab(windows.role,windows.historical_role).reindex(['train','validation','test'])
    lines=['# 随机窗口划分的共同训练预算实验','',
           f'本轮完成{len(complexity)}个随机初始化模型、{len(detailed)}组逐次指标、{len(pooled)}组汇总混淆矩阵。全部模型完成验证选模后统一测试，不加载旧权重或概率代替训练。','',
           '**本支线评价同记录分布内的随机窗口识别，不能作为跨负载、跨记录或独立轴承泛化证据。**','',
           '## 划分与标签预算','',
           f"每条记录先切24000点无重叠窗口，再以固定种子{cfg['split_seed']}按70%/15%/15%随机分配。OR014_3整条排除，剩39条记录、667窗口。候选训练{cfg['candidate_training_windows']}，共同最大训练预算{cfg['training_windows']}、验证{cfg['validation_windows']}、测试{cfg['test_windows']}。",'',
           '各类实际窗口数','']
    lines+=markdown_table(class_support)
    lines+=['','各负载实际窗口数','']+markdown_table(load_support)
    lines+=['','原负载划分角色与新角色的对应数量。原角色仅用于披露历史曝光，不用于本轮分配。','']+markdown_table(history)
    lines+=['',f"候选训练与测试共享{cfg['same_record_train_test_overlap']}条原始记录。候选训练集中有{cfg['historical_test_windows_now_training']}个窗口曾属于历史测试集。实际预算子集的共享记录和曝光数见selected_windows.csv。历史测试曝光不因重新划分消失。",'',
            'IR014_0只有2个完整窗口，本轮1个训练、1个验证、0个测试。该记录不能同时覆盖三个集合，未复制或补齐窗口。其他记录的取整及零头平衡见record_allocation.csv。各集合没有相同窗口编号，原采样区间不重叠，但同记录、邻近窗口仍可能高度相关。','',
            '融合模型训练预算如下，验证窗口预算另列，不属于共同预算比例。','',
            '|共同预算比例|训练窗口|额外标注验证窗口|','|---|---:|---:|']
    for ratio,count in cfg['training_budget_windows'].items():lines.append(f'|{float(ratio):.0%}|{count}|{cfg["validation_windows"]}|')
    lines+=['','## 模型与观测协议','',
            '本轮固定600次更新，每50次更新验证，共12次选模机会，不提前停止。原按负载冻结结果验证200窗且允许提前停止，本轮验证100窗，因此只匹配训练窗口及类别配额，不称全部训练条件已匹配。与原467窗随机划分结果相比，还改变停止规则和验证时点，差异不能只归因于训练数量。小样本没有同预算数据单分支，不证明物理分支提高样本效率。','',
            '模型采用原固定212维完整融合网络，53068参数，无质量q、无类别定向修正。数据单分支33964参数。融合模型共同最大预算及其10%、20%、50%从头训练，数据单分支仅共同最大预算。100%指287窗共同预算，不是467窗候选全集。各档29/58/144/287窗，类别配额与原按负载实验一致。全样本3种子，小样本每比例3子集乘3种子。标准化仅拟合对应新训练子集。','',
            '纯干净协议训练、验证与测试均不人工加噪。AWGN每次窗口出现独立50%干净，否则SNR连续U[-4,4]dB；同一观测供两个分支，单批一次CE。AWGN验证干净权重0.5、五档各0.1。测试各噪声档5实例，按本轮测试窗口顺序生成，不能直接复用旧180窗口概率或声称噪声波形相同。','',
            '## 测试结果','',
            '窗口ACC均值±模型间样本标准差，单位%。先在每模型内平均噪声重复，再跨种子和子集汇总。小样本9个组合不是9个独立数据集，标准差不是独立轴承泛化置信区间。','']
    for protocol in complexity.protocol.unique():
        conditions=['clean'] if protocol=='clean_only' else [key for key,_ in CONDITIONS]
        lines += [f'### {protocol}','','|模型|共同预算比例|'+'|'.join(conditions)+'|','|---|---:|'+'|'.join(['---:']*len(conditions))+'|']
        for row in complexity[complexity.protocol.eq(protocol)][['method','ratio']].drop_duplicates().itertuples():
            cells=[]
            for condition in conditions:
                value=summary[(summary.protocol==protocol)&(summary.method==row.method)&(summary.ratio==row.ratio)&(summary.condition==condition)].iloc[0]
                cells.append(f'{100*value.acc_mean:.2f} ± {100*value.acc_std:.2f}')
            lines.append(f'|{row.method}|{row.ratio:.0%}|'+ '|'.join(cells)+'|')
        lines.append('')
    lines += ['均衡ACC、宏F1、各类召回率见summary.csv与class_metrics.csv。原始逐次预测与混淆计数保留；汇总矩阵逐真实类别归一化，不把噪声重复当新增窗口。','','## 计算开销','',
              f"硬件{cfg['hardware']}，PyTorch {cfg['torch']}，CPU线程{cfg['cpu_threads']}，精度{cfg['precision']}。训练batch32，含在线物理提取、增强和反向，排除验证。全部实际轮计时含首轮与可能不完整末轮；总训练时间、拟合与验证前向耗时另列complexity.csv。峰值为allocated显存。",'',
              '推理batch1、10预热、50重复，CUDA同步墙钟，包括归一化、在线完整物理提取和网络，排除磁盘/MAT解码/H2D。不用缓存隐藏提取成本，不与旧缓存或排除预热轮的训练时间直接排序。','',
              '|协议|模型|比例|推理ms|训练s/轮|峰值MiB|参数K|拟合s|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in cs.itertuples():lines.append(f'|{r.protocol}|{r.method}|{r.ratio:.0%}|{r.latency_ms:.3f}|{r.train_seconds_per_epoch:.3f}|{r.peak_gpu_mib:.1f}|{r.parameters/1000:.3f}|{r.fit_seconds:.3f}|')
    lines += ['','## 图表','']
    for name in images:lines += [f'![{name}](figures/{name})','']
    lines += ['## 解释边界','',
              '该划分允许多数原始记录及0至3HP同时出现在三个集合。没有重复采样点只保证窗口不重叠，不能保证记录独立。较高准确率只说明本随机窗口协议下识别良好，不能转述为跨负载问题已解决。','',
              '与原按负载实验同时改变训练数、验证数、测试组成和负载覆盖。准确率差不能单独解释为模型进步或某一种信息共享机制的因果效果，不作独立确认性SOTA比较。新测试100窗口，类别不平衡，支持数及记录依赖须与指标一起报告。','',
              '本轮只有一个预先固定划分种子，不代表划分不确定性已充分评估。OR014_3事后排除及历史测试曝光继续存在。融合对照改变信息来源、参数量和结构，不能单独证明物理因果机理；两训练协议还改变验证目标，差异不是纯粹噪声增强因果效果。白噪声结论只限当前合成条件。','',
              '[固定协议](PROTOCOL.md)']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'README.md').write_text('# 随机窗口共同训练预算实验\n\n[正式报告](REPORT.md) · [固定协议](PROTOCOL.md)\n\n窗口角色见windows.csv，原始记录共享情况见record_overlap.csv。该结果不属于跨负载或独立记录泛化验证。\n',encoding='utf-8')
    print(summary[['protocol','method','ratio','condition','acc_mean','balanced_acc_mean','f1_macro_mean']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    report(parser.parse_args().results)
