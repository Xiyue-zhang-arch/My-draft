"""V1只从保存的基线预测汇总，不在报告阶段重新训练或推理。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=['acc','balanced_acc','f1_macro','recall_N','recall_IR','recall_OR','recall_B']
KEYS=['split','source_load_hp','target_load_hp','protocol','method','seed','condition']
LABELS=['N','IR','OR','B']


def scores(labels,p):
    counts=confusion_matrix(labels,p.argmax(1),labels=range(4))
    support=counts.sum(1)
    if (support==0).any():
        raise ValueError('测试缺失类别')
    recall=counts.diagonal()/support
    return dict(acc=float(np.trace(counts)/counts.sum()),balanced_acc=float(recall.mean()),
        f1_macro=float(f1_score(labels,p.argmax(1),labels=range(4),average='macro',zero_division=0)),
        **dict(zip(METRICS[3:],map(float,recall)))),counts


def markdown(table):
    return table.to_markdown(index=False)


def plot_comparison(summary,out,split,metric):
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'pdf.fonttype':42})
    groups=[('clean_only','clean','纯干净训练 · 干净测试')]+[
        ('awgn_mixed',condition,'AWGN训练 · '+title) for condition,title in
        [('clean','干净测试'),('white_-4','−4 dB'),('white_-2','−2 dB'),('white_+0','0 dB'),('white_+2','+2 dB'),('white_+4','+4 dB')]]
    fig,axes=plt.subplots(4,2,figsize=(16,17),squeeze=False)
    for ax,(protocol,condition,title) in zip(axes.flat,groups):
        g=summary[(summary.split==split)&(summary.protocol==protocol)&(summary.condition==condition)]
        directions=sorted(set(zip(g.source_load_hp,g.target_load_hp)))
        xx=np.arange(len(directions))
        for offset,method,color,label in [(-.18,'fusion','#2166AC','融合模型'),(.18,'data_only','#E08214','数据单分支')]:
            values=g[g.method==method].set_index(['source_load_hp','target_load_hp']).reindex(directions)
            ax.bar(xx+offset,100*values[f'{metric}_mean'],width=.34,yerr=100*values[f'{metric}_std'],
                color=color,label=label,capsize=2,error_kw={'elinewidth':.8})
            if split=='window_random':
                for x,mean,sd in zip(xx+offset,values[f'{metric}_mean'],values[f'{metric}_std']):
                    ax.text(x,100*(mean+sd)+1,f'{mean*100:.2f} ± {sd*100:.2f}',ha='center',fontsize=8)
        labels=['记录内随机划分'] if split=='window_random' else [f'{s}→{t}' for s,t in directions]
        ax.set(xticks=xx,xticklabels=labels,ylim=(0,112),yticks=np.arange(0,101,20),title=title,
            ylabel=('均衡准确率 (%)' if metric=='balanced_acc' else '窗口准确率 (%)'))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if split=='load_transfer':
            ax.set_xlabel('来源HP → 目标HP')
    axes.flat[-1].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2)
    fig.suptitle(('先切窗再划分' if split=='window_random' else '单一来源负载迁移')+' · 全样本基线',y=.998)
    fig.text(.5,.008,'误差棒为5个训练种子的均值±样本标准差；含噪结果先在每个模型内平均5个噪声实例。',ha='center')
    fig.tight_layout(rect=(0,.025,1,.96))
    name=f'{split}_{metric}'
    for extension in ['png','pdf']:
        fig.savefig(out/'figures'/f'{name}.{extension}',dpi=180)
    plt.close(fig)


def finish(out):
    if (out/'REPORT.md').exists():
        raise FileExistsError('正式报告禁止覆盖')
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(out/'metrics.csv')
    if len(metrics)!=3510:
        raise ValueError(f'逐次指标数量异常 {len(metrics)}')
    per_seed=metrics.groupby(KEYS,as_index=False)[METRICS].mean()
    if per_seed.duplicated(KEYS).any():
        raise ValueError('重复种子指标')
    summary=per_seed.groupby(KEYS[:-2]+['condition'])[METRICS].agg(['mean','std','count'])
    summary.columns=['_'.join(column) for column in summary.columns]
    summary=summary.reset_index()
    assert summary.acc_count.eq(5).all() and len(summary)==182
    summary['balanced_acc_sd_over_0_5pp']=summary.balanced_acc_std.gt(.005)
    per_seed.to_csv(out/'per_seed_metrics.csv',index=False)
    summary.to_csv(out/'mean_std.csv',index=False)
    # 同种子配对差值只描述当前两种方法，不作为结构改进试验。
    index=[key for key in KEYS if key!='method']
    paired=per_seed.pivot(index=index,columns='method',values='balanced_acc').reset_index()
    paired['fusion_minus_data']=paired.fusion-paired.data_only
    paired.to_csv(out/'paired_seed_differences.csv',index=False)
    load=per_seed[per_seed.split=='load_transfer'].groupby(['protocol','method','seed','condition'],as_index=False)[METRICS].mean()
    overall=load.groupby(['protocol','method','condition'])[METRICS].agg(['mean','std'])
    overall.columns=['_'.join(c) for c in overall.columns]
    overall=overall.reset_index()
    overall.to_csv(out/'load_direction_equal_weight.csv',index=False)
    # 再由冻结预测独立复算指标，原始计数与行归一化表同时保留。
    count_rows=[]
    for folder in (out/'models').glob('*/*/*/seed_*/test_*'):
        info=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        with np.load(folder/'predictions.npz') as z:
            labels=z['labels']
            expected=metrics[(metrics.split==info['split'])&(metrics.source_load_hp==info['source_load_hp'])&
                (metrics.target_load_hp==info['target_load_hp'])&(metrics.protocol==info['protocol'])&
                (metrics.method==info['method'])&(metrics.seed==info['seed'])]
            for row in expected.itertuples():
                values,counts=scores(labels,z[f'{row.condition}_n{row.noise_seed}'])
                np.testing.assert_allclose([getattr(row,k) for k in METRICS],[values[k] for k in METRICS],atol=1e-12,rtol=0)
                for i in range(4):
                    for j in range(4):
                        count_rows.append(dict(**{k:getattr(row,k) for k in KEYS},true=i,predicted=j,count=int(counts[i,j]),
                            original_class_support=int((labels==i).sum())))
    counts=pd.DataFrame(count_rows)
    if len(counts)!=3510*16:
        raise ValueError('预测文件不完整')
    counts=counts.groupby([k for k in KEYS if k!='seed']+['true','predicted'],as_index=False).agg(
        count=('count','sum'),original_class_support=('original_class_support','first'))
    counts['row_fraction']=counts['count']/counts.groupby([k for k in KEYS if k!='seed']+['true'])['count'].transform('sum')
    counts.to_csv(out/'confusion_counts.csv',index=False)
    (out/'figures').mkdir()
    for split in ['window_random','load_transfer']:
        for metric in ['balanced_acc','acc']:
            plot_comparison(summary,out,split,metric)
    shown=summary[['split','source_load_hp','target_load_hp','protocol','method','condition','balanced_acc_mean','balanced_acc_std','balanced_acc_sd_over_0_5pp']].copy()
    for c in ['balanced_acc_mean','balanced_acc_std']:
        shown[c]=(100*shown[c]).round(3)
    support=pd.read_csv(out/'dataset_support.csv')
    complexity=pd.read_csv(out/'complexity.csv')
    training=pd.read_csv(out/'training_summary.csv')
    lines=['# CWRU_V1 全样本基线','',
        '两种数据划分、两套噪声训练协议、融合与数据单分支，各配置5种子。100个随机初始化模型完成选模后统一测试，260组模型目标评价、3510组逐次指标。全部40条记录，无排除、无小样本、无训练预算匹配。尚未开展结构探索。','',
        '普通交叉熵，标签平滑0；最多100epoch，每轮验证，连续20轮验证NLL未改善早停，改善阈值1e-6。测试使用最佳验证权重。AdamW与逐epoch余弦学习率，固定划分和种子见[协议](PROTOCOL.md)。','',
        '## 数据使用','',markdown(support),'',
        '全样本是训练角色内全部窗口；保持有放回的类别/记录均衡采样，每epoch抽取ceil(N/32)×32份观测，不保证每窗每轮出现一次。训练采样策略未作为新增变量修改。IR014_0仅2窗，记录内划分无法同时进入三集合，实际分配已记录。来源验证数量不相同。完整原始文件来源与各集合数量见[数据说明](DATA_USAGE.md)。','',
        '## 均值与样本标准差','',
        '主指标均衡准确率，辅助窗口ACC、宏F1与各类召回。先平均每模型5个噪声实例，再计算5训练种子的均值与样本标准差ddof=1；不将噪声实例视为新增种子。表内指标单位为百分数，标准差单位为百分点。','',
        markdown(shown),'',
        '标准差超过0.5个百分点的配置标记为波动超阈值。本轮只建立基线，不宣布方法提升。后续改进必须与同划分、同协议、同种子的这版基线配对，不能以某个最佳种子作为参照。标准差阈值是用户的探索准则，不等于统计显著性检验。','',
        '## 图表','',*[f'![{split} {metric}](figures/{split}_{metric}.png)\n' for split in ['window_random','load_transfer'] for metric in ['balanced_acc','acc']],
        '## 计算成本','',markdown(complexity),'',
        '静态FLOPs只计Conv1d和Linear，乘加记2次，不包括FFT物理描述、归一化等，因此不是端到端总FLOPs。单样本延迟包含归一化、在线物理提取和网络，排除磁盘与H2D，预热10次、重复50次、CUDA同步。每个来源模型只测一次，三个目标方向复用该计时。','',
        f"设备 {cfg['hardware']}；CUDA编码器BF16，其余FP32；训练batch32，单样本延迟batch1。训练计时含在线物理提取和反向传播，不含验证及保存。峰值显存为GPU allocated。",'',
        markdown(training.groupby(['split','protocol','method'])[['parameters','epochs_trained','updates_trained','best_epoch','total_train_seconds','train_seconds_per_epoch','validation_seconds','fit_seconds','peak_gpu_mib','latency_ms']].mean().reset_index().round(4)),'',
        '## 证据边界','',
        '只有一次固定数据划分，5种子描述训练随机性，不能估计数据划分不确定性。记录内随机划分的训练测试共享原始记录，只能描述窗口级识别。按负载迁移的来源训练验证共享记录，目标负载留出不等于独立轴承留出。12方向复用来源模型和目标数据，不能作为12个独立轴承重复。两方案训练量及验证量不同，性能差不能只归因于数据划分方式。所有负载存在历史测试曝光，本轮属于探索性基线，不是独立确认性验证。AWGN结论不推广到所有实际噪声。恢复全部记录不消除既往事后排除与测试曝光。',
        '', '报告对保存预测逐项复算全部3510组指标，未增加模型前向推理。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
