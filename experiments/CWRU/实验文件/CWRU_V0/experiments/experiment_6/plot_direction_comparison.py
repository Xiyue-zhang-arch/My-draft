"""从冻结逐种子指标绘制12方向均值和样本标准差，不推理、不改原结果。"""
from pathlib import Path
from datetime import datetime
import sys
import argparse
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from experiment_paths import output_directory

PROJECT=Path(__file__).resolve().parents[6]
DIRECTIONS=[(s,t) for s in range(4) for t in range(4) if s!=t]
CONDITIONS=[('clean','干净'),('white_-4','−4 dB'),('white_-2','−2 dB'),('white_+0','0 dB'),('white_+2','2 dB'),('white_+4','4 dB')]
METHODS=[('fusion','融合模型','#2166AC'),('data_only','数据单分支','#E08214')]


def panel(ax,summary,protocol,condition,metric,title,labels=False):
    x=np.arange(12);width=.36
    highest=100.
    for offset,(method,label,color) in zip([-.19,.19],METHODS):
        data=summary[(summary.protocol==protocol)&(summary.condition==condition)&(summary.method==method)].set_index(['source_load_hp','target_load_hp']).reindex(DIRECTIONS)
        mean=100*data[metric+'_mean'].to_numpy();sd=100*data[metric+'_std'].to_numpy()
        assert np.isfinite(mean).all() and np.isfinite(sd).all()
        highest=max(highest,float(np.max(mean+sd)))
        ax.bar(x+offset,mean,width,yerr=sd,color=color,label=label,alpha=.95,
               error_kw=dict(ecolor='#303030',elinewidth=1,capsize=3,capthick=1),zorder=3)
        if labels:
            for xx,m,ss in zip(x+offset,mean,sd):
                ax.text(xx,m+ss+1.0,f'{m:.1f}±{ss:.1f}',rotation=65,ha='left',va='bottom',fontsize=7)
    for start in [0,6]:ax.axvspan(start-.5,start+2.5,color='#F1F4F7',zorder=0)
    for boundary in [2.5,5.5,8.5]:ax.axvline(boundary,color='#CED3D8',lw=.8,zorder=1)
    ax.set_xticks(x,[f'{s}→{t}' for s,t in DIRECTIONS])
    ax.set_xlim(-.65,11.65);ax.set_ylim(0,max(110 if labels else 105,5*np.ceil((highest+(12 if labels else 2))/5)))
    ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('均衡准确率（%）' if metric=='balanced_acc' else '窗口准确率（%）')
    ax.set_title(title,fontsize=12,pad=10);ax.grid(axis='y',alpha=.25,zorder=0)
    ax.spines[['top','right']].set_visible(False)


def save(fig,out,stem):
    fig.savefig(out/f'{stem}.png',dpi=220,bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_6')
    p.add_argument('--output',type=Path)
    a=p.parse_args();source=a.source.resolve()
    seeds=pd.read_csv(source/'per_seed_direction.csv')
    keys=['source_load_hp','target_load_hp','protocol','method','condition']
    assert not seeds.duplicated(keys+['seed']).any()
    assert seeds.groupby(keys).seed.nunique().eq(5).all()
    summary=seeds.groupby(keys)[['acc','balanced_acc']].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(x for x in col if x) for col in summary.columns]
    original=pd.read_csv(source/'per_direction_summary.csv')
    check=summary.merge(original,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==168
    for metric in ['acc','balanced_acc']:
        for stat in ['mean','std']:
            np.testing.assert_allclose(check[f'{metric}_{stat}_new'],check[f'{metric}_{stat}_old'],atol=1e-12,rtol=0)
    out=output_directory(a.output or PROJECT/'experiments/CWRU/实验结果/CWRU'/f'experiment_6_direction_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}')
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((name for name in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun'] if name in available),None)
    if font is None:raise RuntimeError('未找到中文绘图字体')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})
    summary.to_csv(out/'direction_mean_std.csv',index=False,encoding='utf-8-sig')
    caption='误差棒为5个训练种子的均值±样本标准差；含噪条件先在每模型内平均5个噪声实例。\n横轴为来源HP→目标HP；方向共享模型或测试数据，不视为独立轴承重复。'
    for metric in ['balanced_acc','acc']:
        fig,axes=plt.subplots(2,1,figsize=(17,10))
        panel(axes[0],summary,'clean_only','clean',metric,'纯干净训练 · 干净测试',labels=True)
        panel(axes[1],summary,'awgn_mixed','white_-4',metric,'AWGN混合训练 · −4 dB测试',labels=True)
        axes[1].set_xlabel('迁移方向（来源HP → 目标HP）')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,.962),frameon=False)
        fig.suptitle('实验6：12个单一来源负载迁移方向对比',fontsize=16,y=.995)
        fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=10)
        fig.tight_layout(rect=[0,.09,1,.93],h_pad=2.5)
        save(fig,out,f'{metric}_overview')
        fig,axes=plt.subplots(3,2,figsize=(20,14))
        for ax,(condition,title) in zip(axes.flat,CONDITIONS):
            panel(ax,summary,'awgn_mixed',condition,metric,f'AWGN混合训练 · {title}测试')
            ax.set_xlabel('来源HP → 目标HP')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,.966),frameon=False)
        fig.suptitle('实验6：AWGN训练模型的全部测试条件',fontsize=17,y=.995)
        fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=11)
        fig.tight_layout(rect=[0,.07,1,.94],h_pad=2,w_pad=2)
        save(fig,out,f'{metric}_awgn_all_conditions')
    lines=['# 实验6 12方向均值与标准差对比图','','只读取原实验6的per_seed_direction.csv作图，没有重新训练、推理或计时，原模型与结果不变。原始统计逐项匹配per_direction_summary.csv。','',
        '每方向5个训练种子1706至1710，误差棒为均值加减样本标准差，ddof=1。含噪数据已先在每模型内平均5个噪声实例，不将其作为25个独立种子。均衡准确率是四类召回率的算术平均，窗口准确率按各类实际测试数量计算，不混用。','',
        '图中来源负载依次为0/1/2/3HP，每组内目标负载递增。均值与标准差都乘100，标准差单位为百分点。均值加标准差可超过100%，这是描述性误差棒，不是准确率预测区间，不截断误差棒。','',
        '主图对比纯干净训练的干净测试及AWGN训练的−4dB测试。另提供AWGN训练后干净及全部五噪声档。12方向共享来源模型和目标数据，不作为12个独立轴承重复；样本标准差描述训练随机性，不是独立轴承泛化置信区间。固定一次训练子集与验证划分的局限继续保留。','',
        '各来源73训练窗，验证19/39/40/40，测试92/195/200/200，全部40记录，无小样本。两模型同方向使用相同数据和600更新，跨来源验证预算未匹配。','',
        '[均值和标准差数值表](direction_mean_std.csv) · [原实验6正式报告](../experiment_6/REPORT.md)','','## 均衡准确率','',
        '![均衡准确率主图](balanced_acc_overview.png)','','[主图PDF](balanced_acc_overview.pdf)','','![AWGN全部条件](balanced_acc_awgn_all_conditions.png)','','[全部条件PDF](balanced_acc_awgn_all_conditions.pdf)','','## 窗口准确率','',
        '![窗口准确率主图](acc_overview.png)','','[主图PDF](acc_overview.pdf)','','![窗口准确率全部条件](acc_awgn_all_conditions.png)','','[全部条件PDF](acc_awgn_all_conditions.pdf)']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('OUTPUT',out,'font',font,'4PNG,4PDF,168rows; all original means/std matched.')


if __name__=='__main__':main()
