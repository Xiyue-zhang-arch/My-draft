from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path(r'D:\Project\MY draft')
RES=ROOT/'experiments/CWRU/实验结果'/'CWRU_V3'
SRC=ROOT/'thesis_document'/'成果'/'V3_论文_物理特征文献说明_20261031'
OUT=ROOT/'thesis_document'/'成果'/'V3_论文_箱线图及误差线_20261031'
OUT.mkdir(parents=True,exist_ok=True)
shutil.copytree(SRC/'media',OUT/'media',dirs_exist_ok=True)
shutil.copy2(SRC/'物理特征文献依据对照.md',OUT/'物理特征文献依据对照.md')
plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'axes.linewidth':.85,'pdf.fonttype':42,'svg.fonttype':'none'})
conditions=['white_+4','white_+2','white_+0','white_-2','white_-4','clean']
condition_labels=['+4','+2','0','−2','−4','Clean']
variants=['baseline_data','data_aux_ce']
palette={'baseline_data':'#c9b8d9','data_aux_ce':'#ff8580'}
names={'baseline_data':'Data-only','data_aux_ce':'Fusion model'}
ratios=[.1,.2,.5,1.]
window=pd.read_csv(RES/'window_random_20261023_0853'/'evaluation'/'per_seed_metrics.csv')
small=pd.read_csv(RES/'load_small_20261025_2017'/'evaluation'/'transfer_equal_direction_per_seed.csv')
full=pd.read_csv(RES/'load_full_20261024_0950'/'evaluation'/'per_seed_metrics.csv')

def get_seed_series(task,ratio,variant,condition):
    frame=window if task=='window' else (full if ratio==1 else small)
    d=frame[(frame.ratio==ratio)&(frame.protocol=='awgn_mixed')&(frame.variant==variant)&(frame.condition==condition)]
    if task=='window' and ratio==1: d=d[d.subset==0]
    if task=='load' and ratio==1:
        assert len(d)==60 and d.groupby('seed').size().eq(12).all()
        per=d.groupby('seed').balanced_acc.mean()
    elif ratio<1:
        assert len(d)==15 and d.groupby('seed').size().eq(3).all()
        per=d.groupby('seed').balanced_acc.mean()
    else:
        assert len(d)==5
        per=d.set_index('seed').balanced_acc.sort_index()
    assert len(per)==5 and set(per.index)==set(range(1706,1711))
    return per*100

records=[]
for task in ['window','load']:
    fig,axs=plt.subplots(2,2,figsize=(14,9.8))
    fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.86,hspace=.48,wspace=.19)
    for idx,(ax,ratio) in enumerate(zip(axs.flat,ratios)):
        for ci,condition in enumerate(conditions):
            for vi,variant in enumerate(variants):
                vals=get_seed_series(task,ratio,variant,condition)
                for seed,v in vals.items():records.append(dict(task=task,ratio=ratio,condition=condition,variant=variant,seed=seed,balanced_acc_percent=v,definition='mean across 3 subsets at fixed seed' if ratio<1 else 'full-budget fixed seed',noise='mean of five instances except clean'))
                pos=ci+(-.17 if vi==0 else .17)
                ax.boxplot([vals.to_numpy()],positions=[pos],widths=.28,patch_artist=True,whis=1.5,showfliers=False,medianprops={'color':'black','linewidth':1.0},boxprops={'facecolor':palette[variant],'edgecolor':'black','linewidth':.8},whiskerprops={'color':'black','linewidth':.8},capprops={'color':'black','linewidth':.8})
                jitter=np.array([-.060,-.030,0,.030,.060]);ax.scatter(pos+jitter,vals,s=12,marker='D',color='#b91e24',edgecolors='white',linewidths=.2,zorder=4)
                ax.errorbar(pos,float(vals.mean()),yerr=float(vals.std(ddof=1)),fmt='s',color='black',markersize=3,capsize=3,elinewidth=.75,zorder=5)
            if ci<5:ax.axvline(ci+.5,color='#777',lw=.65,ls=':')
        ax.set_xticks(range(len(conditions)),condition_labels);ax.set_xlabel('Test SNR (dB)');ax.set_ylabel('Balanced accuracy (%)')
        ax.set_ylim(0,105) if task=='load' else ax.set_ylim(35,105)
        ax.set_yticks(np.arange(0,101,20) if task=='load' else [40,60,80,100]);ax.grid(axis='y',color='#c7c7c7',lw=.6);ax.tick_params(direction='in',top=True,right=True)
        ax.text(.5,-.24,f'({chr(97+idx)}) Training ratio = {int(ratio*100)}%',transform=ax.transAxes,ha='center',va='top',fontsize=14)
    legend=[Patch(facecolor=palette[v],edgecolor='black',label=names[v]) for v in variants]
    legend += [Line2D([0],[0],marker='s',color='black',ls='None',markersize=5,label='Mean ± SD'),Line2D([0],[0],color='black',lw=1,label='Median'),Line2D([0],[0],marker='D',color='#b91e24',ls='None',markersize=4,label='Seed value')]
    fig.legend(handles=legend,loc='upper center',ncol=5,frameon=True,fancybox=False,edgecolor='black',fontsize=12,bbox_to_anchor=(.5,.99))
    fig.text(.5,.915,'AWGN mixed training',ha='center',fontsize=12)
    fig.text(.5,.035,'Boxes: Q1–Q3; whiskers: 1.5 IQR; black square/error bar: mean ± sample SD; n = 5 seeds.',ha='center',fontsize=11)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'media'/f'noise_box_{task}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
pd.DataFrame(records).to_csv(OUT/'boxplot_seed_values.csv',index=False,encoding='utf-8-sig')

# Category bars: mean and SD across the three subset means, matching the prior thesis.
df=pd.read_csv(RES/'load_small_20261025_2017'/'evaluation'/'transfer_between_subset_summary.csv')
cols=['recall_N','recall_IR','recall_OR','recall_B']; labels=['Normal','Inner race','Outer race','Ball']; colors=['#ff8580','#ffc28e','#97c99f','#aabde8']
rows=[];fig,axes=plt.subplots(1,2,figsize=(15,6.8),sharey=True);fig.subplots_adjust(left=.065,right=.985,bottom=.17,top=.84,wspace=.14)
for ax,variant in zip(axes,variants):
    for ci,(col,label,c) in enumerate(zip(cols,labels,colors)):
        means=[];stds=[]
        for ratio in ratios[:3]:
            x=df[(df.ratio==ratio)&(df.protocol=='awgn_mixed')&(df.variant==variant)&(df.condition=='white_-4')]
            assert len(x)==1
            m=float(x.iloc[0][col+'_mean'])*100; sd=float(x.iloc[0][col+'_std'])*100
            means.append(m);stds.append(sd);rows.append(dict(ratio=ratio,variant=variant,category=col,mean_pct=m,sd_pct=sd,sd_basis='3 subset means after averaging five seeds and 12 directions'))
        pos=np.arange(3)+(ci-1.5)*.19
        bars=ax.bar(pos,means,.185,yerr=stds,capsize=3,linewidth=.6,edgecolor='#444',color=c,error_kw={'elinewidth':.8},label=label,zorder=3)
        for bar,m,sd in zip(bars,means,stds):ax.text(bar.get_x()+bar.get_width()/2,m+sd+2,f'{m:.1f}±{sd:.1f}',rotation=90,ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(3),['10%','20%','50%']);ax.set_xlabel('Training sample ratio');ax.set_title(names[variant],fontsize=15);ax.set_ylim(0,145);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',color='#c7c7c7',lw=.6);ax.tick_params(direction='in',top=True,right=True)
axes[0].set_ylabel('Class recall (%)');fig.legend(handles=[Patch(facecolor=c,edgecolor='#444',label=l) for c,l in zip(colors,labels)],loc='upper center',ncol=4,frameon=True,fancybox=False,edgecolor='black',bbox_to_anchor=(.5,.98))
fig.text(.5,.035,'AWGN training / −4 dB; bars = mean, error bars = SD across 3 subset means.',ha='center',fontsize=11)
for ext in ['png','pdf','svg']:fig.savefig(OUT/'media'/f'class_recall_sd.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig);pd.DataFrame(rows).to_csv(OUT/'class_recall_sd_values.csv',index=False,encoding='utf-8-sig')

# Timing SD describes saved per-model benchmark means, not their 50 timing repeats.
cost=pd.read_csv(RES/'load_small_20261025_2017'/'computational_cost.csv')
agg=cost.groupby('variant').agg(n=('job_id','size'),param_mean=('parameters','mean'),param_sd=('parameters','std'),latency_mean=('latency_ms','mean'),latency_sd=('latency_ms','std')).loc[variants]
assert agg.n.eq(360).all() and agg.param_sd.eq(0).all()
fig,axs=plt.subplots(1,2,figsize=(11,5.5));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.84,wspace=.30)
for ax,mean,sd,ylabel in [(axs[0],'param_mean','param_sd','Parameters (thousands)'),(axs[1],'latency_mean','latency_sd','Single-sample latency (ms)')]:
    factor=.001 if mean=='param_mean' else 1
    for i,v in enumerate(variants):
        m=agg.loc[v,mean]*factor; e=agg.loc[v,sd]*factor
        ax.bar(i,m,.55,yerr=e,capsize=5,color=palette[v],edgecolor='#444',linewidth=.7,error_kw={'elinewidth':1})
        ax.text(i,m+e+(1.2 if factor==.001 else .18),f'{m:.2f} ± {e:.2f}',ha='center',fontsize=11)
    ax.set_xticks([0,1],[names[v] for v in variants]);ax.set_ylabel(ylabel);ax.set_ylim(0,(agg[mean]*factor+agg[sd]*factor).max()*(1.28 if factor==.001 else 1.38));ax.grid(axis='y',color='#ccc',lw=.6);ax.tick_params(direction='in',top=True,right=True)
fig.text(.5,.075,'Latency: mean ± SD across 360 model-level benchmark means per method; parameter counts are fixed.',ha='center',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(OUT/'media'/f'cost_sd.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig);agg.reset_index().to_csv(OUT/'cost_sd_values.csv',index=False,encoding='utf-8-sig')

tex=(SRC/'thesis.tex').read_text(encoding='utf-8')
for old,new in [(8,10),(7,8),(6,7),(5,6),(4,5)]:tex=tex.replace(f'图5-{old}',f'图5-临{new}')
tex=tex.replace('图5-临','图5-')
tex=tex.replace('media/image8.png','media/class_recall_sd.pdf').replace('media/image10.png','media/cost_sd.pdf')
def figure(name,caption):
    return '\\clearpage\n\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{media/'+name+'.pdf}\n\\par\\small '+caption+'\n\\end{figure}\n'
for anchor,file,caption in [
('图5-3 记录内随机窗口在不同训练比例和噪声条件下的均衡准确率','noise_box_window','图5-4 随机窗口不同噪声条件下的种子结果分布'),
('图5-8 跨负载实验在不同训练比例和噪声条件下的均衡准确率','noise_box_load','图5-9 跨负载不同噪声条件下的种子结果分布')]:
    a='\\par\\small '+anchor
    assert tex.count(a)==1,(a,tex.count(a))
    # Insert after the enclosing figure, rather than inside it.
    start=tex.index(a);end=tex.index('\\end{figure}',start)+len('\\end{figure}')
    note=('图中两种模型均采用AWGN混合训练，Clean表示测试时不额外加噪，其余各项为对应信噪比。箱线图中每组有五个训练种子。10\\%、20\\%和50\\%预算先将同一种子下的三个子集结果求平均，'
          '再展示五个种子均值的分布；全样本直接展示五种子结果。盒体表示上下四分位数，'
          '黑线为中位数，黑色方点及误差线为均值和样本标准差，红色点为各种子值。'
          '每档噪声的五个实例先在模型内平均；不同种子共享测试数据，不能视为五套独立数据集。'
          '该图的小样本标准差与前文三个子集均值之间的标准差口径不同。')
    if file=='noise_box_load':
        note+=' 跨负载各点均先对12个方向等权平均。10\\%训练比例、−4 dB测试下，数据单分支和融合方案分别为36.17\\%±16.29个百分点和85.96\\%±0.27个百分点。该标准差仅描述子集平均后的种子波动。'
    else:
        note+=' 10\\%训练比例、−4 dB测试下，数据单分支和融合方案分别为82.66\\%±19.54个百分点和94.72\\%±2.27个百分点。该标准差仅描述子集平均后的种子波动。'
    tex=tex[:end]+'\n\n'+figure(file,caption)+note+'\n'+tex[end:]
classcap='\\begin{center}图5-6 小样本跨负载−4 dB类别召回率\\end{center}'
assert tex.count(classcap)==1
tex=tex.replace(classcap,classcap+'\n柱高为均值，误差线为三个子集均值之间的样本标准差。每个子集先对五个训练种子和12个迁移方向求平均。图中数值以均值及标准差表示，单位为百分点。',1)
# Move the small-sample class plot to its own analysis section, then maintain figure order.
start=tex.index('\\noindent\\includegraphics[width=\\linewidth,keepaspectratio]{media/class_recall_sd.pdf}')
stop=tex.index('\\section{小样本跨负载实验结果}',start)
class_block=tex[start:stop]
tex=tex[:start]+tex[stop:]
position=tex.index('进一步对固定方向、预算、子集、协议和测试条件下的五种子平均差值进行统计')
tex=tex[:position]+class_block+'\n'+tex[position:]
for old,new in [(6,9),(7,6),(8,7),(9,8)]:
    tex=tex.replace(f'图5-{old} ',f'图5-次{new} ')
tex=tex.replace('图5-次','图5-')
old_note='在图中，横轴左侧为五个有噪测试信噪比'
assert tex.count(old_note)==2
tex=tex.replace(old_note,'在图5-3中，横轴左侧为五个有噪测试信噪比',1)
tex=tex.replace(old_note,'在图5-7中，横轴左侧为五个有噪测试信噪比',1)
costcap='\\begin{center}图5-10 参数量与推理延迟比较\\end{center}'
assert tex.count(costcap)==1
tex=tex.replace(costcap,costcap+'\n参数量为固定的结构值，标准差为零；延迟为每种方法360个模型各自基准计时均值的总体描述，误差线表示这些模型级均值之间的样本标准差。该误差线不是单次基准中50次重复计时的标准差，也不是跨设备置信区间。',1)
(OUT/'thesis.tex').write_text(tex,encoding='utf-8')
print('done',OUT,'box observations',len(records),'cost',agg.to_string())
