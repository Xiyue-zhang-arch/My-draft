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
for task in ['window']:
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
        # The same 96–101 scale is used for the 20–100% panels. The 10% panel
        # retains its lower seed value and its complete mean ± SD error bars.
        if ratio == .1:
            ax.set_ylim(45,105); ax.set_yticks([50,60,70,80,90,100])
        else:
            ax.set_ylim(96,101); ax.set_yticks([96,97,98,99,100,101])
        ax.grid(axis='y',color='#c7c7c7',lw=.6);ax.tick_params(direction='in',top=True,right=True)
        ax.text(.5,-.24,f'({chr(97+idx)}) Training ratio = {int(ratio*100)}%',transform=ax.transAxes,ha='center',va='top',fontsize=14)
    legend=[Patch(facecolor=palette[v],edgecolor='black',label=names[v]) for v in variants]
    legend += [Line2D([0],[0],marker='s',color='black',ls='None',markersize=5,label='Mean ± SD'),Line2D([0],[0],color='black',lw=1,label='Median'),Line2D([0],[0],marker='D',color='#b91e24',ls='None',markersize=4,label='Seed value')]
    fig.legend(handles=legend,loc='upper center',ncol=5,frameon=True,fancybox=False,edgecolor='black',fontsize=12,bbox_to_anchor=(.5,.99))
    fig.text(.5,.915,'AWGN mixed training',ha='center',fontsize=12)
    fig.text(.5,.035,'Boxes: Q1–Q3; whiskers: 1.5 IQR; square/error bar: mean ± SD; n = 5; y-axis limits differ.',ha='center',fontsize=11)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'media'/f'noise_box_{task}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


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
    ax.set_xticks(range(3),['10%','20%','50%']);ax.set_xlabel('Training sample ratio');ax.set_title(names[variant],fontsize=15);ax.set_ylim(0,105);ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',color='#c7c7c7',lw=.6);ax.tick_params(direction='in',top=True,right=True)
axes[0].set_ylabel('Class recall (%)');fig.legend(handles=[Patch(facecolor=c,edgecolor='#444',label=l) for c,l in zip(colors,labels)],loc='upper center',ncol=4,frameon=True,fancybox=False,edgecolor='black',bbox_to_anchor=(.5,.98))
fig.text(.5,.035,'AWGN training / −4 dB; bars = mean, error bars = SD across 3 subset means.',ha='center',fontsize=11)
for ext in ['png','pdf','svg']:fig.savefig(OUT/'media'/f'class_recall_sd.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)


# The record-internal panel uses different vertical limits; retain all other thesis edits.
tex_path=OUT/'thesis.tex'
tex=tex_path.read_text(encoding='utf-8')
old='图中两种模型均采用AWGN混合训练，Clean表示测试时不额外加噪，其余各项为对应信噪比。'
assert tex.count(old)==2
tex=tex.replace(old,old+'为便于观察，10\\%训练比例子图的纵轴为45\\%至105\\%，其他三个子图的纵轴为96\\%至101\\%；跨子图比较时须注意刻度不同。',1)
old2='图中数值以均值及标准差表示，单位为百分点。'
assert tex.count(old2)==1
tex=tex.replace(old2,'召回率以百分比表示，标准差以百分点表示。',1)
tex_path.write_text(tex,encoding='utf-8')
