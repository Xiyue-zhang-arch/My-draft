from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(r'D:\Project\MY draft')
DATA=ROOT/'experiments/CWRU/实验结果'/'CWRU_V3'
SRC=ROOT/'thesis_document'/'成果'/'V3_毕业论文_LaTeX'
OUT=ROOT/'thesis_document'/'成果'/'V3_论文_箱线图及误差线_20261031'
OUT.mkdir(parents=True,exist_ok=True)

paths={
 'window_small':'window_random_20261023_0853/evaluation/between_subset_summary.csv',
 'window_full':'window_random_20261023_0853/evaluation/mean_std.csv',
 'load_small':'load_small_20261025_2017/evaluation/transfer_between_subset_summary.csv',
 'load_full':'load_full_20261024_0950/evaluation/transfer_equal_direction_mean_std.csv'}
tables={k:pd.read_csv(DATA/p) for k,p in paths.items()}
ratios=[.1,.2,.5,1.]; noises=[-4,-2,0,2,4]; colors={'baseline_data':'#1455d9','data_aux_ce':'#df1a20'}
methods=[('baseline_data','Data-only', 'o'),('data_aux_ce','Fusion','>')]
plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
records=[]
def value(task,ratio,protocol,variant,condition):
    df=tables[f'{task}_{"full" if ratio==1 else "small"}']
    x=df[(df.ratio==ratio)&(df.protocol==protocol)&(df.variant==variant)&(df.condition==condition)]
    assert len(x)==1,(task,ratio,protocol,variant,condition,len(x))
    row=x.iloc[0]
    mean=float(row.balanced_acc_mean)*100; std=float(row.balanced_acc_std)*100
    records.append(dict(experiment=task,ratio=ratio,protocol=protocol,variant=variant,condition=condition,mean_BA_pct=mean,sd_pct=std,sd_basis='five training seeds' if ratio==1 else 'three subset means'))
    return mean,std

for task in ('window','load'):
    fig,axes=plt.subplots(2,2,figsize=(10.4,8.1))
    fig.subplots_adjust(left=.075,right=.985,top=.97,bottom=.19,hspace=.47,wspace=.19)
    for i,(ax,ratio) in enumerate(zip(axes.flat,ratios)):
        for variant,name,marker in methods:
            values=[value(task,ratio,'awgn_mixed',variant,f'white_{n:+d}' if n>=0 else f'white_{n}') for n in noises]
            y=np.array([x[0] for x in values]); e=np.array([x[1] for x in values]); c=colors[variant]
            ax.errorbar(noises,y,yerr=e,fmt=marker,ms=5,mec=c,mfc='white' if variant=='baseline_data' else c,color=c,lw=1.1,ls=':' if variant=='baseline_data' else '-',capsize=2,elinewidth=.65,zorder=3)
            mixed=value(task,ratio,'awgn_mixed',variant,'clean')
            shift=-.08 if variant=='baseline_data' else .08
            ax.errorbar(6+shift,mixed[0],yerr=mixed[1],fmt=marker,ms=5,mec=c,mfc='white' if variant=='baseline_data' else c,color=c,capsize=2,elinewidth=.65)
        ax.axvline(5.1,color='0.55',ls='--',lw=.7)
        ax.set_xticks([-4,-2,0,2,4,6],['−4','−2','0','2','4','Clean'])
        ax.set_xlim(-4.6,7.15); ax.set_ylim(0 if task=='load' else 70,103)
        ax.set_yticks(np.arange(0,101,20) if task=='load' else [70,80,90,100])
        ax.tick_params(direction='in',top=True,right=True,length=3)
        ax.set_xlabel('Test SNR (dB)'); ax.set_ylabel('Balanced accuracy (%)')
        ax.text(.5,-.26,f'({chr(97+i)}) Training ratio = {int(ratio*100)}%',transform=ax.transAxes,ha='center',va='top',fontsize=13)
        handles=[Line2D([0],[0],color=colors['baseline_data'],ls=':',marker='o',markerfacecolor='white',ms=5,label='AWGN train: data-only'),Line2D([0],[0],color=colors['data_aux_ce'],ls='-',marker='>',ms=5,label='AWGN train: fusion')]
        ax.legend(handles=handles,loc='lower right' if task=='window' else 'lower left',fontsize=9,frameon=True,fancybox=False,edgecolor='black',framealpha=.95)
    fig.text(.5,.025,'Five AWGN levels; clean-test points are separate. Error bars: 10–50%: subset SD; 100%: seed SD.',ha='center',fontsize=10)
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/'media'/f'noise_{task}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
