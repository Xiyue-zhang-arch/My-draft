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
OUT=ROOT/'thesis_document'/'成果'/'V3_毕业论文_噪声补充_20261031'
OUT.mkdir(parents=True,exist_ok=True)
shutil.copytree(SRC/'media',OUT/'media',dirs_exist_ok=True)

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
            clean=value(task,ratio,'clean_only',variant,'clean')
            shift=-.08 if variant=='baseline_data' else .08
            ax.errorbar(6+shift,mixed[0],yerr=mixed[1],fmt=marker,ms=5,mec=c,mfc='white' if variant=='baseline_data' else c,color=c,capsize=2,elinewidth=.65)
            ax.errorbar(6.55+shift,clean[0],yerr=clean[1],fmt='x' if variant=='baseline_data' else '*',ms=7,color=c,capsize=2,elinewidth=.65)
        ax.axvline(5.1,color='0.55',ls='--',lw=.7)
        ax.set_xticks([-4,-2,0,2,4,6.27],['−4','−2','0','2','4','Clean'])
        ax.set_xlim(-4.6,7.15); ax.set_ylim(0 if task=='load' else 70,103)
        ax.set_yticks(np.arange(0,101,20) if task=='load' else [70,80,90,100])
        ax.tick_params(direction='in',top=True,right=True,length=3)
        ax.set_xlabel('Test SNR (dB)'); ax.set_ylabel('Balanced accuracy (%)')
        ax.text(.5,-.26,f'({chr(97+i)}) Training ratio = {int(ratio*100)}%',transform=ax.transAxes,ha='center',va='top',fontsize=13)
        handles=[Line2D([0],[0],color=colors['baseline_data'],ls=':',marker='o',markerfacecolor='white',ms=5,label='AWGN train: data-only'),Line2D([0],[0],color=colors['data_aux_ce'],ls='-',marker='>',ms=5,label='AWGN train: fusion'),Line2D([0],[0],color=colors['baseline_data'],ls='None',marker='x',ms=6,label='Clean train: data-only'),Line2D([0],[0],color=colors['data_aux_ce'],ls='None',marker='*',ms=7,label='Clean train: fusion')]
        ax.legend(handles=handles,loc='lower right' if task=='window' else 'lower left',fontsize=9,frameon=True,fancybox=False,edgecolor='black',framealpha=.95)
    fig.text(.5,.025,'Five AWGN levels; clean-test points are separate. Error bars: 10–50%: subset SD; 100%: seed SD.',ha='center',fontsize=10)
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/'media'/f'noise_{task}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)
pd.DataFrame(records).to_csv(OUT/'noise_figure_source.csv',index=False,encoding='utf-8-sig')

tex=(SRC/'thesis.tex').read_text(encoding='utf-8')
for old,new in [(6,8),(5,6),(4,5),(3,4)]:
    tex=tex.replace(f'图5-{old}',f'图5-临{new}')
tex=tex.replace('图5-临','图5-')
def figure(task,number,topic):
    return r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{media/noise_'''+task+r'''.pdf}
\par\small 图5-'''+str(number)+' '+topic+r'''在不同训练比例和噪声条件下的均衡准确率
\end{figure}
'''
noise_note=r'''在图中，横轴左侧为五个有噪测试信噪比，右侧 Clean 为单独的干净测试条件，并非有限数值的信噪比。连续曲线仅表示 AWGN 混合训练模型在五档噪声下的结果，右侧叉号和星号分别表示纯干净训练的两个模型在干净测试下的结果。10\%、20\%和50\%训练预算的误差线是三个子集均值之间的样本标准差；全样本为五个训练种子之间的样本标准差。噪声实例先在同一模型内平均，不能视为独立测试样本。纯干净训练模型没有进行上述五档噪声测试，因此不能从该图推断两种训练方式的抗噪差异。
'''
for anchor,extra in [
('图5-2 随机窗口不同训练比例下的均衡准确率',figure('window',3,'记录内随机窗口')+noise_note+'记录内随机窗口10\\%预算下，AWGN训练的单分支在−4 dB和+4 dB时分别为82.66\\%和83.36\\%，融合方案分别为94.72\\%和96.09\\%。两条曲线在五个噪声水平下均有差距。全样本+4 dB时单分支为99.33\\%，融合方案为99.17\\%，因此不宜认为融合方案在所有预算及条件下都更高。\n'),
('图5-6 不同来源训练比例下12方向等权均衡准确率',figure('load',7,'跨负载实验')+'跨负载图中的每个点先对目标方向等权汇总，再按相应训练预算统计种子或子集波动。来源训练窗口数和验证标签预算在不同负载之间并不相同，因此该图反映既定协议下的整体表现，不代表独立设备上的泛化能力。\n'+noise_note+'在10\\%来源预算下，AWGN训练的单分支从−4 dB的36.17\\%升至+4 dB的38.20\\%，融合方案由85.96\\%升至86.98\\%。在全样本条件下，单分支对应84.31\\%和86.49\\%，融合方案对应89.51\\%和91.00\\%。这些平均结果不能排除个别迁移方向或类别退化。\n')]:
    target=r'\begin{center}'+anchor+r'\end{center}'
    assert tex.count(target)==1,(anchor,tex.count(target))
    tex=tex.replace(target,target+'\n\n'+extra)
(OUT/'thesis.tex').write_text(tex,encoding='utf-8')
print('Saved:',OUT,'rows:',len(records))
