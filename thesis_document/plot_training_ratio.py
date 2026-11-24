from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(r'D:\Project\MY draft')
DATA=ROOT/'experiments/CWRU/实验结果'/'CWRU_V3'
DEST=ROOT/'thesis_document'/'成果'/'V3训练比例对比图'
DEST.mkdir(parents=True,exist_ok=True)
random_dir=DATA/'window_random_20261023_0853'/'evaluation'
small_dir=DATA/'load_small_20261025_2017'/'evaluation'
full_dir=DATA/'load_full_20261024_0950'/'evaluation'
random_small=pd.read_csv(random_dir/'between_subset_summary.csv')
random_full=pd.read_csv(random_dir/'mean_std.csv')
load_small=pd.read_csv(small_dir/'transfer_between_subset_summary.csv')
load_full=pd.read_csv(full_dir/'transfer_equal_direction_mean_std.csv')
conditions=[('clean_only','clean','(a) Clean training / Clean test'),('awgn_mixed','clean','(b) AWGN training / Clean test'),('awgn_mixed','white_-4','(c) AWGN training / -4 dB test')]
variants=[('baseline_data','Data-only branch','#292929','o'),('data_aux_ce','Fusion model','#ad1733','*')]
ratios=[.1,.2,.5,1.0]
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
all_rows=[]
for task,small,full in [('Random_window',random_small,random_full),('Cross_load',load_small,load_full)]:
    for metric,label in [('acc','Accuracy (%)'),('balanced_acc','Balanced accuracy (%)')]:
        fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
        fig.subplots_adjust(left=.075,right=.99,bottom=.26,top=.88,wspace=.13)
        for ax,(protocol,condition,panel) in zip(axs,conditions):
            for variant,name,color,marker in variants:
                means=[]; errors=[]
                for ratio in ratios:
                    source=full if ratio==1 else small
                    row=source[(source.ratio==ratio)&(source.protocol==protocol)&(source.condition==condition)&(source.variant==variant)]
                    assert len(row)==1,(task,metric,ratio,protocol,condition,variant,len(row))
                    v=row.iloc[0]; mean=float(v[metric+'_mean'])*100; sd=float(v[metric+'_std'])*100
                    means.append(mean); errors.append(sd)
                    all_rows.append({'experiment':task,'metric':metric,'protocol':protocol,'condition':condition,'variant':variant,'training_ratio':ratio,'mean_percent':mean,'sd_percent':sd,'sd_basis':'five training seeds' if ratio==1 else 'three subset means'})
                ax.errorbar([10,20,50,100],means,yerr=errors,label=name,color=color,marker=marker,markersize=7 if marker=='o' else 11,linewidth=1.6,capsize=3,elinewidth=1.0)
            ax.set_xticks([10,20,50,100]); ax.set_xlim(5,103)
            ax.set_title(panel,fontsize=11,pad=9)
            ax.grid(True,axis='both',color='#dedede',linewidth=.7); ax.tick_params(direction='in',top=True,right=True)
            ax.set_xlabel('Training sample ratio (%)',fontsize=11)
            if task=='Random_window': ax.set_ylim(65,102); ax.set_yticks([70,75,80,85,90,95,100])
            else: ax.set_ylim(0,105); ax.set_yticks([0,20,40,60,80,100])
        axs[0].set_ylabel(label,fontsize=12)
        axs[-1].legend(loc='lower right',frameon=True,fancybox=False,edgecolor='#777777',framealpha=1,fontsize=9)
        fig.text(.5,.13,'Error bars: 10–50% = SD of 3 subset means; 100% = SD of 5 training seeds.',ha='center',fontsize=10)
        fig.text(.5,.075,'Noise replicates are averaged before aggregation; cross-load results weight 12 directions equally.' if task=='Cross_load' else 'Noise replicates are averaged before aggregation; train/test windows share original records.',ha='center',fontsize=9)
        fig.savefig(DEST/f'{task}_{metric}.png',dpi=300,facecolor='white')
        fig.savefig(DEST/f'{task}_{metric}.svg',facecolor='white')
        plt.close(fig)
pd.DataFrame(all_rows).to_csv(DEST/'训练比例图_数据及误差线口径.csv',index=False,encoding='utf-8-sig')
print('Generated 4 PNG and 4 SVG plots with 96 source rows:',DEST)
