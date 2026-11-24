"""Read-only signal evidence visualization with fixed processing across loads."""
from pathlib import Path
import sys,json,argparse
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[3]
sys.path.insert(0,str(ROOT/'shared'))
import training_utils as U
from bearing_physics_features import Evidence
from experiment_paths import output_directory


def analyze(record,evidence,data_root,reference_order=3.5848,reference_name='BPFO',window_count=10):
    signal=loadmat(data_root/f'{record.record_id}.mat',variable_names=[record.selected_variable])[record.selected_variable].ravel().astype(np.float32)
    assert len(signal)>=window_count*24000
    x=U.normalize(signal[:window_count*24000].reshape(window_count,1,24000))
    with torch.no_grad():env,centered,power,n,duration=evidence._envelope_spectrum(torch.as_tensor(x))
    env,centered,power=env.numpy(),centered.numpy(),power.numpy()
    frequency=np.fft.rfftfreq(n,1/3000);fcf=float(record.rpm)/60*reference_order
    ac=np.fft.irfft(abs(np.fft.rfft(centered,n=2*n,axis=-1))**2,n=2*n,axis=-1)[...,:n]
    ac=ac/ac[...,:1].clip(1e-16)
    ratios=[];rows=[]
    for band in range(4):
        for window in range(window_count):
            p=power[window,band];harm=[]
            for k in range(1,7):
                target=k*fcf;tol=max(2.,.015*target);distance=abs(frequency-target)
                foreground=distance<=tol
                background=(distance<=max(12.,.15*target))&(distance>tol+2)
                peak=p[foreground].max();base=np.exp(np.log(p[background]).mean())
                peak_frequency=frequency[np.where(foreground)[0][p[foreground].argmax()]]
                contrast=10*np.log10(peak/base)
                rows.append(dict(record_id=record.record_id,load_hp=record.load_hp,rpm=record.rpm,
                                 window=window,start_seconds=window*.5,band=band,
                                 band_low=U.CFG['bands_hz'][band][0],band_high=U.CFG['bands_hz'][band][1],
                                 harmonic=k,theory_hz=target,peak_hz=peak_frequency,
                                 peak_power=float(peak),background_power=float(base),peak_background_db=contrast))
                harm.append(contrast)
            peaks=[]
            for k in range(1,4):
                lag=3000*k/fcf;index=np.arange(n)
                local=(index>0)&(abs(index-lag)<=max(1.,.02*lag))
                peaks.append(ac[window,band,local].max())
            ratios.append(dict(record_id=record.record_id,load_hp=record.load_hp,window=window,
                               band=band,harmonic_contrast_db=float(np.mean(harm)),
                               periodic_recurrence=float(np.mean(peaks)),
                               first_period_recurrence=float(peaks[0])))
    return dict(record=record,frequency=frequency,fcf=fcf,power=power,ac=ac,n=n,reference_name=reference_name,
                contrast=pd.DataFrame(rows),metrics=pd.DataFrame(ratios))


def overview(items,fault,out):
    fig,axes=plt.subplots(4,3,figsize=(13,11))
    reference=items[0].get('reference_name','BPFO')
    for load,a in enumerate(items):
        f=a['frequency']/a['fcf'];p=a['power'][:,1]
        db=10*np.log10(p/p[:,1:].mean(-1,keepdims=True))
        ax=axes[load,0];median=np.median(db,axis=0)
        ax.plot(f,median,color='#2266aa',lw=1)
        ax.fill_between(f,*np.percentile(db,[25,75],axis=0),color='#2266aa',alpha=.18)
        for k in range(1,7):ax.axvline(k,color='#c44035',ls='--',lw=.7)
        # Highlight the first harmonic's peak search and background intervals.
        fc=a['fcf'];tol=max(2.,.015*fc)/fc;outer=max(12.,.15*fc)/fc;inner=tol+2/fc
        for lo,hi in [(1-outer,1-inner),(1+inner,1+outer)]:ax.axvspan(lo,hi,color='gray',alpha=.25)
        ax.axvspan(1-tol,1+tol,color='#f4a642',alpha=.35)
        ax.set(xlim=(.35,6.4),ylim=(-32,32),xlabel=f'Envelope frequency / {reference}',ylabel='Power / mean power (dB)',
               title=f'{load} HP, {a["record"].rpm:.0f} rpm: envelope spectrum')
        lag=np.arange(a['n'])/3000*a['fcf'];acf=a['ac'][:,1]
        ax=axes[load,1];ax.plot(lag,np.median(acf,axis=0),color='#2266aa')
        ax.fill_between(lag,*np.percentile(acf,[25,75],axis=0),color='#2266aa',alpha=.18)
        for k in [1,2,3]:ax.axvline(k,color='#c44035',ls='--',lw=.8)
        ax.set(xlim=(.4,3.5),ylim=(-1,1),xlabel=f'Lag / {reference} period',ylabel='Envelope autocorrelation',title='Repeated peaks at 1, 2, 3 periods?')
        grid=np.linspace(.4,3.5,320)
        heat=np.stack([np.interp(grid,lag,v) for v in acf])
        ax=axes[load,2];im=ax.imshow(heat,extent=[.4,3.5,len(acf)*.5-.25,-.25],aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
        for k in [1,2,3]:ax.axvline(k,color='black',ls='--',lw=.7)
        ax.set(xlabel=f'Lag / {reference} period',ylabel='Window start (s)',title='Do periodic ridges persist across windows?')
    count=len(items[0]['power'])
    fig.suptitle(f'{fault}; reference: {reference}; fixed 3-6 kHz band; first {count} non-overlapping windows per record\nBlue: median; shading: window IQR; red: reference harmonics/periods',fontsize=12)
    fig.tight_layout(rect=[0,0,.90,.95]);bar=fig.add_axes([.918,.12,.012,.72]);fig.colorbar(im,cax=bar,label='Autocorrelation')
    fig.savefig(out/f'{fault}_evidence_overview.png',dpi=180);plt.close(fig)


def all_bands(items,fault,out):
    fig,axes=plt.subplots(4,4,figsize=(14,10),sharex=True,sharey=True)
    reference=items[0].get('reference_name','BPFO')
    for load,a in enumerate(items):
        for band,(lo,hi) in enumerate(U.CFG['bands_hz']):
            p=a['power'][:,band];d=10*np.log10(p/p[:,1:].mean(-1,keepdims=True))
            ax=axes[load,band];ax.plot(a['frequency']/a['fcf'],np.median(d,0),color='#2266aa',lw=1)
            ax.fill_between(a['frequency']/a['fcf'],*np.percentile(d,[25,75],axis=0),color='#2266aa',alpha=.15)
            for k in range(1,7):ax.axvline(k,color='#c44035',ls='--',lw=.6)
            ax.set(xlim=(.35,6.4),ylim=(-32,32),title=f'{load} HP; {lo/1000:g}-{hi/1000:g} kHz')
            if load==3:ax.set_xlabel(f'Frequency / {reference}')
            if band==0:ax.set_ylabel('Power / mean (dB)')
    fig.suptitle(f'{fault}: all four prespecified bands; first {len(items[0]["power"])} windows; median and IQR')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/f'{fault}_all_bands.png',dpi=180);plt.close(fig)


def summary_plot(summary,out):
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for band,(lo,hi) in enumerate(U.CFG['bands_hz']):
        for fault,color in [('OR007','#2171b5'),('OR014','#e6550d'),('OR021','#238b45')]:
            a=summary[(summary.fault==fault)&(summary.band==band)].sort_values('load_hp')
            for row,metric in enumerate(['harmonic_contrast_db','periodic_recurrence']):
                axes[row,band].plot(a.load_hp,a[metric],marker='o',label=fault,color=color)
        axes[0,band].set(title=f'{lo/1000:g}-{hi/1000:g} kHz',ylabel='Mean harmonic peak/background (dB)',ylim=(0,32))
        axes[1,band].set(ylabel='Mean 1-3 period autocorrelation',ylim=(-.1,1))
        for row in [0,1]:axes[row,band].set(xticks=[0,1,2,3],xlabel='Motor load (HP)');axes[row,band].grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Descriptive signal evidence; same fixed processing, no best-band selection')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'quality_across_loads.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    out=output_directory(args.output)
    U.configure();e=Evidence(U.CFG)
    records=pd.read_csv(PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_1/awgn_mixed/records.csv')
    data_root=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data'
    analyses=[]
    for fault in ['OR007','OR014','OR021']:
        items=[]
        for load in range(4):
            record=records[records.record_id==f'{fault}_{load}'].iloc[0]
            a=analyze(record,e,data_root);items.append(a);analyses.append(a)
        overview(items,fault,out);all_bands(items,fault,out)
    peaks=pd.concat([a['contrast'] for a in analyses],ignore_index=True)
    metrics=pd.concat([a['metrics'] for a in analyses],ignore_index=True);metrics['fault']=metrics.record_id.str.split('_').str[0]
    summary=metrics.groupby(['fault','load_hp','band'],as_index=False)[['harmonic_contrast_db','periodic_recurrence']].mean()
    peaks.to_csv(out/'harmonic_peak_background.csv',index=False);metrics.to_csv(out/'window_metrics.csv',index=False);summary.to_csv(out/'quality_summary.csv',index=False)
    summary_plot(summary,out)
    (out/'settings.json').write_text(json.dumps(dict(bands=U.CFG['bands_hz'],samples_per_window=24000,windows_per_record=10,
        interval='first 5 s of each of 12 records',overview_band_hz=[3000,6000],overview_band_selected_before_analysis=True,
        envelope_sample_rate=3000,trim_seconds_each_end=.025,fft_effective_seconds=.45,
        peak_search='max(2Hz,1.5% harmonic frequency)',background='within max(12Hz,15% harmonic frequency), outside peak tolerance+2Hz',
        periodic_search='1,2,3 theoretical periods, tolerance max(1 sample,2% lag)',aggregation='record/window descriptive, no independent-window inference'),ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# 故障峰背景与周期结构可视化

本分析只读OR007、OR014、OR021在0至3HP下的12条记录，每条取前10个0.5秒不重叠窗口，即同样的前5秒。不训练模型，不改变任何冻结实验或原始数据，不向分类器新增质量输入。

## 图怎样阅读

每个evidence_overview图有四行，对应0至3HP。左列是带通包络谱，红线为BPFO的前六阶谐波，橙色为第一阶峰搜索区，灰色为其局部背景区；其余谐波使用相同规则。中列是去均值包络自相关，横轴为理论故障周期的倍数。右列是连续10个窗口的自相关热图，检查理论周期处的高值竖带是否持续存在。所有图共同坐标范围和颜色尺度，蓝线为窗口中位数，阴影为窗口四分位区间，不是置信区间。

主图统一使用事先指定的3至6kHz频带，不能据此声称这是最佳频带。all_bands图展示全部四个既有固定频带，quality_across_loads展示三种故障尺寸在所有频带的描述性统计，避免仅展示最清楚的频带。实际响应可能跨频带，不能依据主图一个频带否定故障。

谱曲线仅按每窗口包络谱平均功率归一化，以便看峰与背景形状。定量PBR按每窗口、每谐波计算，10log10(搜索区峰功率/背景几何均值功率)，与现有谐波特征规则一致。六谐波PBR取平均后，再对10窗口平均。局部背景区排除目标峰及额外2Hz保护区，参数详见settings.json。

自相关以零延迟归一化。定量周期重复为一至三倍理论延迟附近局部最大相关的平均，不是周期稳定性的充分统计量。热图用于观察重复结构是否跨窗口持续。不能把搜索区内最大值大于零单独当作故障证据，也不能把搜索峰的轻微频移直接解释为滑移。

## 证据边界

同一记录中的10个窗口相关，四分位区间仅描述记录内部变化。每个负载只有一条对应记录，不能把窗口当作独立重复来检验负载的因果影响。负载与转速同时改变，频率和延迟按各记录实际转速换算为故障阶次及周期，只部分处理转速差异。当前只分析干净信号，不据此声称人工加噪稳定性。未按分类错误修改参数，不改变OR014_3的实验纳入状态。

![所有固定频带的跨负载对照](quality_across_loads.png)

![OR014的谱峰、自相关与连续窗口](OR014_evidence_overview.png)

![OR007参照](OR007_evidence_overview.png)

![OR021参照](OR021_evidence_overview.png)

详细数值见harmonic_peak_background.csv、window_metrics.csv、quality_summary.csv；所有频带图见各all_bands.png。
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(summary[summary.band==1].round(3).to_string(index=False));print('Completed:',out)


if __name__=='__main__':main()
