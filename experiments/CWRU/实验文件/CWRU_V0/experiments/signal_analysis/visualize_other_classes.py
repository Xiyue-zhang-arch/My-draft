"""Read-only descriptive plots for normal, inner-race and rolling-element signals."""
from visualize_signal_quality import (
    Path, argparse, json, np, pd, plt, loadmat, PROJECT, U, Evidence,
    analyze, overview, all_bands, output_directory,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    out=output_directory(parser.parse_args().output)
    U.configure()
    evidence=Evidence(U.CFG)
    records=pd.read_csv(PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_1/awgn_mixed/records.csv')
    data_root=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data'
    specifications=[('Normal',1.,'shaft'),
                    *[(f'IR{s}',5.4152,'BPFI') for s in ['007','014','021']],
                    *[(f'B{s}',order,name) for s in ['007','014','021']
                      for order,name in [(2.35675,'BSF'),(4.7135,'2BSF')]]]
    peaks=[];metrics=[]
    for family,order,reference in specifications:
        items=[]
        for load in range(4):
            record=records[records.record_id==f'{family}_{load}'].iloc[0]
            a=analyze(record,evidence,data_root,order,reference,window_count=2 if family=='IR014' else 10)
            items.append(a)
            for target,table in [(peaks,a['contrast']),(metrics,a['metrics'])]:
                target.append(table.assign(family=family,reference=reference,reference_order=order,
                                           rpm_source=str(record.rpm_source)))
        tag=f'{family}_{reference}'
        overview(items,tag,out)
        all_bands(items,tag,out)
    p=pd.concat(peaks,ignore_index=True)
    m=pd.concat(metrics,ignore_index=True)
    s=m.groupby(['family','reference','load_hp','band'],as_index=False)[
        ['harmonic_contrast_db','periodic_recurrence']].mean()
    p.to_csv(out/'harmonic_peak_background.csv',index=False)
    m.to_csv(out/'window_metrics.csv',index=False)
    s.to_csv(out/'quality_summary.csv',index=False)

    # Display all three sizes instead of selecting a visually favorable record.
    for size in ['007','014','021']:
        fig,axes=plt.subplots(4,4,figsize=(14,9),sharex=True,sharey=True)
        limit=0.
        for load in range(4):
            for col,family in enumerate(['Normal',f'IR{size}',f'OR{size}',f'B{size}']):
                record=records[records.record_id==f'{family}_{load}'].iloc[0]
                raw=loadmat(data_root/f'{record.record_id}.mat',variable_names=[record.selected_variable])[
                    record.selected_variable].ravel()[:24000].astype(np.float32)
                x=U.normalize(raw.reshape(1,1,-1)).ravel()[:4800]
                limit=max(limit,float(np.abs(x).max()))
                ax=axes[load,col]
                ax.plot(np.arange(len(x))/48000,x,lw=.55,color='#2266aa')
                ax.set(title=f'{family}, {load} HP',xlabel='Time (s)')
                if col==0:ax.set_ylabel('Standardized amplitude')
        axes[0,0].set(ylim=(-limit*1.05,limit*1.05),xlim=(0,.1))
        fig.suptitle(f'First 0.1 s of the first 0.5 s window; fault size {int(size)/1000:.3f} inch\n'
                     'Same time/amplitude axes within figure; each full window standardized before display')
        fig.tight_layout(rect=[0,0,1,.94])
        fig.savefig(out/f'waveforms_{size}.png',dpi=180)
        plt.close(fig)

    settings=dict(reference_orders=dict(shaft=1.,BPFI=5.4152,BSF=2.35675,twice_BSF=4.7135),
                  bands_hz=U.CFG['bands_hz'],overview_band_hz=[3000,6000],
                  samples_per_window=24000,windows_per_record='IR014: 2; other families: 10',
                  interval='IR014: first 1 second at every load; others: first 5 seconds',
                  unique_records_for_evidence=28,ball_references='BSF and 2BSF, both reported; no best-reference selection',
                  normal_reference='shaft speed only; not a fault frequency',
                  input='clean DE vibration; no added noise',
                  rpm='records.csv rpm and rpm_source retained in metric tables',
                  envelope_fs=3000,edge_trim_seconds=.025,effective_duration_seconds=.45,
                  harmonic_count=6,peak_search='max(2Hz,1.5% target)',
                  background='max(12Hz,15% target) neighborhood, excluding peak tolerance plus 2Hz',
                  recurrence='mean local-max normalized autocorrelation around periods 1,2,3; max(1 sample,2% lag)',
                  uncertainty='window IQR, descriptive only; not independent repeats or confidence intervals')
    (out/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# 正常、内圈与滚动体信号可视化

本次补充正常类以及内圈、滚动体的007、014、021三种尺寸，覆盖0至3 HP，共28条不同记录。IR014_0所选通道只有63788点，按记录元数据中的48kHz计算约1.33秒。因此IR014四个负载统一只取前两个0.5秒不重叠窗口，即前1秒；其余记录取前10个窗口，即前5秒。不同记录合计248窗口。没有补齐、复制或重采样来增加窗口数。滚动体同时按BSF和2BSF报告，这两套分析使用同一批信号，不是新增独立样本。只读原始数据，不训练模型，不修改冻结结果，不向模型增加质量指标。

## 阅读规则

主图统一用3至6kHz固定带通。每图四行分别为0至3HP，三列分别为包络谱、包络自相关和连续窗口自相关热图。蓝线是所取窗口的中位数，阴影是四分位区间，不是置信区间。IR014只有两个窗口，其区间不具有稳定的分布估计意义，也不足以评价长期周期稳定性。红线标记参考频率的前六阶谐波或前三个参考周期。橙色及灰色区域分别标记第一阶峰搜索区和局部背景区。所有主图使用相同纵轴、颜色尺度和无量纲横轴范围，但不同参考频率对应的实际Hz和秒不同，IR014热图时间范围也更短。

内圈使用BPFI。滚动体分别使用BSF及2BSF，不能依据哪个图更清楚再只报告那个。正常类没有故障频率，图中shaft仅为转频及其周期，出现周期成分也不能作为故障证据。不同参考下的平均峰背景指标不宜直接作为类别质量排名，特别是正常类和故障类不能用它建立故障阈值。

每类同时提供all_bands图展示全部四个既有固定频带，不按结果选择最佳频带。峰背景比为搜索区最大功率除以局部背景几何均值，再取10log10。周期重复指标是三个理论延迟邻域的最大归一化相关的平均，不能单独证明稀疏冲击，更不能仅据此证明分类器使用了这些证据。

waveforms图额外展示四类时域波形。007、014、021全部展示，每条记录固定显示第一个0.5秒窗口的前0.1秒。先按完整窗口去均值并除以标准差，再绘制，纵轴不是物理加速度，不能比较不同负载的绝对振动强度。每张图使用共同时间轴和幅度轴。

## 证据边界

本分析是干净信号的描述性观察，不支持人工加噪抗噪性结论。不同窗口来自同一记录，不是独立采集重复。各负载的转速、传递特性和记录条件也不同，因此不能将曲线差异直接归因为负载。参考频率依赖元数据中的rpm，rpm_source已保留在数值表中，估计转速的误差也会影响对齐。0.45秒有效包络导致约2.22Hz频率间隔，弱峰、邻近峰以及局部搜索最大值需要谨慎解释。某个固定频带内理论证据弱，不等于该类无故障或整条记录没有可辨识信息。

本次沿用之前外圈可视化的处理规则。增加BSF与2BSF展示是对滚动体理论参考的完整呈现，没有修改训练特征、选择规则或模型。

## 当前图中的观察

在统一3至6kHz频带下，IR007的BPFI附近谐波峰较突出，六阶平均峰背景比在各负载为18.20至21.45dB，前三个周期邻域平均相关为0.339至0.562。图中理论周期附近的相关峰也在多个窗口持续出现。这支持该频带内存在与BPFI一致的重复调制结构，但不是分类器使用该结构的证明。

B类的理论对齐证据并不都强。B007在BSF参考下平均峰背景比为3.39至4.90dB，周期相关接近零；B014在2BSF参考下平均周期相关为0.098至0.152，且后续周期峰和跨窗口一致性弱于IR007。不能将这些差异直接解释为滑移、故障程度或抗噪性差异，其他固定频带也需要同时阅读。

正常类在约四倍转频处有明显包络谱峰，自相关也有稳定周期结构。这说明有周期性本身不足以判断故障，必须同时核对周期对应的频率、谐波分布和其他证据。当前图未确定该正常周期成分的具体激励来源。

## 图表索引

'''
    for family,order,reference in specifications:
        tag=f'{family}_{reference}'
        report+=f'- [{tag} 主图]({tag}_evidence_overview.png)，[全部频带]({tag}_all_bands.png)\n'
    for size in ['007','014','021']:
        report+=f'- [四类时域波形，{size}尺寸](waveforms_{size}.png)\n'
    report+='\n详细数值见quality_summary.csv、window_metrics.csv和harmonic_peak_background.csv，处理参数见settings.json。\n'
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(s[s.band==1].round(3).to_string(index=False))
    print('Completed:',out)


if __name__=='__main__':
    main()
