# 单一来源负载迁移，共同73窗，全部40记录

完成80个随机初始化模型，240组模型与目标评价，3240组逐次指标，168组方向混淆计数。未排除任何原始记录，未开展小样本实验。

## 设计与公平性

每次仅一个来源负载参与训练和验证，目标完整负载只在80模型全部完成选模后测试。12个方向。同一来源权重直接测试另外三个目标，不按目标重新训练或选模。来源内部记录随机80/20后抽取共同73训练窗，类别配额8/17/24/24。验证保留19/39/40/40，跨来源验证预算未匹配，不能声称全部条件完全匹配。

|source_HP|candidate_train|selected_train|validation|unused_source_train|train_records|validation_records|
|---|---|---|---|---|---|---|
|0|73|73|19|0|10|10|
|1|156|73|39|83|10|10|
|2|160|73|40|87|10|10|
|3|160|73|40|87|10|10|

两个模型、两协议、5种子1706至1710共享相同来源73窗。固定600更新、每步32观测、每50验证、无提前停止。标准化仅拟合当前训练集，物理特征在线提取。网络参数、优化器、抽样及固定选模规则见[协议](PROTOCOL.md)。

clean_only不人工加噪，只评价干净目标。awgn_mixed每次出现独立50%干净，否则AWGN SNR连续U[-4,4]dB，单批一次CE。验证用来源类别/记录均衡NLL，干净权重0.5、各噪声档0.1。AWGN测试五噪声档各5实例，各模型共享目标观测。

## 12方向等权汇总

先平均每模型噪声实例，再平均每方向5种子，最后12方向等权平均。主指标均衡准确率，保留窗口ACC和宏F1。下表SD为方向均值之间的样本标准差，不是独立轴承置信区间。

|协议|模型|条件|窗口ACC均值±方向SD (%)|均衡ACC (%)|宏F1 (%)|
|---|---|---|---:|---:|---:|
|awgn_mixed|data_only|clean|82.52 ± 11.27|85.47|83.34|
|awgn_mixed|data_only|white_+0|82.53 ± 11.12|85.41|83.16|
|awgn_mixed|data_only|white_+2|82.70 ± 11.22|85.60|83.40|
|awgn_mixed|data_only|white_+4|82.73 ± 11.24|85.66|83.49|
|awgn_mixed|data_only|white_-2|81.97 ± 11.11|84.74|82.44|
|awgn_mixed|data_only|white_-4|80.65 ± 11.15|83.17|80.71|
|awgn_mixed|fusion|clean|94.20 ± 4.70|95.22|94.87|
|awgn_mixed|fusion|white_+0|87.67 ± 6.21|89.93|89.36|
|awgn_mixed|fusion|white_+2|88.06 ± 6.43|90.27|89.66|
|awgn_mixed|fusion|white_+4|88.38 ± 6.53|90.53|89.89|
|awgn_mixed|fusion|white_-2|86.96 ± 6.15|89.23|88.77|
|awgn_mixed|fusion|white_-4|85.54 ± 6.32|87.50|87.33|
|clean_only|data_only|clean|80.31 ± 12.47|82.61|80.87|
|clean_only|fusion|clean|97.60 ± 1.98|97.82|97.80|

## 各方向干净及强噪声结果

|来源HP|目标HP|协议|模型|条件|ACC均值±种子SD (%)|均衡ACC (%)|
|---:|---:|---|---|---|---:|---:|
|0|1|awgn_mixed|data_only|clean|78.87 ± 7.93|82.72|
|0|1|awgn_mixed|data_only|white_-4|79.12 ± 10.91|82.95|
|0|1|awgn_mixed|fusion|clean|93.64 ± 3.90|94.83|
|0|1|awgn_mixed|fusion|white_-4|81.93 ± 3.59|84.30|
|0|1|clean_only|data_only|clean|71.38 ± 11.12|76.53|
|0|1|clean_only|fusion|clean|98.97 ± 0.89|99.17|
|0|2|awgn_mixed|data_only|clean|80.80 ± 4.35|84.00|
|0|2|awgn_mixed|data_only|white_-4|77.38 ± 9.24|81.15|
|0|2|awgn_mixed|fusion|clean|94.60 ± 1.39|95.50|
|0|2|awgn_mixed|fusion|white_-4|81.34 ± 3.42|82.08|
|0|2|clean_only|data_only|clean|71.90 ± 12.87|76.25|
|0|2|clean_only|fusion|clean|97.40 ± 1.24|97.83|
|0|3|awgn_mixed|data_only|clean|75.90 ± 3.93|79.92|
|0|3|awgn_mixed|data_only|white_-4|77.16 ± 7.16|80.80|
|0|3|awgn_mixed|fusion|clean|93.90 ± 2.16|94.92|
|0|3|awgn_mixed|fusion|white_-4|80.02 ± 3.70|80.25|
|0|3|clean_only|data_only|clean|71.90 ± 5.33|75.58|
|0|3|clean_only|fusion|clean|96.60 ± 0.89|97.17|
|1|0|awgn_mixed|data_only|clean|82.61 ± 2.03|86.12|
|1|0|awgn_mixed|data_only|white_-4|74.09 ± 1.93|76.17|
|1|0|awgn_mixed|fusion|clean|91.30 ± 3.77|92.73|
|1|0|awgn_mixed|fusion|white_-4|80.78 ± 2.77|84.66|
|1|0|clean_only|data_only|clean|76.96 ± 5.82|78.52|
|1|0|clean_only|fusion|clean|96.96 ± 0.91|97.06|
|1|2|awgn_mixed|data_only|clean|98.90 ± 1.02|99.08|
|1|2|awgn_mixed|data_only|white_-4|96.48 ± 0.48|97.00|
|1|2|awgn_mixed|fusion|clean|99.60 ± 0.42|99.67|
|1|2|awgn_mixed|fusion|white_-4|92.74 ± 2.14|93.92|
|1|2|clean_only|data_only|clean|99.90 ± 0.22|99.92|
|1|2|clean_only|fusion|clean|99.60 ± 0.22|99.67|
|1|3|awgn_mixed|data_only|clean|86.40 ± 6.90|88.67|
|1|3|awgn_mixed|data_only|white_-4|86.12 ± 3.02|88.37|
|1|3|awgn_mixed|fusion|clean|98.40 ± 0.82|98.67|
|1|3|awgn_mixed|fusion|white_-4|91.68 ± 1.84|92.70|
|1|3|clean_only|data_only|clean|86.20 ± 5.01|88.50|
|1|3|clean_only|fusion|clean|97.90 ± 1.08|98.25|
|2|0|awgn_mixed|data_only|clean|80.00 ± 7.11|83.45|
|2|0|awgn_mixed|data_only|white_-4|73.43 ± 5.88|74.63|
|2|0|awgn_mixed|fusion|clean|95.00 ± 1.24|95.56|
|2|0|awgn_mixed|fusion|white_-4|87.26 ± 1.40|89.47|
|2|0|clean_only|data_only|clean|75.43 ± 8.78|74.95|
|2|0|clean_only|fusion|clean|96.52 ± 1.19|96.06|
|2|1|awgn_mixed|data_only|clean|99.90 ± 0.23|99.92|
|2|1|awgn_mixed|data_only|white_-4|98.15 ± 1.45|98.47|
|2|1|awgn_mixed|fusion|clean|98.26 ± 1.00|98.58|
|2|1|awgn_mixed|fusion|white_-4|93.21 ± 1.89|94.47|
|2|1|clean_only|data_only|clean|98.77 ± 2.01|98.91|
|2|1|clean_only|fusion|clean|99.59 ± 0.23|99.67|
|2|3|awgn_mixed|data_only|clean|85.10 ± 4.26|87.58|
|2|3|awgn_mixed|data_only|white_-4|88.68 ± 2.74|90.57|
|2|3|awgn_mixed|fusion|clean|94.90 ± 1.67|95.75|
|2|3|awgn_mixed|fusion|white_-4|92.38 ± 0.93|93.65|
|2|3|clean_only|data_only|clean|88.30 ± 1.44|90.25|
|2|3|clean_only|fusion|clean|99.20 ± 0.27|99.33|
|3|0|awgn_mixed|data_only|clean|55.65 ± 2.92|62.48|
|3|0|awgn_mixed|data_only|white_-4|56.48 ± 3.55|63.29|
|3|0|awgn_mixed|fusion|clean|82.61 ± 5.80|86.06|
|3|0|awgn_mixed|fusion|white_-4|73.96 ± 5.04|78.17|
|3|0|clean_only|data_only|clean|55.65 ± 9.01|59.48|
|3|0|clean_only|fusion|clean|92.39 ± 1.72|92.89|
|3|1|awgn_mixed|data_only|clean|80.10 ± 6.82|83.33|
|3|1|awgn_mixed|data_only|white_-4|77.87 ± 9.19|79.52|
|3|1|awgn_mixed|fusion|clean|89.95 ± 3.97|91.83|
|3|1|awgn_mixed|fusion|white_-4|82.17 ± 3.79|85.48|
|3|1|clean_only|data_only|clean|82.67 ± 3.65|85.25|
|3|1|clean_only|fusion|clean|97.44 ± 1.92|97.92|
|3|2|awgn_mixed|data_only|clean|86.00 ± 6.04|88.33|
|3|2|awgn_mixed|data_only|white_-4|82.88 ± 5.05|85.10|
|3|2|awgn_mixed|fusion|clean|98.30 ± 1.30|98.58|
|3|2|awgn_mixed|fusion|white_-4|88.96 ± 1.23|90.80|
|3|2|clean_only|data_only|clean|84.60 ± 7.22|87.17|
|3|2|clean_only|fusion|clean|98.60 ± 0.65|98.83|

## 融合较低的条件

下表列出均衡准确率的负增益，单位为百分点。完整窗口ACC差异和逐种子差异分别保存在paired_direction_differences.csv和paired_seed_differences.csv，不筛除不利方向。

|source_load_hp|target_load_hp|protocol|condition|acc_delta|balanced_acc_delta|
|---|---|---|---|---|---|
|0|3|awgn_mixed|white_-4|2.86|-0.55|
|1|2|awgn_mixed|white_+0|-2.24|-1.8667|
|1|2|awgn_mixed|white_+2|-2.12|-1.7667|
|1|2|awgn_mixed|white_+4|-1.92|-1.6|
|1|2|awgn_mixed|white_-2|-2.44|-2.0333|
|1|2|awgn_mixed|white_-4|-3.74|-3.0833|
|1|2|clean_only|clean|-0.3|-0.25|
|2|1|awgn_mixed|clean|-1.641|-1.3333|
|2|1|awgn_mixed|white_+0|-4.9436|-4.0121|
|2|1|awgn_mixed|white_+2|-3.8154|-3.1|
|2|1|awgn_mixed|white_+4|-3.1179|-2.5333|
|2|1|awgn_mixed|white_-2|-5.6821|-4.6061|
|2|1|awgn_mixed|white_-4|-4.9436|-3.9939|

## 计算开销

硬件NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch 2.11.0+cu128，CPU线程2，精度CUDA encoder BF16, remainder FP32。

训练batch32，600更新，共19200有放回观测；73窗的一抽样轮为96观测、3步。训练总/轮均时间、验证前向及标准化拟合分别记录。推理batch1、10预热50重复，CUDA同步，含归一化、在线物理提取与网络，排除磁盘和H2D。每来源模型在首个目标计时一次，共80次独立计时，其他目标引用该计时，不称逐目标单独测量。峰值为allocated GPU显存。

|protocol|method|parameters|latency_ms|train_seconds_per_epoch|total_train_seconds|peak_gpu_mib|fit_seconds|validation_forward_seconds|
|---|---|---|---|---|---|---|---|---|
|awgn_mixed|data_only|33964.0|2.4435|0.2443|48.8528|1196.3794|0.0024|1.8615|
|awgn_mixed|fusion|53068.0|7.2645|0.267|53.3987|1197.0542|0.017|2.501|
|clean_only|data_only|33964.0|2.424|0.2156|43.121|1196.3794|0.0024|0.316|
|clean_only|fusion|53068.0|7.2101|0.234|46.8063|1197.0542|0.154|0.6075|

## 证据边界

本实验是人工故障CWRU单一来源负载到另一负载的评价，不是独立轴承或现场泛化证明。训练验证共享来源MAT，只有一个固定内划分和一次固定73窗抽取，5训练种子反映拟合随机性，不反映数据划分的完整不确定性。

不同方向共享来源模型、目标窗口，且不同负载可能使用相同故障试件。12方向及噪声重复不作为新增独立轴承，不据此夸大统计显著性。各来源验证标注预算和类内记录组成不同，不能将方向差异全归因于负载。

全部记录均恢复参与本次实验，但历史测试曝光不消失，本次仍属于已有数据上的后续实验。两模型对照同时改变信息、结构和参数量，不能独立证明物理因果机理。合成白噪声不能代表所有现场噪声。历史实验的训练数量、验证及停止规则不同，不用于严格等条件排名。

[数据数量与原始MAT使用说明](DATA_USAGE.md) · [逐方向完整指标](per_direction_summary.csv) · [逐种子结果](per_seed_direction.csv)

## 图表

![direction_awgn_mixed_white_+4.png](figures/direction_awgn_mixed_white_+4.png)

![direction_awgn_mixed_white_+2.png](figures/direction_awgn_mixed_white_+2.png)

![direction_awgn_mixed_white_+0.png](figures/direction_awgn_mixed_white_+0.png)

![direction_awgn_mixed_white_-2.png](figures/direction_awgn_mixed_white_-2.png)

![direction_awgn_mixed_white_-4.png](figures/direction_awgn_mixed_white_-4.png)

![direction_awgn_mixed_clean.png](figures/direction_awgn_mixed_clean.png)

![direction_clean_only_clean.png](figures/direction_clean_only_clean.png)

![confusion_clean_only_fusion_clean.png](figures/confusion_clean_only_fusion_clean.png)

![confusion_clean_only_data_only_clean.png](figures/confusion_clean_only_data_only_clean.png)

![confusion_awgn_mixed_fusion_clean.png](figures/confusion_awgn_mixed_fusion_clean.png)

![confusion_awgn_mixed_fusion_white_-4.png](figures/confusion_awgn_mixed_fusion_white_-4.png)

![confusion_awgn_mixed_fusion_white_-2.png](figures/confusion_awgn_mixed_fusion_white_-2.png)

![confusion_awgn_mixed_fusion_white_+0.png](figures/confusion_awgn_mixed_fusion_white_+0.png)

![confusion_awgn_mixed_fusion_white_+2.png](figures/confusion_awgn_mixed_fusion_white_+2.png)

![confusion_awgn_mixed_fusion_white_+4.png](figures/confusion_awgn_mixed_fusion_white_+4.png)

![confusion_awgn_mixed_data_only_clean.png](figures/confusion_awgn_mixed_data_only_clean.png)

![confusion_awgn_mixed_data_only_white_-4.png](figures/confusion_awgn_mixed_data_only_white_-4.png)

![confusion_awgn_mixed_data_only_white_-2.png](figures/confusion_awgn_mixed_data_only_white_-2.png)

![confusion_awgn_mixed_data_only_white_+0.png](figures/confusion_awgn_mixed_data_only_white_+0.png)

![confusion_awgn_mixed_data_only_white_+2.png](figures/confusion_awgn_mixed_data_only_white_+2.png)

![confusion_awgn_mixed_data_only_white_+4.png](figures/confusion_awgn_mixed_data_only_white_+4.png)
