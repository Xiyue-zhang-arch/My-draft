# 故障证据的信号可视化

只读比较OR007、OR014、OR021在0至3HP的12条原始记录，各取前10个0.5秒窗口。画包络谱、自相关、连续窗口自相关热图，以及全部四个固定频带的跨负载指标。主图统一使用预设3至6kHz，未按分类结果选择最佳频带。仅是信号分析，不训练或修改模型，不改变原实验的记录纳入标准。

[正式说明与图](../../../../实验结果/CWRU/signal_analysis/REPORT.md)包含设置、数据和解释边界。直接复用现有固定物理分支的包络计算，峰背景与周期重复规则一致。统计仅描述各记录内部，不把10个窗口视为独立轴承重复。

在项目根目录运行，输出必须是新目录。

```powershell
python experiments/CWRU/实验文件/CWRU_V0/experiments/signal_analysis/visualize_signal_quality.py --output experiments/CWRU/实验结果/CWRU/新的信号分析目录
```

补充的[正常、内圈、滚动体图与说明](../../../../实验结果/CWRU/signal_analysis/REPORT.md#正常内圈与滚动体分析)覆盖0至3HP。内圈使用BPFI，滚动体同时展示BSF和2BSF，正常类仅用转频作参照。IR014_0记录较短，因此IR014各负载统一取前两个窗口，其余取前10个窗口。另提供三种故障尺寸的四类时域波形对照。图表不改变模型或训练设置。

```powershell
python experiments/CWRU/实验文件/CWRU_V0/experiments/signal_analysis/visualize_other_classes.py --output experiments/CWRU/实验结果/CWRU/新的其他类别信号分析目录
```

现有两组结果已合并至项目级experiments/CWRU/实验结果/CWRU/signal_analysis，统一入口为REPORT.md。外圈统计表与参数保留原文件名，正常、内圈和滚动体的同名文件增加other_classes_前缀。图像保持原文件名，未重算或跨设置合并统计。两个可视化入口仍用于分别生成新的独立输出目录，不覆盖已有结果。


## 数据集使用情况

[集合数量、原始文件及各训练子集用量](DATA_USAGE.md)。训练候选与实际用量分开记录，噪声、模型和频带重复不增加独立数据。
