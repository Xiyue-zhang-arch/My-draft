# 排除OR014_3后的固定212维从头训练

[正式协议](PROTOCOL.md) · [本轮结果](../../../../实验结果/CWRU/experiment_4/REPORT.md)

原直接筛选旧预测的主结果及入口已清除。当前主实验重新随机初始化66个模型，覆盖纯干净和非配对AWGN两协议。每协议包含全样本与10%、20%、50%小样本融合模型，以及全样本单数据分支。训练287窗口、验证200窗口不变，只排除测试记录OR014_3，测试180窗口（20/60/40/60）。

从项目根目录运行，输出必须为新目录。

```powershell
python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_4/run_exclusion.py --output experiments/CWRU/实验结果/CWRU/新的排除训练结果
python experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_4/infer_exclusion.py --model 新模型目录/model.pt --input 待推理.npz --output experiments/CWRU/实验结果/CWRU/新的推理结果
```

infer.py也可按protocol、method、ratio、subset、seed选择本轮主结果中的模型。它不再加载原完整200窗口实验的冻结权重。原始数据只读；旧306模型、方法比较及信号可视化不变。历史附加6模型及其结果已清除，当前仅保留本轮66模型。

重新训练不能消除事后排除性质。完整200窗口结果仍需保留作为主要评价，不将删除测试记录解释为模型改善。

本轮66模型和891组评价已完成并冻结，保存35组汇总混淆矩阵及6张汇总图。原始计数和各类召回率保留。


## 数据集使用情况

[集合数量、原始文件及各训练子集用量](DATA_USAGE.md)。训练候选与实际用量分开记录，噪声、模型和频带重复不增加独立数据。
