# 故障诊断项目

服务器的统一入口为 `python run_experiments.py list`。环境配置、数据依赖和手动运行示例见[服务器运行说明](docs/服务器运行.md)。

## 当前目录入口

| 目录 | 用途 |
| --- | --- |
| [experiments](experiments/README.md) | CWRU 和 PU 的实验文件、实验协议及正式结果 |
| [data](data/README.md) | 原始数据与登记的共享输入 |
| [docs](docs/) | 文献核查、方法设计与后续实验对照约定 |
| [thesis_document](thesis_document/) | 论文源材料、制图脚本与[论文成果](thesis_document/成果/) |

## 现存实验

- **CWRU** 保留 V0 至 V6 七套结果。各版本设置和状态见 [CWRU 实验说明](experiments/CWRU/实验说明.md)，权重、预测和报告在 [CWRU 实验结果](experiments/CWRU/实验结果/)。
- **PU** 保留四工况留出、10% 小样本及类别不平衡、直接拼接、四谐波等权平均四项实验。样本划分和证据边界见 [PU 实验说明](experiments/PU/实验说明.md)，正式输出在 [PU 实验结果](experiments/PU/实验结果/)。

各批次是否完成，以其结果内的配置、完成标记和报告为准。目录约定、执行要求与早期实验记录见 [AGENTS.md](AGENTS.md)。

## CWRU V0 早期实验索引

以下是 V0 的六项实验及信号分析。训练和推理入口见 [V0 使用说明](experiments/CWRU/实验文件/CWRU_V0/README.md)。

|编号|实验内容|主要设置|模型数|结果入口|数据使用说明|
|---|---|---|---:|---|---|
|实验1|原纯干净与原AWGN混合训练|按负载划分，保留OR014_3，训练287、验证200、测试200窗；内部保留两套协议|66|[总入口](experiments/CWRU/实验结果/CWRU/experiment_1/README.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/experiment_1/DATA_USAGE.md)|
|实验2|四方法匹配比较|WDCNN、DRSN-CW、PG-RDGFN-adapted、AW_Hybrid_adapted，两套协议|240|[正式报告](experiments/CWRU/实验结果/CWRU/experiment_2/matched/REPORT.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/experiment_2/DATA_USAGE.md)|
|实验3|先切窗再随机划分|记录内70%/15%/15%，训练467、验证100、测试100窗|120|[原报告](experiments/CWRU/实验结果/CWRU/experiment_3/REPORT.md) · [小样本对照补充](experiments/CWRU/实验结果/CWRU/experiment_3_data_small/REPORT.md)|[原数据说明](experiments/CWRU/实验结果/CWRU/experiment_3/DATA_USAGE.md) · [补充用量](experiments/CWRU/实验结果/CWRU/experiment_3_data_small/DATA_USAGE.md)|
|实验4|排除OR014_3的按负载实验|训练287、验证200、测试180窗，两协议从头训练|66|[正式报告](experiments/CWRU/实验结果/CWRU/experiment_4/REPORT.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/experiment_4/DATA_USAGE.md)|
|实验5|随机窗口训练预算匹配|训练29/58/144/287窗，验证100、测试100窗|66|[正式报告](experiments/CWRU/实验结果/CWRU/experiment_5/REPORT.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/experiment_5/DATA_USAGE.md)|
|实验6|单一来源负载迁移，一个负载训练、另一负载测试|全部40记录，共同73训练窗，无小样本，12方向、两协议两模型各5种子|80|[正式报告](experiments/CWRU/实验结果/CWRU/experiment_6/REPORT.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/experiment_6/DATA_USAGE.md)|
|信号分析|四类信号描述性分析|正常、内圈、外圈、滚动体的信号和频谱描述|0|[正式报告](experiments/CWRU/实验结果/CWRU/signal_analysis/REPORT.md)|[来源与用量](experiments/CWRU/实验结果/CWRU/signal_analysis/DATA_USAGE.md)|

表中的模型数按当时的训练和引用口径记录。实验 2 引用实验 1 的部分评价，实验 3 并列原结果和补充结果。实验 3、5 的随机窗口划分允许同一原始记录进入不同集合，跨负载证据应查阅各对应协议。
