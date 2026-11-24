# 四方法比较

训练入口为[matched/train_matched_comparison.py](matched/train_matched_comparison.py)，使用说明见[统一说明](../../README.md)。

本目录两个网络文件定义WDCNN、DRSN-CW、PG-RDGFN-adapted及AW_Hybrid_adapted。PROTOCOL.md和HYBRID_PROTOCOL.md只说明方法来源与适配选择。当前共同设置见matched/PROTOCOL.md。新训练结果与已有冻结报告分开保存。


## 数据集使用情况

[集合数量、原始文件及各训练子集用量](DATA_USAGE.md)。训练候选与实际用量分开记录，噪声、模型和频带重复不增加独立数据。
