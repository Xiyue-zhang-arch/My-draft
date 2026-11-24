# 阶段1至3 训练、物理特征和融合诊断

只读V1权重、训练记录以及训练/验证窗口；未读取目标测试预测。20个基础模型的最佳验证NLL与训练记录一致，基线可运行。

## 验证集结果

| protocol   | method    | condition   |   balanced_acc |   validation_nll |
|:-----------|:----------|:------------|---------------:|-----------------:|
| awgn_mixed | data_only | clean       |       1        |         0.000963 |
| awgn_mixed | data_only | white_+0    |       1        |         0.001821 |
| awgn_mixed | data_only | white_+2    |       1        |         0.001216 |
| awgn_mixed | data_only | white_+4    |       1        |         0.001054 |
| awgn_mixed | data_only | white_-2    |       1        |         0.004722 |
| awgn_mixed | data_only | white_-4    |       0.992188 |         0.015849 |
| awgn_mixed | fusion    | clean       |       1        |         0.000469 |
| awgn_mixed | fusion    | white_+0    |       1        |         0.002674 |
| awgn_mixed | fusion    | white_+2    |       1        |         0.001059 |
| awgn_mixed | fusion    | white_+4    |       1        |         0.000914 |
| awgn_mixed | fusion    | white_-2    |       1        |         0.004359 |
| awgn_mixed | fusion    | white_-4    |       0.996875 |         0.01261  |
| clean_only | data_only | clean       |       1        |         0.000479 |
| clean_only | fusion    | clean       |       1        |         0.000311 |

## 固定验证集物理残差干预

| protocol   | condition   |   residual_multiplier |   balanced_acc |      nll |
|:-----------|:------------|----------------------:|---------------:|---------:|
| awgn_mixed | clean       |                   0   |       0.909429 | 0.277723 |
| awgn_mixed | clean       |                   0.1 |       0.948276 | 0.124868 |
| awgn_mixed | clean       |                   0.5 |       1        | 0.001641 |
| awgn_mixed | clean       |                   1   |       1        | 0.000469 |
| awgn_mixed | white_-4    |                   0   |       0.732328 | 0.753669 |
| awgn_mixed | white_-4    |                   0.1 |       0.801562 | 0.499409 |
| awgn_mixed | white_-4    |                   0.5 |       0.984375 | 0.051419 |
| awgn_mixed | white_-4    |                   1   |       0.996875 | 0.01261  |
| clean_only | clean       |                   0   |       0.780496 | 0.494613 |
| clean_only | clean       |                   0.1 |       0.965571 | 0.180066 |
| clean_only | clean       |                   0.5 |       1        | 0.001215 |
| clean_only | clean       |                   1   |       1        | 0.000311 |
| clean_only | white_-4    |                   0   |       0.450528 | 1.19037  |
| clean_only | white_-4    |                   0.1 |       0.634343 | 0.9637   |
| clean_only | white_-4    |                   0.5 |       0.600938 | 1.79361  |
| clean_only | white_-4    |                   1   |       0.584978 | 2.91136  |

残差倍率0仍保留融合LayerNorm，不等于原数据单分支。门控均值不是物理贡献率。干预后没有重新训练，不能把其结果当作新方法性能。

物理特征分析在当前训练集标准化坐标中报告加噪偏移、裁剪比例和训练类间/类内分离度。该诊断使用3个固定验证噪声种子，正式选模仍只用原6201。诊断没有自动去噪或按类修改特征。

## 阶段决策

进入固定嵌套小样本基线。之后最多分别检验残差缩放和数据辅助CE两个单变量候选，每个直接对比原融合。是否进入跨负载由预先规定的验证门槛决定，不追逐测试性能。
