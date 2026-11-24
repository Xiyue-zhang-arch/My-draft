# 小样本基线验证结果

108模型已完成来源验证选模，测试尚未读取。以下只用于诊断及后续候选筛选。

|   ratio | protocol   | variant         | condition   |   balanced_acc |   validation_nll |
|--------:|:-----------|:----------------|:------------|---------------:|-----------------:|
|     0.1 | awgn_mixed | baseline_data   | clean       |       0.696061 |         0.811719 |
|     0.1 | awgn_mixed | baseline_data   | white_+0    |       0.71872  |         0.81324  |
|     0.1 | awgn_mixed | baseline_data   | white_+2    |       0.708483 |         0.810675 |
|     0.1 | awgn_mixed | baseline_data   | white_+4    |       0.705011 |         0.809993 |
|     0.1 | awgn_mixed | baseline_data   | white_-2    |       0.719409 |         0.820363 |
|     0.1 | awgn_mixed | baseline_data   | white_-4    |       0.714709 |         0.836927 |
|     0.1 | awgn_mixed | baseline_fusion | clean       |       0.945312 |         0.342023 |
|     0.1 | awgn_mixed | baseline_fusion | white_+0    |       0.892361 |         0.390837 |
|     0.1 | awgn_mixed | baseline_fusion | white_+2    |       0.890625 |         0.380246 |
|     0.1 | awgn_mixed | baseline_fusion | white_+4    |       0.894097 |         0.379119 |
|     0.1 | awgn_mixed | baseline_fusion | white_-2    |       0.890625 |         0.414439 |
|     0.1 | awgn_mixed | baseline_fusion | white_-4    |       0.880364 |         0.446086 |
|     0.1 | clean_only | baseline_data   | clean       |       0.812231 |         0.661114 |
|     0.1 | clean_only | baseline_fusion | clean       |       0.988715 |         0.22725  |
|     0.2 | awgn_mixed | baseline_data   | clean       |       0.920139 |         0.312858 |
|     0.2 | awgn_mixed | baseline_data   | white_+0    |       0.921875 |         0.310851 |
|     0.2 | awgn_mixed | baseline_data   | white_+2    |       0.917535 |         0.308069 |
|     0.2 | awgn_mixed | baseline_data   | white_+4    |       0.919271 |         0.307916 |
|     0.2 | awgn_mixed | baseline_data   | white_-2    |       0.921007 |         0.319954 |
|     0.2 | awgn_mixed | baseline_data   | white_-4    |       0.913194 |         0.343142 |
|     0.2 | awgn_mixed | baseline_fusion | clean       |       0.994792 |         0.075201 |
|     0.2 | awgn_mixed | baseline_fusion | white_+0    |       0.930556 |         0.182556 |
|     0.2 | awgn_mixed | baseline_fusion | white_+2    |       0.935764 |         0.168187 |
|     0.2 | awgn_mixed | baseline_fusion | white_+4    |       0.939236 |         0.164006 |
|     0.2 | awgn_mixed | baseline_fusion | white_-2    |       0.922743 |         0.199862 |
|     0.2 | awgn_mixed | baseline_fusion | white_-4    |       0.904849 |         0.238828 |
|     0.2 | clean_only | baseline_data   | clean       |       0.96441  |         0.208244 |
|     0.2 | clean_only | baseline_fusion | clean       |       0.999132 |         0.03113  |
|     0.5 | awgn_mixed | baseline_data   | clean       |       1        |         0.033347 |
|     0.5 | awgn_mixed | baseline_data   | white_+0    |       1        |         0.039286 |
|     0.5 | awgn_mixed | baseline_data   | white_+2    |       1        |         0.03534  |
|     0.5 | awgn_mixed | baseline_data   | white_+4    |       1        |         0.033577 |
|     0.5 | awgn_mixed | baseline_data   | white_-2    |       0.998264 |         0.04791  |
|     0.5 | awgn_mixed | baseline_data   | white_-4    |       0.988715 |         0.066289 |
|     0.5 | awgn_mixed | baseline_fusion | clean       |       1        |         0.008273 |
|     0.5 | awgn_mixed | baseline_fusion | white_+0    |       0.982639 |         0.054904 |
|     0.5 | awgn_mixed | baseline_fusion | white_+2    |       0.991319 |         0.039432 |
|     0.5 | awgn_mixed | baseline_fusion | white_+4    |       0.987847 |         0.039946 |
|     0.5 | awgn_mixed | baseline_fusion | white_-2    |       0.977431 |         0.055663 |
|     0.5 | awgn_mixed | baseline_fusion | white_-4    |       0.96432  |         0.084072 |
|     0.5 | clean_only | baseline_data   | clean       |       1        |         0.010479 |
|     0.5 | clean_only | baseline_fusion | clean       |       0.999132 |         0.003325 |