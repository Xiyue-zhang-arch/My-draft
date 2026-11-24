# CWRU_V1 新基线

代码使用普通交叉熵，标签平滑0，最多100epoch，每epoch结束验证选模，连续20epoch未改善早停。两种划分均全部40记录、全样本，不对齐训练预算。固定5个训练种子，融合与数据单分支、纯干净与AWGN混合训练，共100模型。仅基线，无结构探索。

[完整协议](PROTOCOL.md)

本轮 100 个模型已完成，见[完成配置](../../实验结果/CWRU_V1/baseline_20261020/config.json)与[数据用量](../../实验结果/CWRU_V1/baseline_20261020/DATA_USAGE.md)。全部 40 条记录的随机窗口划分为训练 481、验证 103、测试 103。按负载来源分别用 73、156、160、160 窗训练。

从项目根目录执行以下入口时会启动正式训练，结果只写新的独立目录，不覆盖历史成果。

```powershell
D:\anaconda\envs\pytorch_GPU\python.exe -B experiments/CWRU/实验文件/CWRU_V1/run_baseline.py
```

增加--prepare-only只保存固定划分、打印复杂度，不训练模型。准备目录必须尚无models且stage=prepared，才可使用--prepared启动一次；不支持覆盖或自动续训已有模型。

模型实现只读复用experiments/CWRU/实验文件/CWRU_V0/shared的基础网络和212维物理特征。训练、选模、划分和报告均使用本目录实现，不调用旧标签平滑或固定更新数训练逻辑。
