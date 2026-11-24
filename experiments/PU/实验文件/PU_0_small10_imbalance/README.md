# 10%小样本及训练类别不平衡

从四工况7∶2∶1父实验各集合按工况/轴承取10%窗口，训练1120、验证320、测试160。再建训练健康280、外圈28、内圈28、复合28的10∶1∶1∶1协议，及每类91、共364训练窗的等量均衡对照。三个协议共用均衡验证和测试。原记录划分及模型保持不变，详情见PROTOCOL.md。

三协议六方法五种子90新模型，全部50轮末轮。1120训练窗每模型900更新，364窗每模型300更新。全部完成核验后统一7560条件预测并CPU独立审计。有限退化结果保留，异常停止不自动恢复，旧实验保留。

仓库根目录运行，拒绝覆盖已有输出。

以下为四谐波版入口。首次运行 `prepare.py` 时，新描述符和三套训练集的标准化参数写入 `data/harmonic_calibration`。请将 `新的时间戳` 替换为未使用的名称。2026-11-16 冻结结果使用旧描述符，见[历史完成标记](../../实验结果/PU_0_small10_imbalance/main_20261116_171613/RUN_COMPLETE.json)。

```powershell
$env:PYTHONUTF8='1'
$env:PATH='D:\anaconda\envs\pytorch_GPU;D:\anaconda\envs\pytorch_GPU\Library\bin;D:\anaconda\envs\pytorch_GPU\Scripts;'+$env:PATH
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_small10_imbalance/prepare.py
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_small10_imbalance/check.py --output experiments/PU/实验文件/PU_0_small10_imbalance/checks/replay_新的时间戳
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_small10_imbalance/run.py --preflight experiments/PU/实验文件/PU_0_small10_imbalance/checks/replay_新的时间戳/preflight.json --output experiments/PU/实验结果/PU_0_small10_imbalance/replay_新的时间戳
```

bootstrap仅首次建立数值代码副本，定稿后不要重跑。数据清单见data/prepared/split_class_counts.csv、selected_records.csv、parent_indices.npz和windows.json。活动状态以结果config.json为准，完成需RUN_COMPLETE.json及独立审计通过。持续跟进到完成并主动汇报。

2026-11-16 17:27:20完成并冻结，90模型、7560评价及CPU独立审计齐全，41317旧文件和8归档一致，运行锁释放。结果评审见ASSESSMENT.md，完整三协议结果及各类召回、逐种子配对差见结果report目录。不平衡物理干净/受扰BA86.62/81.54%，等权81.88/77.37%，差4.7500/4.1750个百分点、均5/5正差；受扰物理比替代频率低0.2438个百分点，不能称正确频率稳定特异优势。等量均衡物理97.50/93.09%，不平衡仍明显退化。已持续跟进完成后主动报告。
