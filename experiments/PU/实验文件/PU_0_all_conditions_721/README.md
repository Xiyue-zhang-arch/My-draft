# PU_0四工况混合7∶2∶1实验

四个工况均进入训练、验证和测试。按工况及轴承分层，完整去重记录组不跨集合，窗口数严格11200/3200/1600。每工况2800/800/400，每类每工况700/200/100。各记录文件与重复别名的分配和取窗数见data/prepared/records_inventory.csv，逐类分配见split_class_counts.csv。完整说明见PROTOCOL.md。

保持原六组PU_0模型、五种子1706至1710、Adam0.001、batch64、50轮末轮。数据量增加后每模型8750更新。30个模型完成与核验后统一测试四工况84条件，2520预测并自动独立CPU审计。旧实验不变，不按测试追加范围。

在仓库根目录执行，输出已存在时拒绝覆盖。

以下为四谐波版入口。数据仍使用现有记录划分。首次运行 `prepare.py` 时，新描述符和标准化参数写入 `data/harmonic_calibration`。请将 `新的时间戳` 替换为未使用的名称。2026-11-16 冻结结果使用旧描述符，见[历史完成标记](../../实验结果/PU_0_all_conditions_721/main_20261116_134258/RUN_COMPLETE.json)。

```powershell
$env:PYTHONUTF8 = '1'
$env:PATH = 'D:\anaconda\envs\pytorch_GPU;D:\anaconda\envs\pytorch_GPU\Library\bin;D:\anaconda\envs\pytorch_GPU\Scripts;' + $env:PATH
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_all_conditions_721/prepare.py
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_all_conditions_721/check.py --output experiments/PU/实验文件/PU_0_all_conditions_721/checks/replay_新的时间戳
& 'D:\anaconda\envs\pytorch_GPU\python.exe' -B experiments/PU/实验文件/PU_0_all_conditions_721/run.py --preflight experiments/PU/实验文件/PU_0_all_conditions_721/checks/replay_新的时间戳/preflight.json --output experiments/PU/实验结果/PU_0_all_conditions_721/replay_新的时间戳
```

bootstrap.py仅首次建立原数值计算代码副本，定稿后不得重跑。功能验收的六个单轮模型不用于正式训练。有限零梯度等退化结果保留，NaN/Inf及身份错误停止，正式异常保留现场不自动恢复。结果状态以config.json和RUN_COMPLETE.json为准。

2026-11-16 14:45:17已完成并冻结。30正式模型、2520预测及40320混淆单元齐全，独立审计通过，35936旧文件和8归档一致，运行锁释放。结果评审见ASSESSMENT.md；逐工况、逐种子及配对差见结果report目录。物理受扰BA91.83±1.23%，等权89.56±0.75%，差2.2650个百分点、4/5正差；减替代频率仅0.5256个百分点、3/5正差，减自身0.8831个百分点、仅2/5正差，不支持稳定机制特异收益。多数方法干净100%，存在上限效应。未新增网络推理。
