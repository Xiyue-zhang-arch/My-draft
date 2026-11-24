# 运行和复核

`PROTOCOL.md` 说明倍次数、谱点保护、训练配对和预先固定的汇总规则。`run_mean.py` 生成实际前景和背景栅格，训练四个来源的五个种子，保存权重、训练轨迹和目标预测。新运行使用独立输出目录。

```powershell
$puPython = 'D:/anaconda/envs/pytorch_GPU/python.exe'
$puMean = 'experiments/PU/实验文件/PU_harmonic_mean'
$puRun = 'experiments/PU/实验结果/PU_harmonic_mean/replay'
& $puPython -B "$puMean/run_mean.py" --output "$puMean/preflight_replay" --preflight
& $puPython -B "$puMean/run_mean.py" --output $puRun
& $puPython -B "$puMean/audit_mean.py" --output $puRun
```

`frequency_masks.json` 记录分析范围、倍次数、前景、背景和实际移除的重复栅格。`bank` 记录按来源拟合的描述符和定标。`report` 给出四种方法的总体、逐路线、逐来源、逐目标、逐种子和逐噪声条件结果。重放使用已冻结的四谐波窗口顺序和训练记录。四谐波的 1260 份预测可以复算。普通融合、V2、三四倍候选使用四谐波正式报告内已核验的 3780 行逐条件指标。

复算冻结结果时，`--output` 指向 `main_20261122_a`，另加 `--report-dir` 指向一个新目录。
