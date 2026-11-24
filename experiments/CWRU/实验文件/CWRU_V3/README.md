# CWRU_V3

## 现存结果

30 个基础选参模型完成并锁定 λ 为 0.3，原选参批次以暂停状态留存。随机窗口、全样本跨负载及小样本跨负载均有已完成的独立批次。四种适配对比方法的小样本和全样本批次也均完成，见 [V3 结果目录](../../实验结果/CWRU_V3/)及[四方法汇总](../../实验结果/CWRU_V3/comparison_summary_20261103_201729/)。

## 复现与历史协议

2026-11-03按用户要求把四种对比方法收紧为**只做跨负载小样本、纯干净**，一次手动跑一个方法。入口`run_load_comparisons.py`，方案[COMPARISON_LOAD_PROTOCOL.md](COMPARISON_LOAD_PROTOCOL.md)，手动说明[对比实验手动运行说明.md](对比实验手动运行说明.md)，启动器`start_load_comparisons.ps1`。四方法WDCNN、DRSN-CW、PG-RDGFN-adapted、AW_Hybrid-adapted；每方法180个新模型（4来源×10/20/50%×3子集×5种子），合计720模型、2160个模型目标组合与2160行纯干净指标。数据划分、子集与种子同V3小样本，本文方法结果只读复用。一次运行只写一个新的`experiments/CWRU/实验结果/CWRU_V3/load_comparison_small_*`目录，训练选模锁定后统一测试并生成该方法报告；跨方法汇总由四个目录人工合并。真实状态见各目录config.json，准备代码不等于已启动。

同一入口用`-Sample full`（或`--sample full`）补跑**全样本纯干净**：每方法4来源×5种子=20个新模型、60个组合、60行指标，四方法合计80/240/240；报告不生成子集波动表，本文方法对照改取`load_full_20261024_0950`的纯干净部分。输出目录为`load_comparison_full_*`。

```powershell
cd 'D:\Project\MY draft'
# 一次跑完所有未完成的方法（已完成自动跳过，每个方法各写新目录）
pwsh -NoProfile -File '.\experiments\CWRU\实验文件\CWRU_V3\start_load_comparisons.ps1' -Method all
# 也可只跑单个方法
pwsh -NoProfile -File '.\experiments\CWRU\实验文件\CWRU_V3\start_load_comparisons.ps1' -Method WDCNN
```

2026-10-26原小样本进程在311个完整模型后退出，m0312在53轮中断。用户最新明确继续完成所有剩余模型，覆盖此前360暂停安排。恢复入口resume_load_small_experiment.py resume-all复用311完整模型，保留m0312现场并在新目录从头补训，再补至720后统一测试和报告。见[LOAD_SMALL_RESUME_PROTOCOL.md](LOAD_SMALL_RESUME_PROTOCOL.md)。只适用于此次精确中断状态，禁止用原run覆盖或重复训练。实际状态以原输出内config.json及continuation_request.json为准。

2026-10-25剩余跨负载小样本使用run_load_small_experiment.py，见[LOAD_SMALL_PROTOCOL.md](LOAD_SMALL_PROTOCOL.md)。固定已选λ=0.3，四来源、三预算10%/20%/50%、三个既有嵌套子集、两协议、两方法、五种子，共720新模型。全部选模后统一测试2160组合、29160指标并生成报告后结束。只写新的experiments/CWRU/实验结果/CWRU_V3/load_small_*目录，旧选参、随机窗口、全样本来源结果不重训、不重测、不覆盖。准备或启动不等于完成，实际状态见新config.json。

```powershell
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_load_small_experiment.py plan
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_load_small_experiment.py run --output 'D:/Project/MY draft/experiments/CWRU/实验结果/CWRU_V3/load_small_新的时间戳'
```

2026-10-24全样本负载迁移使用run_load_experiment.py，见[LOAD_PROTOCOL.md](LOAD_PROTOCOL.md)。固定λ=0.3，两协议、两方法、四来源各五种子，共80新模型，全部选模后测试12方向240组合、3240指标并结束。不执行小样本。输出必须为新的experiments/CWRU/实验结果/CWRU_V3/load_full_*目录。

2026-10-23随机窗口续跑使用独立入口run_window_experiment.py，协议见[WINDOW_PROTOCOL.md](WINDOW_PROTOCOL.md)。固定已选λ=0.3，仅新增全样本AWGN10模型及三预算小样本两协议180模型，另只读复用10个全样本干净模型及预测。合计200模型、2700指标，全部新模型选模后测试并生成报告，然后结束，不启动跨负载。新结果须位于experiments/CWRU/实验结果/CWRU_V3/window_random_*全新直接子目录。不要恢复或删除旧目录中的m0031控制文件。

```powershell
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_window_experiment.py plan
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_window_experiment.py run --output 'D:/Project/MY draft/experiments/CWRU/实验结果/CWRU_V3/window_random_新的时间戳'
```

2026-10-22最新运行安排为30个基础选参模型完成后锁定λ、测试这30模型并暂停。已启动的旧完整流程由selection_control.py执行阶段控制，models/m0031是阻止后续训练的普通控制文件，不是模型。selection_test_request.json的watching仅表示等待，须selection_test_status=complete且stage=paused才算本批测试完成并暂停。恢复需用户明确继续，不能重跑本批预测或直接重启原run命令。下文1020模型为整个方案规模，不表示今天继续全部阶段。

正式设置见[预定协议](PROTOCOL.md)。本目录复制V2当前的两种方法并独立实现V3调度、基础选参、固定预算和统一测试。底层骨干、物理特征和MAT读取组件仍只读复用，不调用V2训练入口。

使用D:/anaconda/envs/pytorch_GPU/python.exe。默认命令只打印计划。

```powershell
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_experiment.py plan
& 'D:/anaconda/envs/pytorch_GPU/python.exe' -X utf8 -B experiments/CWRU/实验文件/CWRU_V3/run_experiment.py run --output 'D:/Project/MY draft/experiments/CWRU/实验结果/CWRU_V3/run_新的时间戳'
```

run拒绝已有输出目录。先30模型完成基础选参，再固定λ完成其余990模型。主实验1000模型全部选模后统一测试，另20个敏感性模型只验证。失败保留现场并停止，不能再次执行同输出以自动重试。不得依据测试改λ或预算。

config.json记录真实阶段、已完成模型数、当前模型、λ锁定和测试访问状态。training_console.log及每模型training.csv记录进度。启动不是完成；最多200轮也不表示每模型实际跑满200轮。临时检查不计为正式训练。
